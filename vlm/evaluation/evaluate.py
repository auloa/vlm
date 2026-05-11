import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vlm.configs.training_schema import TrainingConfig
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.common import (
    build_instruction,
    ensure_dir,
    prepare_tokenizer,
    set_projector_only_trainable,
)
from vlm.training.rewards import compute_reward
from vlm.utils.device import get_device
from vlm.utils.json_extractor import extract_json_object
from vlm.utils.training import set_seed

REQUIRED_KEYS = {"line_items", "total"}


@dataclass
class SampleEval:
    index: int
    ground_truth: str
    prediction: str
    extracted_json: str | None
    strict_json_valid: bool
    extractable_json_valid: bool
    has_required_keys: bool
    line_items_is_list: bool
    total_present: bool
    format_adherent: bool
    reward: float
    total_match: bool


@dataclass
class EvalSummary:
    stage: str
    checkpoint_path: str
    num_samples: int
    strict_json_rate: float
    extractable_json_rate: float
    required_keys_rate: float
    line_items_list_rate: float
    total_present_rate: float
    format_adherence_rate: float
    total_match_rate: float
    mean_reward: float


def evaluate_checkpoint(
    cfg: TrainingConfig,
    checkpoint_path: str | Path,
    stage: str,
    num_samples: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    do_sample: bool = False,
) -> EvalSummary:
    """Evaluate one projector checkpoint on the held-out test split.

    The main assignment metric is format_adherence_rate:
        prediction is exact valid JSON
        AND contains required keys: line_items and total
        AND line_items is a list
    """
    device = get_device()
    set_seed(42)

    checkpoint_path = Path(checkpoint_path)
    results_dir = ensure_dir(cfg.results_dir)
    eval_dir = ensure_dir(results_dir / f"eval_{stage}")

    max_samples = num_samples or cfg.eval.num_samples
    max_completion_tokens = max_completion_tokens or cfg.eval.max_completion_tokens
    temperature = temperature if temperature is not None else cfg.eval.temperature

    print(f"evaluating stage: {stage}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")

    model = _load_model_with_projector(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    tokenizer = prepare_tokenizer(model.lm.tokenizer)
    instruction = build_instruction(tokenizer, cfg.model.instruction)

    dataset = CORDDataset(
        split=cfg.data.test_split,
        max_samples=max_samples,
        dataset_name=cfg.data.dataset_name,
        tokenizer=tokenizer,
        max_target_length=cfg.sft.max_target_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_eval_collate,
    )

    samples: list[SampleEval] = []

    with SummaryWriter(log_dir=str(cfg.root_dir / "runs" / f"eval_{stage}")) as writer:
        for index, batch in enumerate(loader):
            image = batch["image"][0]
            ground_truth = batch["label"][0]

            prediction = generate_one(
                model=model,
                image=image,
                tokenizer=tokenizer,
                instruction=instruction,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

            sample_eval = score_prediction(
                index=index,
                prediction=prediction,
                ground_truth=ground_truth,
            )
            samples.append(sample_eval)

            writer.add_scalar(
                "eval/format_adherent",
                float(sample_eval.format_adherent),
                index,
            )
            writer.add_scalar(
                "eval/reward",
                sample_eval.reward,
                index,
            )

            if index < 10:
                writer.add_text(
                    f"eval/sample_{index}",
                    _format_sample_markdown(sample_eval),
                    index,
                )

    summary = summarize_samples(
        stage=stage,
        checkpoint_path=checkpoint_path,
        samples=samples,
    )

    _save_json(eval_dir / "summary.json", asdict(summary))
    _save_jsonl(eval_dir / "samples.jsonl", [asdict(sample) for sample in samples])
    _save_markdown_report(eval_dir / "samples.md", summary, samples)

    print_summary(summary)
    print(f"saved summary: {eval_dir / 'summary.json'}")
    print(f"saved samples: {eval_dir / 'samples.jsonl'}")
    print(f"saved report:  {eval_dir / 'samples.md'}")

    return summary


def compare_sft_and_rl(
    cfg: TrainingConfig,
    num_samples: int | None = None,
    max_completion_tokens: int | None = None,
) -> None:
    """Evaluate SFT and RL checkpoints and save a comparison summary."""
    summaries = []

    summaries.append(
        evaluate_checkpoint(
            cfg=cfg,
            checkpoint_path=cfg.sft_best_checkpoint,
            stage="sft",
            num_samples=num_samples,
            max_completion_tokens=max_completion_tokens,
            do_sample=False,
        )
    )

    summaries.append(
        evaluate_checkpoint(
            cfg=cfg,
            checkpoint_path=cfg.rl_best_checkpoint,
            stage="rl",
            num_samples=num_samples,
            max_completion_tokens=max_completion_tokens,
            do_sample=False,
        )
    )

    comparison = {
        summary.stage: asdict(summary)
        for summary in summaries
    }

    results_dir = ensure_dir(cfg.results_dir)
    _save_json(results_dir / "eval_comparison.json", comparison)

    print("\ncomparison")
    print("-" * 80)
    for summary in summaries:
        print(
            f"{summary.stage:>4} | "
            f"format={summary.format_adherence_rate:.1%} | "
            f"strict_json={summary.strict_json_rate:.1%} | "
            f"required_keys={summary.required_keys_rate:.1%} | "
            f"mean_reward={summary.mean_reward:.3f}"
        )

    print(f"\nsaved comparison: {results_dir / 'eval_comparison.json'}")


def _load_model_with_projector(
    cfg: TrainingConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> ReceiptVLM:
    model = ReceiptVLM(
        device=device,
        vision_model_name=cfg.vision.model_name,
        default_vision_processor=cfg.vision.default_processor,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.projector.load_state_dict(checkpoint["projector_state_dict"])

    set_projector_only_trainable(model)
    model.eval()

    return model


def generate_one(
    model: ReceiptVLM,
    image: Image.Image,
    tokenizer,
    instruction: str,
    max_completion_tokens: int,
    temperature: float,
    do_sample: bool,
) -> str:
    device = model.device

    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = prompt_tokens["input_ids"].to(device)
    attention_mask = prompt_tokens["attention_mask"].to(device)

    with torch.no_grad():
        inputs_embeds, full_attention_mask = model.prepare_inputs_embeds(
            images=[image],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention_mask,
            "max_length": inputs_embeds.shape[1] + max_completion_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": True,
            "repetition_penalty": 1.1,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.95

        output = model.lm.model.generate(**generation_kwargs)

    return tokenizer.decode(
        output.sequences[0],
        skip_special_tokens=True,
    ).strip()


def score_prediction(
    index: int,
    prediction: str,
    ground_truth: str,
) -> SampleEval:
    strict_parsed = _loads_exact_json(prediction)
    extracted = extract_json_object(prediction)
    extracted_parsed = _loads_exact_json(extracted) if extracted is not None else None

    parsed_for_schema = strict_parsed if strict_parsed is not None else extracted_parsed

    strict_json_valid = strict_parsed is not None
    extractable_json_valid = extracted_parsed is not None

    has_required_keys = _has_required_keys(parsed_for_schema)
    line_items_is_list = _line_items_is_list(parsed_for_schema)
    total_present = _total_present(parsed_for_schema)

    # This is the main assignment metric.
    # It requires the model output itself to be strict JSON, not JSON buried in prose.
    format_adherent = (
        strict_json_valid
        and has_required_keys
        and line_items_is_list
        and total_present
    )

    reward = compute_reward(prediction, ground_truth)
    total_match = _total_matches(parsed_for_schema, ground_truth)

    return SampleEval(
        index=index,
        ground_truth=ground_truth,
        prediction=prediction,
        extracted_json=extracted,
        strict_json_valid=strict_json_valid,
        extractable_json_valid=extractable_json_valid,
        has_required_keys=has_required_keys,
        line_items_is_list=line_items_is_list,
        total_present=total_present,
        format_adherent=format_adherent,
        reward=reward.total,
        total_match=total_match,
    )


def summarize_samples(
    stage: str,
    checkpoint_path: Path,
    samples: list[SampleEval],
) -> EvalSummary:
    n = max(1, len(samples))

    return EvalSummary(
        stage=stage,
        checkpoint_path=str(checkpoint_path),
        num_samples=len(samples),
        strict_json_rate=sum(s.strict_json_valid for s in samples) / n,
        extractable_json_rate=sum(s.extractable_json_valid for s in samples) / n,
        required_keys_rate=sum(s.has_required_keys for s in samples) / n,
        line_items_list_rate=sum(s.line_items_is_list for s in samples) / n,
        total_present_rate=sum(s.total_present for s in samples) / n,
        format_adherence_rate=sum(s.format_adherent for s in samples) / n,
        total_match_rate=sum(s.total_match for s in samples) / n,
        mean_reward=mean([s.reward for s in samples]) if samples else 0.0,
    )


def print_summary(summary: EvalSummary) -> None:
    print("\nsummary")
    print("-" * 80)
    print(f"stage:                 {summary.stage}")
    print(f"num samples:           {summary.num_samples}")
    print(f"strict JSON rate:      {summary.strict_json_rate:.1%}")
    print(f"extractable JSON rate: {summary.extractable_json_rate:.1%}")
    print(f"required keys rate:    {summary.required_keys_rate:.1%}")
    print(f"line_items list rate:  {summary.line_items_list_rate:.1%}")
    print(f"total present rate:    {summary.total_present_rate:.1%}")
    print(f"FORMAT ADHERENCE RATE: {summary.format_adherence_rate:.1%}")
    print(f"total match rate:      {summary.total_match_rate:.1%}")
    print(f"mean reward:           {summary.mean_reward:.3f}")


def _loads_exact_json(text: str | None):
    if text is None:
        return None

    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None


def _has_required_keys(parsed) -> bool:
    return isinstance(parsed, dict) and REQUIRED_KEYS.issubset(parsed.keys())


def _line_items_is_list(parsed) -> bool:
    return isinstance(parsed, dict) and isinstance(parsed.get("line_items"), list)


def _total_present(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False

    total = parsed.get("total")
    return total is not None and str(total).strip() != ""


def _total_matches(parsed, ground_truth: str) -> bool:
    if not isinstance(parsed, dict):
        return False

    try:
        gt = json.loads(ground_truth)
    except json.JSONDecodeError:
        return False

    pred_total = _normalize_numberish(parsed.get("total", ""))
    gt_total = _normalize_numberish(gt.get("total", ""))

    return bool(pred_total and gt_total and pred_total == gt_total)


def _normalize_numberish(value) -> str:
    return re.sub(r"[^0-9]", "", str(value).lower().strip())


def _eval_collate(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "label": [sample["label"] for sample in batch],
    }


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_markdown_report(
    path: Path,
    summary: EvalSummary,
    samples: list[SampleEval],
    max_samples: int = 20,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Evaluation Report: {summary.stage}",
        "",
        "## Summary",
        "",
        f"- Checkpoint: `{summary.checkpoint_path}`",
        f"- Number of samples: {summary.num_samples}",
        f"- Strict JSON rate: {summary.strict_json_rate:.1%}",
        f"- Extractable JSON rate: {summary.extractable_json_rate:.1%}",
        f"- Required keys rate: {summary.required_keys_rate:.1%}",
        f"- Format adherence rate: **{summary.format_adherence_rate:.1%}**",
        f"- Total match rate: {summary.total_match_rate:.1%}",
        f"- Mean reward: {summary.mean_reward:.3f}",
        "",
        "## Sample Outputs",
        "",
    ]

    for sample in samples[:max_samples]:
        lines.append(_format_sample_markdown(sample))
        lines.append("\n---\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def _format_sample_markdown(sample: SampleEval) -> str:
    return (
        f"### Sample {sample.index}\n\n"
        f"- Format adherent: `{sample.format_adherent}`\n"
        f"- Strict JSON valid: `{sample.strict_json_valid}`\n"
        f"- Extractable JSON valid: `{sample.extractable_json_valid}`\n"
        f"- Required keys: `{sample.has_required_keys}`\n"
        f"- Reward: `{sample.reward:.3f}`\n"
        f"- Total match: `{sample.total_match}`\n\n"
        f"**Ground truth**\n\n"
        f"```json\n{sample.ground_truth}\n```\n\n"
        f"**Prediction**\n\n"
        f"```text\n{sample.prediction[:1200]}\n```\n"
    )