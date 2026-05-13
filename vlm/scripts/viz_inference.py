"""
Visualise model predictions side by side with receipt images.

Runs inference on N samples from the CORD test split and produces
an HTML report showing the receipt image, ground truth JSON, and
model prediction for each sample.

Usage:
    python -m vlm.scripts.viz_inference -c b4_e25_drop05
    python -m vlm.scripts.viz_inference -c b4_e25_drop05 --n 20 --stage rl
    python -m vlm.scripts.viz_inference -c b4_e25_drop05 --stage both --output report.html
    python -m vlm.scripts.viz_inference -c b4_e25_drop05 --index 5
"""

import argparse
import base64
import io
import json
from pathlib import Path

import torch
from datasets import load_dataset

from vlm.configs.training_configs import get_training_config
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.common import build_instruction, prepare_tokenizer
from vlm.training.generate import generate_k_outputs
from vlm.training.rewards import compute_reward
from vlm.utils.device import get_device
from vlm.utils.training import set_seed


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────

def load_model(cfg, checkpoint_path: str | Path, device):
    model = ReceiptVLM(
        device=device,
        vision_model_name=cfg.vision.model_name,
        default_vision_processor=cfg.vision.default_processor,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
        cross_attention_projector=cfg.projector.cross_attention,
        cross_attention_projector_num_queries=cfg.projector.num_queries,
        cross_attention_projector_num_heads=cfg.projector.num_heads,
        cross_attention_projector_num_layers=cfg.projector.num_layers,
        cross_attention_projector_ffn_mult=cfg.projector.ffn_mult,
        projector_mult=cfg.projector.projector_mult,
        projector_dropout=None,  # no dropout at inference
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.projector.load_state_dict(ckpt["projector_state_dict"])
    model.projector.eval()
    return model


def predict(model, image, tokenizer, instruction, cfg) -> str:
    with torch.no_grad():
        gen = generate_k_outputs(
            model=model,
            image=image,
            tokenizer=tokenizer,
            instruction=instruction,
            k=1,
            max_completion_tokens=cfg.eval.max_completion_tokens,
            temperature=cfg.eval.temperature,
            do_sample=False,
        )
    return gen.texts[0]


def image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except Exception:
        return text


# ─────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f5; color: #1a1a1a; margin: 0; padding: 24px; }}
  h1   {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }}
  .meta {{ font-size: 0.85rem; color: #666; margin-bottom: 24px; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
           margin-bottom: 24px; overflow: hidden; }}
  .card-header {{ background: #f0f0f0; padding: 10px 16px;
                  font-size: 0.8rem; font-weight: 600; color: #444;
                  display: flex; justify-content: space-between; align-items: center; }}
  .reward {{ font-size: 0.75rem; padding: 2px 8px; border-radius: 12px;
             font-weight: 700; }}
  .reward-good  {{ background: #d4edda; color: #155724; }}
  .reward-mid   {{ background: #fff3cd; color: #856404; }}
  .reward-bad   {{ background: #f8d7da; color: #721c24; }}
  .columns {{ display: grid; grid-template-columns: {col_template}; gap: 0; }}
  .col {{ padding: 16px; border-right: 1px solid #eee; }}
  .col:last-child {{ border-right: none; }}
  .col-label {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: .05em; color: #888; margin-bottom: 8px; }}
  .col img {{ width: 100%; border-radius: 4px; display: block; }}
  pre {{ margin: 0; font-size: 0.72rem; line-height: 1.55;
         white-space: pre-wrap; word-break: break-word;
         background: #f8f8f8; border-radius: 4px; padding: 10px; }}
  .match    {{ color: #155724; }}
  .mismatch {{ color: #721c24; }}
  .format-ok  {{ color: #155724; font-weight: 700; }}
  .format-bad {{ color: #721c24; font-weight: 700; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{meta}</p>
{cards}
</body>
</html>
"""

CARD_BOTH = """\
<div class="card">
  <div class="card-header">
    <span>Sample {idx}</span>
    <span>
      SFT: <span class="reward {sft_cls}">{sft_reward:.3f}</span>
      &nbsp;
      RL: <span class="reward {rl_cls}">{rl_reward:.3f}</span>
    </span>
  </div>
  <div class="columns">
    <div class="col">
      <div class="col-label">Receipt image</div>
      <img src="data:image/jpeg;base64,{img_b64}" alt="receipt">
    </div>
    <div class="col">
      <div class="col-label">Ground truth</div>
      <pre>{gt}</pre>
    </div>
    <div class="col">
      <div class="col-label">SFT prediction
        <span class="{sft_fmt_cls}">{sft_fmt_label}</span>
      </div>
      <pre>{sft_pred}</pre>
    </div>
    <div class="col">
      <div class="col-label">RL prediction
        <span class="{rl_fmt_cls}">{rl_fmt_label}</span>
      </div>
      <pre>{rl_pred}</pre>
    </div>
  </div>
</div>
"""

CARD_SINGLE = """\
<div class="card">
  <div class="card-header">
    <span>Sample {idx}</span>
    <span class="reward {reward_cls}">{stage_label} reward: {reward:.3f}</span>
  </div>
  <div class="columns">
    <div class="col">
      <div class="col-label">Receipt image</div>
      <img src="data:image/jpeg;base64,{img_b64}" alt="receipt">
    </div>
    <div class="col">
      <div class="col-label">Ground truth</div>
      <pre>{gt}</pre>
    </div>
    <div class="col">
      <div class="col-label">{stage_label} prediction
        <span class="{fmt_cls}">{fmt_label}</span>
      </div>
      <pre>{pred}</pre>
    </div>
  </div>
</div>
"""


def reward_cls(r: float) -> str:
    if r >= 0.75: return "reward-good"
    if r >= 0.50: return "reward-mid"
    return "reward-bad"


def fmt_cls_label(pred: str):
    try:
        import json as _json
        p = pred.strip()
        _json.loads(p)
        ok = p.startswith("{") and p.endswith("}")
        return ("format-ok", "✓ JSON") if ok else ("format-mid", "~ JSON")
    except Exception:
        return ("format-bad", "✗ not JSON")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise model predictions")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--index", type=int, default=None,
                        help="Single sample index (overrides --n)")
    parser.add_argument("--stage", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: viz_<config>_<stage>.html)")
    args = parser.parse_args()

    cfg = get_training_config(args.config)
    device = get_device()
    set_seed(42)

    print(f"config: {args.config}")
    print(f"stage:  {args.stage}")
    print(f"device: {device}")

    # Dataset
    print("loading dataset...")
    dataset = CORDDataset(
        split=cfg.data.test_split,
        max_samples=cfg.data.test_samples,
        dataset_name=cfg.data.dataset_name,
    )

    if args.index is not None:
        indices = [args.index]
    else:
        indices = list(range(min(args.n, len(dataset))))

    print(f"samples: {len(indices)}")

    # Tokenizer + instruction
    stages_to_run = ["sft", "rl"] if args.stage == "both" else [args.stage]

    # Load models
    models = {}
    tokenizers = {}
    instructions = {}

    for stage in stages_to_run:
        ckpt_path = (cfg.sft_best_checkpoint if stage == "sft"
                     else cfg.rl_best_checkpoint)
        if not Path(ckpt_path).exists():
            print(f"[warn] {stage} checkpoint not found: {ckpt_path} — skipping")
            continue
        print(f"loading {stage} checkpoint: {ckpt_path}")
        model = load_model(cfg, ckpt_path, device)
        tok = prepare_tokenizer(model.lm.tokenizer)
        instr = build_instruction(tok, cfg.model.instruction)
        models[stage] = model
        tokenizers[stage] = tok
        instructions[stage] = instr

    # Run inference
    cards_html = []
    for i, idx in enumerate(indices):
        print(f"  [{i+1}/{len(indices)}] sample {idx}")
        sample = dataset[idx]
        image = sample["image"]
        gt = sample["label"]
        img_b64 = image_to_b64(image)
        gt_pretty = pretty_json(gt)

        if args.stage == "both" and "sft" in models and "rl" in models:
            sft_pred = predict(models["sft"], image, tokenizers["sft"],
                               instructions["sft"], cfg)
            rl_pred  = predict(models["rl"],  image, tokenizers["rl"],
                               instructions["rl"],  cfg)
            sft_r = compute_reward(sft_pred, gt).total
            rl_r  = compute_reward(rl_pred,  gt).total
            sft_fc, sft_fl = fmt_cls_label(sft_pred)
            rl_fc,  rl_fl  = fmt_cls_label(rl_pred)
            cards_html.append(CARD_BOTH.format(
                idx=idx,
                img_b64=img_b64,
                gt=gt_pretty,
                sft_pred=pretty_json(sft_pred),
                rl_pred=pretty_json(rl_pred),
                sft_reward=sft_r, sft_cls=reward_cls(sft_r),
                rl_reward=rl_r,   rl_cls=reward_cls(rl_r),
                sft_fmt_cls=sft_fc, sft_fmt_label=sft_fl,
                rl_fmt_cls=rl_fc,   rl_fmt_label=rl_fl,
            ))
        else:
            stage = list(models.keys())[0]
            pred = predict(models[stage], image, tokenizers[stage],
                           instructions[stage], cfg)
            r = compute_reward(pred, gt).total
            fc, fl = fmt_cls_label(pred)
            cards_html.append(CARD_SINGLE.format(
                idx=idx,
                img_b64=img_b64,
                gt=gt_pretty,
                pred=pretty_json(pred),
                reward=r, reward_cls=reward_cls(r),
                stage_label=stage.upper(),
                fmt_cls=fc, fmt_label=fl,
            ))

    # Render
    col_template = "220px 1fr 1fr 1fr" if args.stage == "both" else "220px 1fr 1fr"
    title = f"Receipt VLM — {args.config} / {args.stage}"
    meta = (f"{len(indices)} samples · config: {args.config} · "
            f"stage: {args.stage} · "
            f"image: {cfg.vision.image_height}×{cfg.vision.image_width}")

    html = HTML_TEMPLATE.format(
        title=title,
        meta=meta,
        col_template=col_template,
        cards="\n".join(cards_html),
    )

    out_path = args.output or f"viz_{args.config}_{args.stage}.html"
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"\nsaved: {out_path}")
    print(f"open in browser: file://{Path(out_path).resolve()}")


if __name__ == "__main__":
    main()