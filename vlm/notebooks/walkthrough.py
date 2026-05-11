"""Walkthrough notebook for the receipt VLM.

What this shows:
1. Where the parameters live and what's frozen
2. The shape transformations from image to LM input
3. SFT vs RL generation on the same image
4. Reward function breakdown on those outputs

Run from the repo root:
    uv run marimo edit notebooks/walkthrough.py

The notebook is robust to the RL checkpoint not existing yet — it will
just show SFT in that case.
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Receipt VLM — Walkthrough

    Frozen Donut encoder + trainable projector + frozen TinyLlama, trained
    with SFT then aligned with group-relative REINFORCE + KL.

    This notebook walks through:
    1. Parameter counts and what's trained
    2. Tensor shapes from image to LM input
    3. SFT vs RL generation on a sample receipt
    4. Reward decomposition for the same outputs
    """)
    return


@app.cell
def _():
    import torch

    from vlm.configs.training_configs import get_training_config
    from vlm.data.dataset import CORDDataset
    from vlm.models.receipt_vlm import ReceiptVLM
    from vlm.training.common import build_instruction, prepare_tokenizer
    from vlm.training.generate import generate_k_outputs
    from vlm.training.rewards import compute_reward
    from vlm.utils.device import get_device

    cfg = get_training_config("receipt-base")
    device = get_device()
    return (
        CORDDataset,
        ReceiptVLM,
        build_instruction,
        cfg,
        compute_reward,
        device,
        generate_k_outputs,
        prepare_tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(ReceiptVLM, cfg, device, mo, prepare_tokenizer):
    model = ReceiptVLM(
        device=device,
        vision_model_name=cfg.vision.model_name,
        default_vision_processor=cfg.vision.default_processor,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
    )
    tokenizer = prepare_tokenizer(model.lm.tokenizer)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mo.md(
        f"""
        ## Parameter counts

        | Component | Params | Trainable |
        |---|---|---|
        | Donut encoder | {sum(p.numel() for p in model.vision_encoder.parameters()):,} | frozen |
        | Projector (MLP+LN) | {sum(p.numel() for p in model.projector.parameters()):,} | ✅ |
        | TinyLlama | {sum(p.numel() for p in model.lm.parameters()):,} | frozen |
        | **Total** | **{total:,}** | **{trainable:,} ({100*trainable/total:.2f}%)** |
        """
    )
    return model, tokenizer


@app.cell(hide_code=True)
def _(CORDDataset, cfg, mo):
    val_dataset = CORDDataset(
        split=cfg.data.val_split,
        max_samples=20,
        dataset_name=cfg.data.dataset_name,
        prefer_high_resolution=True,
    )

    sample = val_dataset[0]
    image = sample["image"]
    ground_truth = sample["label"]

    mo.md(f"## Sample receipt\n\nGround truth label:\n\n```json\n{ground_truth}\n```")
    return ground_truth, image


@app.cell
def _(image, mo):
    mo.image(image, width=400)
    return


@app.cell
def _(build_instruction, cfg, image, mo, model, tokenizer, torch):
    # Architecture walkthrough: show the shape at each step.

    # 1. Image -> Donut features.
    with torch.no_grad():
        visual_features = model.vision_encoder([image])

    # 2. Visual features -> projector -> visual tokens in LM space.
    with torch.no_grad():
        visual_tokens = model.projector(visual_features)

    # 3. Instruction -> chat-templated -> tokenized -> embedded.
    instruction = build_instruction(tokenizer, cfg.model.instruction)
    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )
    prompt_ids = prompt_tokens["input_ids"].to(model.device)

    with torch.no_grad():
        text_embeds = model.lm.model.get_input_embeddings()(prompt_ids).float()

    # 4. Concat in the LM's continuous embedding space.
    inputs_embeds = torch.cat([visual_tokens.float(), text_embeds], dim=1)

    shape_table = mo.md(
        f"""## Tensor shapes through the pipeline

    | Stage | Shape | Notes |
    |---|---|---|
    | Image (PIL) | `{image.size}` | (width, height) |
    | Donut output | `{tuple(visual_features.shape)}` | (batch, n_visual, vis_dim) |
    | Projector output | `{tuple(visual_tokens.shape)}` | mapped to LM dim |
    | Instruction tokens | `{tuple(prompt_ids.shape)}` | (batch, n_text) |
    | Text embeddings | `{tuple(text_embeds.shape)}` | from LM's embedding table |
    | LM input `inputs_embeds` | `{tuple(inputs_embeds.shape)}` | visual ++ text, fed instead of `input_ids` |

    The LM sees **{inputs_embeds.shape[1]} positions** for this sample (`n_visual + n_text`)."""
    )

    # Render the chat-templated instruction in its own widget; embedding it
    # in the markdown above breaks the code-fence indentation because of the
    # real newlines inside the template.
    instruction_display = mo.vstack([
        mo.md("**Chat-templated instruction (verbatim):**"),
        mo.plain_text(instruction),
    ])

    mo.vstack([shape_table, instruction_display])
    return (instruction,)


@app.cell
def _(cfg, mo):
    # Load checkpoints. RL may not exist yet if training is still running.
    from pathlib import Path

    sft_path = Path(cfg.sft_best_checkpoint)
    rl_path = Path(cfg.rl_best_checkpoint)

    sft_exists = sft_path.exists()
    rl_exists = rl_path.exists()

    mo.md(
        f"""
        ## Checkpoints

        - SFT: `{sft_path}` → {"✅ found" if sft_exists else "⚠️ missing"}
        - RL: `{rl_path}` → {"✅ found" if rl_exists else "⚠️ not yet (RL still running?)"}
        """
    )
    return Path, rl_exists, rl_path, sft_exists, sft_path


@app.cell
def _(
    Path,
    cfg,
    device,
    generate_k_outputs,
    image,
    instruction,
    model,
    rl_exists,
    rl_path,
    sft_exists,
    sft_path,
    tokenizer,
    torch,
):
    def _generate_with_checkpoint(ckpt_path: Path) -> str:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.projector.load_state_dict(ckpt["projector_state_dict"])
        model.projector.eval()

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
        return gen.texts[0].strip()

    sft_output = _generate_with_checkpoint(sft_path) if sft_exists else None
    rl_output = _generate_with_checkpoint(rl_path) if rl_exists else None
    return rl_output, sft_output


@app.cell
def _(ground_truth, mo, rl_output, sft_output):
    if sft_output is None:
        mo.md("⚠️ Skipping SFT generation — no checkpoint.")
    else:
        gt_block = f"### Ground truth\n```json\n{ground_truth}\n```"
        sft_block = f"### SFT output\n```\n{sft_output}\n```"

        if rl_output is None:
            mo.md(f"## Generation\n\n{gt_block}\n\n{sft_block}\n\n*(RL checkpoint not available yet.)*")
        else:
            rl_block = f"### RL output\n```\n{rl_output}\n```"
            mo.md(f"## Generation: SFT vs RL\n\n{gt_block}\n\n{sft_block}\n\n{rl_block}")
    return


@app.cell
def _(compute_reward, ground_truth, mo, rl_output, sft_output):
    def _row(label, output):
        if output is None:
            return None
        b = compute_reward(output, ground_truth)
        return (
            f"| {label} | {b.total:.3f} | {b.format:.2f} | {b.schema:.2f} | "
            f"{b.content:.2f} | {b.hallucination:+.2f} |"
        )

    rows = [r for r in [_row("SFT", sft_output), _row("RL", rl_output)] if r]

    if not rows:
        mo.md("⚠️ No outputs to score.")
    else:
        mo.md(
            "## Reward breakdown\n\n"
            "Components clipped to [0, 1] in `total`. Maximum per-component is "
            "0.30 for format/schema/content; hallucination is a penalty (≤ 0).\n\n"
            "| Checkpoint | Total | Format | Schema | Content | Hallucination |\n"
            "|---|---|---|---|---|---|\n"
            + "\n".join(rows)
        )
    return


if __name__ == "__main__":
    app.run()
