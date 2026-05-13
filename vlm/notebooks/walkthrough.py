import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # vlm — walkthrough

    Interactive evidence companion to `README.md`. The README is the reproducible
    entry point; this notebook is for inspecting artifacts from
    `training_runs/b4_stable_short/`: TensorBoard curves, evaluation summaries,
    qualitative samples, and design notes.

    **Sections:**
    1. Architecture overview
    2. Visual token routing (static + live forward pass)
    3. Dataset (image + parsed GT side by side)
    4. Training curves
    5. Cross-run comparison
    6. Eval results
    7. Per-sample inspector
    8. Design decisions (retrospective)
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import pandas as pd

    return Path, json, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Architecture

    ```
      image ──► vision encoder ──► visual features
                                        │
                                        ▼
                              projector (MLP + LayerNorm)
                                        │
                                        ▼  visual tokens (in LM embedding space)
                                                                      ┐
                                                                      │
                                                                      ├──► concat ──► LM ──► JSON
                                                                      │
      instruction ──► tokenize ──► LM embed ──► text embeddings ──────┘

      Frozen: vision encoder (Donut), language model (TinyLlama).
      Trainable: projector (~12M params).
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Visual token routing

    The LM's standard forward pass goes `input_ids → embedding_lookup → transformer`.
    Visual tokens have no ids — the projector output already lives in the LM's
    embedding space — so the lookup is bypassed and visual + text embeddings are
    concatenated directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    routing_diagram = mo.md(
        """
        ### Sequence layout at training time

        For a single sample with `K=1` (batch=1), the inputs to `lm.model(...)`:

        ```
        positions:        0 ────────────── 599 │ 600 ──── prompt_len-1 │ prompt_len ─── end
                          ┌──────────────────┐ ┌─────────────────────┐ ┌────────────────┐
        inputs_embeds:    │  visual_embeds   │ │  prompt_embeds      │ │  label_embeds  │
                          │  (1, 600, 2048)  │ │  (1, N_p, 2048)     │ │  (1, N_l-1,    │
                          │   from projector │ │  from LM.embed      │ │   2048)        │
                          └──────────────────┘ └─────────────────────┘ └────────────────┘
                          ┌──────────────────┐ ┌─────────────────────┐ ┌────────────────┐
        attention_mask:   │  1 1 1 ... 1     │ │  prompt_attn_mask   │ │  label mask    │
                          └──────────────────┘ └─────────────────────┘ └────────────────┘
                          ┌──────────────────┐ ┌─────────────────────┐ ┌────────────────┐
        labels:           │  -100 -100 ...   │ │  -100 -100 ... -100 │ │  target ids    │
                          └──────────────────┘ └─────────────────────┘ └────────────────┘
        ```

        Visual prefix is fully masked from loss (`-100`). Prompt is masked too.
        Only the JSON target tokens contribute gradients.
        """
    )
    routing_diagram
    return


@app.cell(hide_code=True)
def _(mo):
    routing_code = mo.md(
        """
        ### Code path

        ```python
        # 1. Encode image, project to LM embedding space (only projector is trainable).
        visual_embeds = self._get_visual_embeddings(images)      # (B, 600, llm_dim)  The function combines vison encoder plus projection layer
        text_embeds   = self._embed_input_ids(input_ids)         # (B, N_text, llm_dim)

        # 2. Concatenate.
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 3. Build matching attention mask.
        visual_attn = torch.ones(B, visual_len, device=device, dtype=torch.long)
        full_attn   = torch.cat([visual_attn, attention_mask], dim=1)

        # 4. Build labels: -100 over visual prefix and instruction.
        visual_labels = torch.full((B, visual_len), -100, ...)
        full_labels   = torch.cat([visual_labels, labels], dim=1)

        # 5. Forward through frozen LM with inputs_embeds (NOT input_ids).
        out = self.lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )
        ```

        At generation time the same `prepare_inputs_embeds` is reused with
        `model.generate(inputs_embeds=..., attention_mask=...)`.
        """
    )
    routing_code
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Live routing — real tensor shapes

    Loads the model (random projector weights, frozen Donut + TinyLlama) and runs
    a real forward pass on one CORD image. The shapes below are produced from
    the actual code path, not hard-coded.

    Click the button to load — first run takes ~30s to download weights.
    """)
    return


@app.cell
def _():
    import torch
    from vlm.models.receipt_vlm import ReceiptVLM
    from vlm.training.common import build_instruction, prepare_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReceiptVLM(
        device=device,
        vision_model_name="naver-clova-ix/donut-base-finetuned-cord-v2",
        default_vision_processor=False,
        image_height=960,
        image_width=640,
        lm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cross_attention_projector=False,
        projector_mult=2,
    )

    tokenizer = prepare_tokenizer(model.lm.tokenizer)
    instruction = build_instruction(
        tokenizer,
        "Extract the tabular data from this document and output it in JSON format.",
    )
    return instruction, model, tokenizer, torch


@app.cell
def _(image, instruction, mo, model, tokenizer, torch):
    # Encode the currently-selected dataset image.
    with torch.no_grad():
        visual_features = model.vision_encoder([image])
        visual_embeds = model.projector(visual_features).float()

    # Tokenize the instruction the way the SFT collator does.
    prompt_ids = tokenizer(instruction, return_tensors="pt", add_special_tokens=True)["input_ids"]
    with torch.no_grad():
        prompt_embeds = model.lm.model.get_input_embeddings()(prompt_ids.to(model.device)).float()

    inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)

    visual_attn = torch.ones(1, visual_embeds.shape[1], device=model.device, dtype=torch.long)
    prompt_attn = torch.ones_like(prompt_ids).to(model.device)
    full_attn = torch.cat([visual_attn, prompt_attn], dim=1)

    shapes = {
        "vision_encoder output (visual_features)": tuple(visual_features.shape),
        "projector output (visual_embeds)": tuple(visual_embeds.shape),
        "prompt embeds": tuple(prompt_embeds.shape),
        "concatenated inputs_embeds (visual | text)": tuple(inputs_embeds.shape),
        "full attention_mask": tuple(full_attn.shape),
    }

    mo.md(
        "**Real tensor shapes from this image:**\n\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in shapes.items())
        + f"\n\nLM embedding dim: `{visual_embeds.shape[-1]}` — "
        f"visual_embeds and prompt_embeds share it after the projector."
    )
    return full_attn, inputs_embeds, prompt_ids, visual_embeds


@app.cell(hide_code=True)
def _(full_attn, inputs_embeds, mo, model, tokenizer, torch):
    # Generate a short completion to demonstrate the inputs_embeds path works end-to-end.
    with torch.no_grad():
        out = model.lm.model.generate(
            inputs_embeds=inputs_embeds.to(dtype=model.lm.model_dtype),
            attention_mask=full_attn,
            max_new_tokens=40,
            do_sample=False,
        )

    completion = tokenizer.decode(out[0], skip_special_tokens=True)
    mo.md(
        "**Generated completion (random projector — output is gibberish but "
        "demonstrates the inputs_embeds path works end-to-end):**\n\n"
        f"```\n{completion}\n```"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Label masking — what contributes to the loss

    Training uses the full `[visual | prompt | target]` sequence as input, but
    only the **target JSON tokens** should contribute gradients. Everything
    else is masked to `-100` in the labels tensor (PyTorch's ignore index for
    cross-entropy).

    The breakdown on the currently-selected sample:
    """)
    return


@app.cell(hide_code=True)
def _(mo, parsed, prompt_ids, tokenizer, torch, visual_embeds):
    import json as _json_mask

    # Tokenize the target JSON the way the SFT collator does.
    target_str = _json_mask.dumps(parsed, ensure_ascii=False)
    eos = tokenizer.eos_token or ""
    target_ids = tokenizer(target_str + eos, return_tensors="pt", add_special_tokens=False)["input_ids"]

    visual_len = visual_embeds.shape[1]
    prompt_len = prompt_ids.shape[1]
    target_len = target_ids.shape[1]
    total_len = visual_len + prompt_len + target_len

    # Build the full labels tensor with -100 everywhere except target tokens.
    full_labels = torch.full((1, total_len), -100, dtype=torch.long)
    full_labels[0, visual_len + prompt_len : visual_len + prompt_len + target_len] = target_ids[0]

    contributing = (full_labels[0] != -100).sum().item()
    ignored = (full_labels[0] == -100).sum().item()

    mo.md(
        f"**Sequence breakdown:**\n\n"
        f"| Region | Token positions | Length | In loss? |\n"
        f"|---|---|---|---|\n"
        f"| Visual prefix | `0 … {visual_len - 1}` | {visual_len} | masked |\n"
        f"| Instruction prompt | `{visual_len} … {visual_len + prompt_len - 1}` | {prompt_len} | masked |\n"
        f"| Target JSON | `{visual_len + prompt_len} … {total_len - 1}` | {target_len} | contributes |\n\n"
        f"**Loss-contributing tokens:** {contributing} / {total_len} "
        f"({100 * contributing / total_len:.1f}%) — the rest is `-100`.\n\n"
        f"**Target string ({target_len} tokens):**\n\n"
        f"```json\n{target_str[:500]}{'...' if len(target_str) > 500 else ''}\n```"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Dataset

    CORD-v2 receipts parsed into the native CORD-style JSON schema: `menu`,
    `sub_total`, `total`, and optional visible fields. Pick a sample to see the
    receipt image alongside the parsed ground truth.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    return (load_dataset,)


@app.cell(hide_code=True)
def _(load_dataset, mo):
    raw_ds = load_dataset("naver-clova-ix/cord-v2", split="train")
    mo.md(f"Loaded **{len(raw_ds)}** raw samples.")
    return (raw_ds,)


@app.cell(hide_code=True)
def _(mo, raw_ds):
    ds_idx = mo.ui.slider(
        start=0,
        stop=len(raw_ds) - 1,
        value=0,
        label="sample index",
        show_value=True,
    )
    ds_idx
    return (ds_idx,)


@app.cell(hide_code=True)
def _():
    from vlm.data.dataset import parse_ground_truth

    return (parse_ground_truth,)


@app.cell(hide_code=True)
def _(ds_idx, json, parse_ground_truth, raw_ds):
    sample = raw_ds[ds_idx.value]
    image = sample["image"]
    raw_gt = json.loads(sample["ground_truth"])["gt_parse"]
    parsed = parse_ground_truth(sample["ground_truth"])
    return image, parsed


@app.cell(hide_code=True)
def _(image):
    # Resize for display; CORD images can be huge.
    w, h = image.size
    max_w = 400
    if w > max_w:
        scale = max_w / w
        display_img = image.resize((max_w, int(h * scale)))
    else:
        display_img = image
    return (display_img,)


@app.cell(hide_code=True)
def _(display_img, json, mo, parsed):
    img_panel = mo.image(display_img, alt="receipt")


    parsed_panel = mo.md(
        f"**Parsed for training:**\n\n"
        f"```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"
    )

    mo.hstack([img_panel, parsed_panel], gap=1, widths="equal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Training curves

    Pick a run and inspect its SFT and RL TensorBoard scalars. All tags from
    all sub-writers are loaded; use the multiselect to pick which to chart.
    """)
    return


@app.cell(hide_code=True)
def _(Path):
    runs_root = Path("training_runs")

    def list_runs() -> list[str]:
        if not runs_root.exists():
            return []
        return sorted(p.name for p in runs_root.iterdir() if p.is_dir())

    return list_runs, runs_root


@app.cell(hide_code=True)
def _(list_runs, mo):
    available_runs = list_runs()
    run_picker = mo.ui.dropdown(
        options=available_runs,
        value="b4_stable_short" if "b4_stable_short" in available_runs else (available_runs[0] if available_runs else None),
        label="run",
    )
    run_picker
    return available_runs, run_picker


@app.cell(hide_code=True)
def _(run_picker, runs_root):
    selected_run = run_picker.value
    if selected_run is None:
        sft_dir = None
        rl_dir = None
        results_dir = None
    else:
        run_dir = runs_root / selected_run
        sft_dir = run_dir / "runs" / "sft"
        rl_dir = run_dir / "runs" / "rl"
        results_dir = run_dir / "results"
    return results_dir, rl_dir, sft_dir


@app.cell(hide_code=True)
def _(Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    def _event_dirs(root: Path) -> list[Path]:
        if not root or not root.exists():
            return []
        dirs = set()
        for p in root.rglob("events.out.tfevents.*"):
            dirs.add(p.parent)
        return sorted(dirs)

    def load_all_scalars(run_dir):
        """Load every scalar across all event sub-directories.

        Returns {tag: [(step, value), ...]}. Tags from sub-writers keep their
        relative subdirectory prefix so series stay distinct.
        """
        if run_dir is None:
            return {}
        out_ = {}
        for d in _event_dirs(run_dir):
            ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
            ea.Reload()
            for t in ea.Tags().get("scalars", []):
                points = [(e.step, e.value) for e in ea.Scalars(t)]
                if d == run_dir:
                    key = t
                else:
                    rel = d.relative_to(run_dir)
                    key = f"{rel}::{t}"
                out_[key] = points
        return out_

    return (load_all_scalars,)


@app.cell(hide_code=True)
def _(load_all_scalars, rl_dir, sft_dir):
    sft_scalars = load_all_scalars(sft_dir)
    rl_scalars = load_all_scalars(rl_dir)
    return rl_scalars, sft_scalars


@app.cell(hide_code=True)
def _(pd):
    import altair as alt

    def scalars_to_df(scalars_dict: dict) -> pd.DataFrame:
        rows = []
        for tag, points in scalars_dict.items():
            for step, value in points:
                rows.append({"tag": tag, "step": step, "value": value})
        return pd.DataFrame(rows)

    def line_chart(df, title=""):
        if df.empty:
            return None
        return (
            alt.Chart(df)
            .mark_line()
            .encode(x="step:Q", y="value:Q", color="tag:N")
            .properties(width=700, height=300, title=title)
            .interactive()
        )

    return alt, line_chart, scalars_to_df


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### SFT
    """)
    return


@app.cell(hide_code=True)
def _(mo, sft_scalars):
    sft_tags_all = sorted(sft_scalars.keys())
    sft_tag_picker = mo.ui.multiselect(
        options=sft_tags_all,
        value=[t for t in sft_tags_all if "sft_epoch_loss" in t.lower()][:4],
        label="SFT tags",
    )
    sft_tag_picker
    return (sft_tag_picker,)


@app.cell(hide_code=True)
def _(line_chart, scalars_to_df, sft_scalars, sft_tag_picker):
    sft_selected = sft_tag_picker.value
    sft_df = scalars_to_df({t: sft_scalars[t] for t in sft_selected})
    line_chart(sft_df, "SFT")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### RL
    """)
    return


@app.cell
def _(mo, rl_scalars):
    rl_tags_all = sorted(rl_scalars.keys())
    rl_tag_picker = mo.ui.multiselect(
        options=rl_tags_all,
        value=[t for t in rl_tags_all if "reward/ema" in t.lower() or "reward/mean" in t.lower()],
        label="RL tags",
    )
    rl_tag_picker
    return (rl_tag_picker,)


@app.cell
def _(line_chart, rl_scalars, rl_tag_picker, scalars_to_df):
    rl_selected = rl_tag_picker.value
    rl_df = scalars_to_df({t: rl_scalars[t] for t in rl_selected})
    line_chart(rl_df, "RL")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Cross-run comparison

    Pick up to 3 runs and a tag substring. All matching scalars are overlaid
    on a single chart, color-coded by run.
    """)
    return


@app.cell(hide_code=True)
def _(available_runs, mo):
    compare_runs = mo.ui.multiselect(
        options=available_runs,
        value=available_runs[: min(2, len(available_runs))],
        label="runs to compare (max 3)",
        max_selections=3,
    )
    compare_stage = mo.ui.dropdown(
        options=["sft", "rl"],
        value="sft",
        label="stage",
    )
    compare_tag_substring = mo.ui.text(
        value="epoch_loss",
        label="tag substring",
    )

    mo.hstack([compare_runs, compare_stage, compare_tag_substring])
    return compare_runs, compare_stage, compare_tag_substring


@app.cell(hide_code=True)
def _(
    alt,
    compare_runs,
    compare_stage,
    compare_tag_substring,
    load_all_scalars,
    pd,
    runs_root,
):
    compare_rows = []
    for run_name in compare_runs.value:
        stage_dir = runs_root / run_name / "runs" / compare_stage.value
        scalars = load_all_scalars(stage_dir)
        substring = compare_tag_substring.value.lower()
        for tag, points in scalars.items():
            if substring and substring not in tag.lower():
                continue
            for step, value in points:
                compare_rows.append({
                    "run": run_name,
                    "tag": tag,
                    "step": step,
                    "value": value,
                    "series": f"{run_name} :: {tag}",
                })

    compare_df = pd.DataFrame(compare_rows)
    if compare_df.empty:
        compare_chart = None
    else:
        compare_chart = (
            alt.Chart(compare_df)
            .mark_line()
            .encode(
                x="step:Q",
                y="value:Q",
                color="series:N",
                strokeDash="run:N",
            )
            .properties(width=700, height=350, title="Cross-run comparison")
            .interactive()
        )

    compare_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Results

    Summary metrics and per-sample comparison for the selected run. The submitted

    The main comparison should be read as evidence, not only as a leaderboard:
    SFT already solves strict JSON formatting well, while RL tests whether the
    reward can improve grounded content without breaking format.
    """)
    return


@app.cell(hide_code=True)
def _(json, mo, results_dir):
    if results_dir is None:
        comparison = None
        comparison_msg = mo.md("_no run selected_")
    else:
        comparison_path = results_dir / "eval_comparison.json"
        if comparison_path.exists():
            with open(comparison_path) as f:
                comparison = json.load(f)
            comparison_msg = mo.md(f"Loaded `{comparison_path}`")
        else:
            comparison = None
            comparison_msg = mo.md(f"`{comparison_path}` not found — run `evaluate.py`.")

    comparison_msg
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison, mo, pd):
    if comparison is None:
        eval_table = mo.md("_no eval data_")
    else:
        sft_m = comparison.get("sft", {})
        rl_m = comparison.get("rl", {})
        metric_rows = [
            ("format_adherence_rate", "Format adherence", True),
            ("mean_full_structure_score", "Full structure score", True),
            ("strict_json_rate", "Strict JSON", True),
            ("total_match_rate", "Total match", True),
            ("mean_key_coverage", "Key coverage", True),
            ("mean_value_accuracy", "Value accuracy", True),
            ("mean_extra_keys", "Extra keys (hallucinated)", False),
            ("mean_reward", "Mean reward", False),
        ]
        table_rows = []
        for key, label, is_pct in metric_rows:
            s = sft_m.get(key)
            r = rl_m.get(key)
            delta = (r - s) if (s is not None and r is not None) else None
            fmt = lambda v: f"{v:.1%}" if is_pct and v is not None else (f"{v:.3f}" if v is not None else "—")
            table_rows.append({
                "metric": label,
                "sft": fmt(s),
                "rl": fmt(r),
                "delta": (f"{delta:+.1%}" if is_pct else f"{delta:+.3f}") if delta is not None else "—",
            })
        eval_table = pd.DataFrame(table_rows)

    eval_table
    return


@app.cell
def _(comparison, mo):
    if comparison is None:
        interpretation = mo.md("_no eval comparison loaded_")
    else:
        sft_m_ = comparison.get("sft", {})
        rl_m_ = comparison.get("rl", {})
        interpretation = mo.md(
            f"""
            ### Result interpretation

            For the final `b4_stable_short` run, SFT already reaches high JSON/schema
            reliability. RL is therefore judged mainly by content and hallucination
            behavior, not by whether it can make JSON parse.

            - SFT format adherence: **{sft_m_.get('format_adherence_rate', 0):.1%}**
            - RL format adherence: **{rl_m_.get('format_adherence_rate', 0):.1%}**
            - SFT mean reward: **{sft_m_.get('mean_reward', 0):.3f}**
            - RL mean reward: **{rl_m_.get('mean_reward', 0):.3f}**

            In the latest held-out evaluation, RL maintained format adherence and
            slightly improved key coverage, but it did **not** improve total match,
            value accuracy, extra-key rate, or mean reward. This supports the main
            conclusion: once SFT has learned strict JSON, projector-only RL has limited
            ability to fix weak visual grounding.
            """
        )
    interpretation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Per-sample delta: SFT → RL
    """)
    return


@app.cell(hide_code=True)
def _(json, results_dir):
    def _load_jsonl(path):
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    if results_dir is None:
        sft_samples = []
        rl_samples = []
    else:
        sft_samples = _load_jsonl(results_dir / "eval_sft" / "samples.jsonl")
        rl_samples = _load_jsonl(results_dir / "eval_rl" / "samples.jsonl")
    return rl_samples, sft_samples


@app.cell(hide_code=True)
def _(mo, pd, rl_samples, sft_samples):
    if not sft_samples or not rl_samples:
        delta_table = mo.md("_need both SFT and RL samples_")
    else:
        _n = min(len(sft_samples), len(rl_samples))
        _improved = _regressed = _unchanged = 0
        _tm_gained = _tm_lost = 0
        _fs_gained = _fs_lost = 0
        for _i in range(_n):
            _sr = sft_samples[_i].get("reward", 0) or 0
            _rr = rl_samples[_i].get("reward", 0) or 0
            _stm = sft_samples[_i].get("total_match", False)
            _rtm = rl_samples[_i].get("total_match", False)
            _sfs = sft_samples[_i].get("full_structure_score", 0.0)
            _rfs = rl_samples[_i].get("full_structure_score", 0.0)
            if _rr > _sr + 1e-6:
                _improved += 1
            elif _rr < _sr - 1e-6:
                _regressed += 1
            else:
                _unchanged += 1
            if not _stm and _rtm:
                _tm_gained += 1
            elif _stm and not _rtm:
                _tm_lost += 1
            if _rfs > _sfs + 0.05:
                _fs_gained += 1
            elif _sfs > _rfs + 0.05:
                _fs_lost += 1
        delta_table = pd.DataFrame([
            {"category": "RL reward > SFT", "count": _improved},
            {"category": "RL reward < SFT", "count": _regressed},
            {"category": "RL reward == SFT", "count": _unchanged},
            {"category": "Gained total_match", "count": _tm_gained},
            {"category": "Lost total_match", "count": _tm_lost},
            {"category": "Gained full_structure (>5%)", "count": _fs_gained},
            {"category": "Lost full_structure (>5%)", "count": _fs_lost},
        ])
    delta_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Per-sample inspector
    """)
    return


@app.cell(hide_code=True)
def _(mo, sft_samples):
    if not sft_samples:
        sample_picker = None
        _msg = mo.md("_no samples — run evaluate.py_")
    else:
        sample_picker = mo.ui.slider(
            start=0,
            stop=len(sft_samples) - 1,
            value=0,
            label="sample index",
        )
        _msg = sample_picker
    _msg
    return (sample_picker,)


@app.cell(hide_code=True)
def _(mo, rl_samples, sample_picker, sft_samples):
    if sample_picker is None or not sft_samples:
        view = mo.md("")
    else:
        _idx = sample_picker.value
        _sft = sft_samples[_idx]
        _rl = rl_samples[_idx] if _idx < len(rl_samples) else {}

        view = mo.md(
            "**Ground truth:**\n\n"
            f"```json\n{_sft.get('ground_truth', '')}\n```\n\n"
            f"**SFT** — format={_sft.get('format_adherent')}, "
            f"structure={_sft.get('full_structure_score', 0):.1%}, "
            f"total_match={_sft.get('total_match')}, "
            f"key_coverage={_sft.get('key_coverage', 0):.1%}, "
            f"value_accuracy={_sft.get('value_accuracy', 0):.1%}, "
            f"reward={_sft.get('reward', 0):.3f}\n\n"
            f"```json\n{_sft.get('prediction', '')}\n```\n\n"
            f"**RL** — format={_rl.get('format_adherent')}, "
            f"structure={_rl.get('full_structure_score', 0):.1%}, "
            f"total_match={_rl.get('total_match')}, "
            f"key_coverage={_rl.get('key_coverage', 0):.1%}, "
            f"value_accuracy={_rl.get('value_accuracy', 0):.1%}, "
            f"reward={_rl.get('reward', 0):.3f}\n\n"
            f"```json\n{_rl.get('prediction', '')}\n```"
        )
    view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Design decisions

    Short retrospective on the choices that shaped this submission, with the
    evidence behind each.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Vision encoder:

    Donut is pretrained on document images with a text-decoding objective.
    Picked Donut because it was finetuned on CORD v2 dataset; dropped the BART decoder, kept only the encoder.

    Image processor's resize is overridden to 960×640 — default Donut targets
    ~2560×1920 and produces ~4800 visual tokens, blowing past TinyLlama's
    context window. Plus computational cost will be quite high.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### CORD-native schema: verbatim keys

    The training target uses CORD's `gt_parse` keys directly (`nm`, `cnt`,
    `price`, `sub_total`, `total_price`, `cashprice`, etc.) with no renaming.
    Every field visible on the receipt maps to a target token.


    **Alternative considered:** flattening to `{line_items, total}` would make
    the schema easier, but it would also discard visible fields such as taxes,
    subtotals, payment details, and item modifiers. For a digitization task, the
    native CORD schema is a better test of whether the model is grounded in the
    receipt image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Projector: MLP + LayerNorm

    Two-layer MLP from `vis_dim` to `2 × llm_dim` to `llm_dim`, with output
    LayerNorm. Layer normalization layer helps with the normalization of magnitude for the LLM.

    **Alternative tried:** cross-attention resampler with 64 learned queries
    (~92M trainable params vs 12M for MLP). Assignment constrainsts (computational expense) led to the choice of MLP
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Language model: TinyLlama

    Small chat-tuned LM that fits on a single consumer GPU alongside the vision
    encoder. Used `tokenizer.apply_chat_template` at runtime — an earlier
    hand-written prompt suffix caused the model to leak its own role tokens.

    **Alternative tried:** Qwen-2.5-1.5B for better Indonesian coverage (CORD
    is largely Indonesian). Output distributions shifted toward coherent
    Indonesian dish names. Item names still didn't match the actual receipt —
    sampling from a better prior, not reading the image better. Visual grounding
    is the bottleneck, not vocabulary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reward: content-first grading

    Earlier reward versions over-weighted format and schema, which made valid-looking
    JSON too rewarding even when values were weakly grounded. The final reward makes
    format/schema small bonuses and gates, while content carries the main signal.

    Content is scored dynamically against the GT for that receipt:

    - `total` fields receive high weight, especially `total_price`.
    - `menu` is matched approximately by item name and price rather than strict index.
    - other visible fields are scored through recursive leaf matching.
    - text uses token F1, not recall-only overlap, so extra junk is penalized.
    - numeric/money values use tolerant numeric matching.

    Hallucination penalties apply to extra sections, extra leaf keys, duplicate or
    unmatched menu items, leaked role tokens, and repeated-character garbage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Training: batch size, dropout, and checkpoint choice

    Batch 4 (no gradient accumulation) learned the projector alignment faster than
    larger effective batches in this small-data setting. The tradeoff is that it can
    move into memorization after the useful SFT window, so the run saves checkpoints
    and evaluates deterministic generations rather than trusting loss alone.

    Dropout in the projector helps keep the SFT solution less sharp. Larger effective
    batches were smoother, but learned more slowly and did not produce better held-out
    generations in the same time budget.

    The important practical lesson is not "batch 4 is always best"; it is that with
    frozen base models and a small dataset, checkpoint selection should consider
    generated JSON quality, total matching, hallucinations, and validation loss together.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### RL: GRPO with PPO clipping, ppo_epochs=1

    The loop implements the clipped PPO surrogate with snapshotted old log-probs.
    At `ppo_epochs=1` clipping is a no-op. Tried `ppo_epochs=4` — no improvement.

    Step-based EMA-reward checkpointing. JSON/schema rates are logged with the
    same parser used by the reward function, while reward components show whether
    RL is improving content or merely preserving format.

    In the final evaluation, RL maintained the 98% format adherence achieved by
    SFT and slightly increased key coverage, but it did not improve mean held-out
    reward. That is the main post-training finding: RL is a useful alignment loop,
    but when the SFT model already outputs valid JSON, the remaining bottleneck is
    visual grounding rather than formatting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What didn't work / wasn't enough

    - **Visual grounding is the ceiling.** The frozen LM cannot learn new attention
      behavior over visual tokens. The projector can only map image features into
      a subspace the frozen LM already knows how to use.
    - **RL is not a substitute for grounding.** If SFT completions are plausible but
      visually wrong, RL mostly chooses among plausible wrong receipts.
    - **Better language priors are not enough.** The frozen LM did not help with grasping the visual information very well.

    Real fix: LoRA on LM attention layers. A small number of trainable attention
    parameters would let the LM actually learn to attend to the visual prefix. That
    is out of scope for the projector-only assignment, but it is the most likely
    next improvement. Plus hyperparameter tuning
    """)
    return


if __name__ == "__main__":
    app.run()
