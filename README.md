# vlm

Custom vision-language model for structured data extraction from scanned receipts. Built for the Document AI Alignment take-home assignment.

## Submission contents

This repository has two documents with different roles:

- **`README.md`** is the reproducible entry point: setup, commands, architecture summary, final config, reward design, headline results, and limitations.
- **`walkthrough.py`** is the evidence companion: an interactive Marimo report that loads artifacts from `training_runs/b4_stable_short/` and shows TensorBoard curves, evaluation summaries, qualitative samples, and design notes.
  - Also given as a html export in `walkthrough.html` for easy viewing without running the code.

To avoid duplication, detailed plots and per-sample outputs live in the walkthrough. The README summarizes the final decisions and results.

---

## Setup & usage

### Requirements

- Single CUDA GPU; the assignment target is consumer-class hardware.
- Python 3.10+
- `uv` for dependency management
- Hugging Face access for Donut and TinyLlama weights

### Install

```bash
git clone https://github.com/auloa/vlm.git
cd vlm
uv sync
uv run python -m vlm.utils.hf_login  # expects HF_TOKEN in .env
```

### Run the submitted config

The final config used in this submission is **`b4_stable_short`**.

```bash
# Supervised fine-tuning
uv run python -m vlm.scripts.train_sft -c b4_stable_short

# RL alignment; starts from the SFT best checkpoint
uv run python -m vlm.scripts.train_rl  -c b4_stable_short

# Evaluate both SFT and RL checkpoints on the held-out test split
uv run python -m vlm.scripts.evaluate  -c b4_stable_short

# Evaluate one stage explicitly
uv run python -m vlm.scripts.evaluate  -c b4_stable_short

```

Outputs are written to:

```text
training_runs/b4_stable_short/
├── checkpoints/
│   ├── sft/
│   │   ├── best.pt
│   │   └── epoch_XX.pt
│   └── rl/
│       ├── best.pt
│       └── step_XXXXXX.pt
├── runs/
│   ├── sft/                 # TensorBoard events
│   └── rl/
└── results/
    ├── eval_sft/
    │   ├── summary.json
    │   ├── samples.jsonl
    │   └── samples.md
    ├── eval_rl/
    │   └── ...
    └── eval_comparison.json
```


### Model test
```
# SFT and RL side by side, 10 samples
uv run python -m vlm.scripts.viz_inference -c b4_stable_short

# Single sample
uv run python -m vlm.scripts.viz_inference -c b4_stable_short --index 5

# More samples, SFT only
uv run python -m vlm.scripts.viz_inference -c b4_stable_short --stage sft --n 20

# Custom output path
uv run python -m vlm.scripts.viz_inference -c b4_stable_short --output results/viz.html```


### Sanity check

```bash
uv run python -m vlm.scripts.train_sft -c debug
uv run python -m vlm.scripts.train_rl  -c debug
uv run python -m vlm.scripts.evaluate  -c debug
```

### TensorBoard and walkthrough

```bash
tensorboard --logdir training_runs/b4_stable_short/runs
uv run marimo run walkthrough.py
```
---
Then run with `-c my_run`. The config name becomes the directory under `training_runs/`.

**Registered configs:**

| Config | Purpose |
|---|---|
| `debug` | Full pipeline on 20 samples — environment check |
| `base` | Baseline — batch 4, dropout 0.15, default settings |
| `b4_e25_nodrop` | 17 SFT epochs, no dropout, tight RL (KL 0.20) — strong SFT, RL degrades format |
| `b4_stable_short` | 17 SFT epochs, dropout 0.05, default RL — submitted model, RL improves over SFT |

> For full training curves, per-sample predictions, and design decision analysis see the interactive walkthrough:
> ```bash
> uv run marimo run walkthrough.py
> ```
**Key config fields (Defaults):**

| Section | Field                       | Default | What it controls                                               |
|---|-----------------------------|---------|----------------------------------------------------------------|
| `sft` | `epochs`                    | 17      | SFT training epochs                                            |
| `sft` | `batch_size`                | 4       | Batch size (no accumulation)                                   |
| `sft` | `grad_accum_steps`          | 4       | Gradient accumulation steps (1 means no gradient accumulation) |
| `sft` | `learning_rate`             | 5e-5    | AdamW LR for SFT                                               |
| `sft` | `max_target_length`         | 256     | Max target tokens; longer samples dropped                      |
| `rl` | `completions_per_image`     | 4       | K for group-relative advantages (≥ 2)                          |
| `rl` | `learning_rate`             | 5e-6    | AdamW LR for RL                                                |
| `rl` | `kl_coef`                   | 0.02    | KL penalty weight against SFT reference                        |
| `rl` | `temperature`               | 0.7     | Sampling temperature for RL rollouts                           |
| `vision` | `image_height, image_width` | 960x640 | Resize target for Donut processor                              |
| `projector` | `cross_attention`           | False   | Use resampler instead of MLP                                   |
| `projector` | `dropout`                   | 0.15    | Dropout rate in projector                                      |
| `eval` | `num_samples`               | 50      | Test samples for evaluation                                    |



Full dataclass definitions in `vlm/configs/training_schema.py`.

## Architecture

```text
image ──► frozen Donut encoder ──► visual features
                                      │
                                      ▼
                              trainable projector
                                      │
                                      ▼
                         visual tokens in LM embedding space
                                      │
                                      ▼
instruction ──► TinyLlama embeddings ─► concat [visual | text] ─► frozen TinyLlama ─► JSON
```

- **Vision encoder:** `naver-clova-ix/donut-base-finetuned-cord-v2`
- **Language model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable component:** projector only, about **12.6M parameters**
- **Frozen components:** Donut encoder and TinyLlama

### Visual token routing

The LLM normally receives `input_ids` and performs its own embedding lookup. Visual tokens do not have token IDs, so the model bypasses the lookup and passes `inputs_embeds` directly:

```python
visual_embeddings = projector(vision_encoder(images))       # (B, 600, 2048)
text_embeddings = lm.get_input_embeddings()(input_ids)      # (B, prompt_len, 2048)
inputs_embeds = torch.cat([visual_embeddings, text_embeddings], dim=1)
```

The visual prefix and instruction are masked with `-100`; only target JSON tokens contribute to the SFT cross-entropy loss.

### Projector

The submitted config uses an MLP projector:

```python
nn.Sequential(
    nn.Linear(vis_dim, 2 * llm_dim),
    nn.GELU(),
    nn.Linear(2 * llm_dim, llm_dim),
    nn.LayerNorm(llm_dim),
)
```

The output LayerNorm is important because the frozen LLM only responds reliably if projected visual embeddings land in a scale similar to its token embeddings.

---

## Data and schema

Dataset: **CORD-v2**, scanned restaurant receipts with parsed JSON ground truth.

The target mirrors CORD's native `gt_parse` structure rather than renaming it to a simplified schema:

```json
{
  "menu": [
    {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"}
  ],
  "sub_total": {
    "subtotal_price": "135,000",
    "service_price": "10,125",
    "tax_price": "14,513"
  },
  "total": {
    "total_price": "159,638",
    "cashprice": "159,638"
  }
}
```

All fields are optional except that the training/evaluation pipeline filters samples without usable menu items or `total.total_price`. Keeping CORD-native keys means the model is trained to extract visible receipt content instead of ignoring fields that do not fit a simplified schema.

Final run dataset statistics:

| Split | Requested | Used after filtering | Notes |
|---|---:|---:|---|
| train | 800 | 690 | 89 too long for `max_target_length=256` |
| validation | 100 | 93 | 5 too long |
| test | 50 | 50 | held-out evaluation slice |

---

## Training setup: `b4_stable_short`

### SFT

SFT teaches the projector to condition the frozen LLM on receipt images and produce CORD-style JSON.

| Setting | Value |
|---|---:|
| Epochs | 15 |
| Batch size | 4 |
| Gradient accumulation | 1 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Gradient clipping | 0.5 |
| Target length | 256 |
| Projector dropout | 0.15 |
| LR schedule | cosine with warmup |
| Mixed precision | bf16/AMP on CUDA |

Batch size 4 was kept because it learned the visual-language bridge faster than larger effective batches in this small-data setup. The tradeoff is earlier overfitting, so every epoch is checkpointed and final selection is made with validation loss, generation/evaluation method not used.

### RL alignment

RL starts from the SFT checkpoint and updates only the projector.

1. Sample `K=4` completions for one receipt image.
2. Score each completion with the task reward.
3. Compute group-relative advantages: `(reward_i - mean) / (std + eps)`.
4. Skip the update if all sampled rewards are identical.
5. Optimize a clipped policy objective with a KL penalty against a frozen SFT reference projector.

| Setting | Value |
|---|---:|
| Completions per image | 4 |
| Optimizer | AdamW |
| RL learning rate | 5e-6 |
| KL coefficient | 0.02 |
| PPO epochs | 1 |
| Clip epsilon | 0.2 |
| Max steps | 500 |
| Checkpoint criterion | EMA reward |

---

## Reward design

The final reward treats JSON/schema as **small bonuses and gates**, not as the main reward. SFT already makes JSON structure reliable, so RL should mainly optimize grounded content.

| Component | Role |
|---|---|
| Format | Tiny bonus for parseable/clean JSON |
| Schema | Small bonus and gate for content reward |
| Content | Main reward: total fields, menu item matching, subtotal/other fields |
| Hallucination | Negative reward for extra sections, extra fields, duplicate/unmatched menu items, leaked role tokens, and garbage repetition |

Important changes from earlier reward versions:

- Valid JSON alone receives only a small reward; `{}` is not allowed to score well.
  - Valid JSONs format learned in SFT phase are preserved, but RL is not rewarded for maintaining them and can be guided more by content.
- Text values use token F1, not recall-only overlap, so extra junk is penalized.
- Numeric/money fields use tolerant numeric matching.
- The reward is dynamic: only fields present in the ground truth for that sample are scored.

---

## Evaluation

```bash
uv run python -m vlm.scripts.evaluate -c b4_stable_short
```

Headline metric: **`format_adherence_rate`** — strict JSON with top-level `menu`, top-level `total`, non-empty `menu`, and non-empty `total.total_price`.

Additional metrics:

- `mean_full_structure_score`: top-level and per-item structural coverage.
- `total_match_rate`: digit-normalized match for `total.total_price`.
- `mean_key_coverage`: fraction of ground-truth leaf keys present in prediction.
- `mean_value_accuracy`: average value match over matched keys.
- `mean_extra_keys`: average hallucinated leaf keys per sample.
- `mean_reward`: task reward on deterministic held-out generations.

---

## Results: SFT vs RL on 50 held-out test receipts

| Metric | SFT |    RL |  Change |
|---|---:|------:|--------:|
| Strict JSON rate | 98.0% | 98.0% |    +0.0 |
| Format adherence rate | 98.0% | 98.0% |    +0.0 |
| Mean full structure score | 95.8% |   96% | +0.3 pp |
| Total match rate | 54.0% | 46.0% |  8.0 pp |
| Mean key coverage | 84.4% | 86.8% | +2.4 pp |
| Mean value accuracy | 66.7% | 62.5% | -4.2 pp |
| Mean extra keys | 1.22 |  1.46 |  +0.240 |
| Mean reward | 0.541 | 0.514 |  -0.027 |

### Interpretation

SFT solved the strict JSON/schema requirement well: both SFT and RL reach **98% format adherence** on the held-out test slice. RL slightly increased key coverage but did not improve overall held-out extraction quality; total accuracy, value accuracy, extra-key rate, and mean reward all moved slightly in the wrong direction.

This is useful evidence rather than a failure of the pipeline: once SFT already produces valid JSON, the remaining bottleneck is visual grounding. With both the vision encoder and LLM frozen, RL can only select among the behaviors the projector already makes available. It cannot fully teach the frozen LLM to read small receipt text or attend differently to visual tokens.

Finetuning of the reward design and hyperparameters could potentially improve the RL stage.

---

## Training progress evidence

Early SFT outputs were malformed or repetitive, including role-token leakage and long repeated numeric strings. As SFT progressed, outputs became strict JSON with plausible `menu`, `sub_total`, and `total` sections. Later examples still hallucinated item names/prices, which is why held-out value accuracy remains the main limitation.

The walkthrough shows this progression using saved TensorBoard scalars and sample outputs. Recommended plots to include in a report or screenshot set:

- SFT train/validation loss
- SFT gradient norm and learning rate
- RL reward and EMA reward
- RL content reward vs hallucination penalty
- RL KL loss
- SFT/RL qualitative predictions on the same held-out samples

---

## Design decisions

**Donut encoder.** Donut was chosen because it is pretrained on document images with a text-decoding objective, which is a better prior for receipts than object-centric image encoders.

**Image resize to 960x640.** The default Donut resolution produces too many visual tokens for TinyLlama's context budget. Resizing keeps the visual prefix around 600 tokens, leaving room for the instruction and JSON completion.

**TinyLlama.** A small chat-tuned LLM fits the single-GPU constraint and supports JSON-style generation. The tokenizer chat template is used consistently in SFT, RL, and evaluation.

**Projector-only training.** Only the projection layer is trainable, matching the assignment constraint. The final model has 1.19B total parameters and 12.6M trainable parameters.

**CORD-native schema.** Keeping native keys avoids training the model to ignore visible fields that do not fit a simplified schema.

**Reward as content-first.** Format/schema compliance is measured, but the reward primarily grades grounded receipt content and penalizes hallucination.

**Conservative RL.** RL uses a frozen SFT projector as reference policy and skips updates when all K completions receive identical rewards.

---

## Limitations

- **Visual grounding remains weak.** The projector can steer the frozen LLM, but the LLM itself cannot learn new attention behavior over visual tokens.
- **Receipt text is small and noisy.** Full line-item accuracy is much harder than producing valid JSON or matching totals.
- **RL did not improve final held-out reward.** It maintained format adherence and slightly improved coverage, but worsened value accuracy and hallucination rate on the 50-sample held-out evaluation.
- **Small dataset sensitivity.** Batch size 4 learns quickly but can overfit around later epochs. Larger effective batches are smoother but learned more slowly in this setup.

---

## Scaling to production

For dense, multi-page supply-chain documents such as bills of lading, this projector-only setup is not sufficient. The next steps would be:

- Add LoRA adapters to LLM attention layers so the LM can learn how to attend to visual tokens.
  - Larger LLMs with more attention heads would also help.
- Use higher-resolution page tiling or page-level encoders for dense documents.
- Finetuning of visual encoders to be aware of the layout.
- Finetuning the model with schema included prompts so different document types can be tackled.
- Evaluate with document-level precision/recall metrics, not only JSON validity.

---

## Project structure

```text
vlm/
├── configs/
│   ├── paths.py
│   ├── training_configs.py
│   └── training_schema.py
├── data/
│   ├── collator.py
│   └── dataset.py
├── models/
│   ├── language_model.py
│   ├── projector.py
│   ├── resampler.py
│   ├── receipt_vlm.py
│   └── vision_encoder.py
├── training/
│   ├── common.py
│   ├── generate.py
│   ├── rewards.py
│   ├── rl.py
│   ├── rl_utils.py
│   └── sft.py
├── evaluation/
│   └── evaluate.py
├── utils/
│   ├── device.py
│   ├── hf_login.py
│   ├── json_extractor.py
│   └── training.py
└── scripts/
    ├── train_sft.py
    ├── train_rl.py
    └── evaluate.py
```
