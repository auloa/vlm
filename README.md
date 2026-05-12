# vlm

Custom vision-language model for structured data extraction from scanned receipts. Submitted for the Document AI Alignment take-home assignment.

## Setup & Usage

### Requirements

- Single GPU with 8–16 GB VRAM (tested on RTX 3090)
- Python 3.10+
- `uv` for dependency management
- Hugging Face account with token (for downloading Donut and TinyLlama weights)

### Install

```bash
git clone <repo>
cd vlm
uv sync
python -m vlm.utils.hf_login   # saves HF token to ~/.cache/huggingface
```

### Run the full pipeline

All scripts take a `-c <config_name>` argument pointing at a named config in `vlm/configs/training_configs.py`.

```bash
# Supervised fine-tuning
python -m vlm.scripts.train_sft -c base

# RL alignment (starts from SFT best checkpoint automatically)
python -m vlm.scripts.train_rl  -c base

# Evaluate both SFT and RL checkpoints on the held-out test split
python -m vlm.scripts.evaluate  -c base

# Resume an interrupted run
python -m vlm.scripts.train_sft -c base --resume
python -m vlm.scripts.train_rl  -c base --resume
```

Outputs land in `training_runs/<config_name>/`:

```
training_runs/base/
├── checkpoints/
│   ├── sft/
│   │   ├── best.pt          # best val-loss checkpoint
│   │   └── epoch_XX.pt      # per-epoch checkpoints (resume targets)
│   └── rl/best.pt           # best EMA-reward RL checkpoint
├── runs/                    # TensorBoard event files
│   ├── sft/
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

### Sanity check

```bash
python -m vlm.scripts.train_sft -c debug
python -m vlm.scripts.train_rl  -c debug
python -m vlm.scripts.evaluate  -c debug
```

Runs the full pipeline on 20 samples in a few minutes.

### TensorBoard

```bash
tensorboard --logdir training_runs/base/runs
```

### Interactive walkthrough

```bash
marimo run walkthrough.py   # read-only app mode
marimo edit walkthrough.py  # editable notebook mode
```

Requires: `uv add marimo altair pandas tensorboard datasets pillow`

---

### Configuration

All training hyperparameters live in `vlm/configs/training_configs.py`. Named configs are registered with `@register_config` and inherit from `_base_receipt_config`. To create a new run:

```python
@register_config
def my_run(name: str) -> TrainingConfig:
    cfg = _base_receipt_config(name)
    cfg.sft.epochs = 25
    cfg.rl.kl_coef = 0.20
    return cfg
```

Then run with `-c my_run`. The config name becomes the directory under `training_runs/`.

**Registered configs:**

| Config | Purpose |
|---|---|
| `debug` | Full pipeline on 20 samples — environment check |
| `base` | Submitted model — batch 4, dropout 0.15, 15 epochs |
| `nodrop` | Ablation: no projector dropout |
| `long` | 25 SFT epochs with dropout |
| `tight` | Tighter RL: KL 0.20, LR 1e-6 |
| `best` | Long SFT + tight RL combined |
| `ca` | Cross-attention resampler with dropout |

**Key config fields:**

| Section | Field                   | Default | What it controls                         |
|---|-------------------------|---------|------------------------------------------|
| `sft` | `epochs`                | 15      | SFT training epochs                      |
| `sft` | `batch_size`            | 4       | Batch size (no accumulation)             |
| `sft` | `grad_accum_steps`       | 1       | Effective batch size                     |
| `sft` | `learning_rate`         | 5e-5    | AdamW LR for SFT                         |
| `sft` | `max_target_length`     | 256     | Max target tokens; longer samples dropped |
| `rl` | `completions_per_image` | 4       | K for group-relative advantages (≥ 2)    |
| `rl` | `learning_rate`         | 5e-6    | AdamW LR for RL                          |
| `rl` | `kl_coef`               | 0.02    | KL penalty weight against SFT reference  |
| `rl` | `temperature`           | 0.7     | Sampling temperature for RL rollouts     |
| `vision` | `image_height/width`    | 640×960 | Resize target for Donut processor        |
| `projector` | `cross_attention`       | False   | Use resampler instead of MLP             |
| `projector` | `dropout`               | 0.15    | Dropout rate in projector                |
| `eval` | `num_samples`           | 50      | Test samples for evaluation              |

Full dataclass definitions in `vlm/configs/training_schema.py`.

## Architecture

```
  image ──► vision encoder ──► visual features
                                    │
                                    ▼
                          projector (MLP + LayerNorm)
                                    │
                                    ▼  visual tokens (in LM embedding space)
                                                                  ┐
                                                                  │
                                                                  ├──► concat ──► language model ──► JSON
                                                                  │
  instruction ──► tokenize ──► LM embed ──► text embeddings ──────┘

  Frozen: vision encoder (Donut), language model (TinyLlama).
  Trainable: projector (~12M params).
```

### Vision encoder

`naver-clova-ix/donut-base-finetuned-cord-v2`. Donut is a Swin transformer encoder + BART-style decoder fine-tuned end-to-end on CORD. Only the encoder is used — the decoder is dropped after loading:

```python
full_model = VisionEncoderDecoderModel.from_pretrained(model_name)
self.model = full_model.encoder
del full_model
```

The image processor's resize is overridden to 640×960. The default Donut processor targets ~2560×1920, producing ~4800 visual tokens that exceed TinyLlama's context window. At 640×960 the count drops to 600.

### Visual token routing

The LM's standard forward pass expects token ids via an embedding lookup. Visual tokens have no ids — the projector output already lives in the LM's embedding space — so the lookup is bypassed:

```python
visual_embeddings = self._get_visual_embeddings(images)   # (B, 600, llm_dim)
text_embeddings   = self._embed_input_ids(input_ids)      # (B, N_text, llm_dim)
inputs_embeds     = torch.cat([visual_embeddings, text_embeddings], dim=1)
```

Labels are masked to `-100` over the visual prefix and instruction. Only the target JSON tokens contribute to the loss.

### Projector

```python
nn.Sequential(
    nn.Linear(vis_dim, 2 * llm_dim),
    nn.GELU(),
    nn.Linear(2 * llm_dim, llm_dim),
    nn.LayerNorm(llm_dim),
)
```

Two-layer MLP mapping Donut features into the LM's embedding space. The output LayerNorm was added after early runs where the LM ignored the visual prefix entirely — pre-norm outputs sat at magnitudes the frozen LM attention didn't react to.

### Language model

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`. `tokenizer.apply_chat_template` is used at runtime via `build_instruction`. An earlier hand-written prompt suffix caused the model to leak its own role tokens into outputs — and evaluation with the raw instruction string instead of the chat-templated version produced misleading results (the model was trained on one prompt format and evaluated on another).

## Data

CORD-v2 — scanned restaurant receipts with parsed JSON ground truth.

### Schema

The training target mirrors CORD's native `gt_parse` structure verbatim — every field visible on the receipt has a place in the label:

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

All fields are optional — emitted only when the receipt shows them. Using CORD's native keys directly means the model is trained to extract every visible line. Renaming or dropping fields introduces "ignore this visual region" signal that weakens the grounding prior across all samples.

### Parser

Samples are filtered if they fail to parse, have no menu items, or have no `total.total_price`. Per-reason counts are logged at startup via `CORDDataset.print_drop_summary()`.

~25 samples (~3%) are dropped because the raw `total` dict uses `cashprice` or `creditcardprice` instead of `total_price`.

#### Target length distribution (800 samples)

| p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|
| 98 | 152 | 232 | 480 | 712 |

`max_target_length = 256` drops ~8% of samples.

## Training

Two stages: SFT to teach the projector basic extraction, then RL to align output format and content.

### SFT

Cross-entropy over the target JSON tokens. Visual prefix and instruction masked from the loss.

| Epochs | 15 |
|---|---|
| Batch size | 4 (no gradient accumulation) |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01 |
| LR schedule | Cosine with warmup |
| Mixed precision | bf16 on CUDA |
| Dropout | 0.15 in projector |

Best val-loss checkpoint saved per epoch for resume support.


### RL

Starts from the SFT best checkpoint (lowest validation loss in this case not based on reward computation).

1. Sample K=4 completions from the current policy
2. Score each with the reward function
3. Compute group-relative advantages `(R_i − μ) / (σ + ε)`
4. Skip if all rewards equal — no preference signal
5. Optimize against a frozen copy of the SFT projector as KL reference

| K (completions per image) | 4 |
|---|---|
| Optimizer | AdamW, lr 5e-6, weight decay 0.01 |
| KL coefficient β | 0.02 |
| Gradient clipping | 0.5 |
| Max steps | 500 |
| Best checkpoint | EMA-reward improvement, snapshots every 200 steps |

## Reward

Four components, total clipped to [0, 1]:

| Component | Range | What it measures |
|---|---|---|
| format | 0–0.30 | 0.30 for strict JSON, 0.10 for parseable-but-wrapped |
| schema | 0–0.30 | menu is a list with nm+price items, total dict has total_price |
| content | 0–0.40 | dynamic per-key grading against GT structure |
| hallucination | ≤ 0 | extra keys, duplicate items, wrong text values, garbage |

**Content grading** walks the GT recursively. For each leaf key, the model is scored by tolerant numeric match (money fields) or token overlap (text fields). Score = `coverage × value_accuracy`.

**Hallucination penalty** has two parts:
- *Structural:* extra keys in pred not in GT (-0.04 per key, cap -0.20), duplicate item names, excessive menu length, leaked role tokens
- *Text value:* for text fields present in both pred and GT, `penalty = -0.03 × (1 - token_overlap)` — starts negative, erodes to 0 at full overlap. Capped at -0.15 total. Numeric fields are not penalised here — those are handled by the content score.

Both are dynamic — only keys present in that sample's GT are scored.

## Evaluation

```bash
python -m vlm.scripts.evaluate -c base
```

**Headline metric: `format_adherence_rate`** — parseable strict JSON with `menu` (non-empty list) and `total.total_price` present. Matches the assignment's required metric.

**`full_structure_rate`** — stricter: all GT top-level keys present in pred AND all GT per-item keys appear in at least one pred item.

Additional metrics:

- `strict_json_rate`, `extractable_json_rate`
- `total_match_rate` — digit-exact match with GT `total.total_price`
- `mean_key_coverage` — fraction of GT leaf keys present in prediction
- `mean_value_accuracy` — average value match score across matched keys
- `mean_extra_keys` — avg hallucinated keys per sample

Full per-sample outputs in `results/eval_{stage}/samples.{jsonl,md}`.

## Results

Full results in `results/eval_comparison.json` and the interactive walkthrough (`marimo run walkthrough.py`).

### SFT training behavior

![SFT training loss](assets/sft_loss.png)

No-dropout (blue/green): val bottoms at ~0.47 around epoch 9, then climbs — overfitting. Dropout (pink/orange): val still declining at epoch 15 — slower convergence but no overfitting.

**Key finding:** earlier runs showed only 4% and 24% SFT format adherence. This was caused by an evaluation bug — `build_instruction` (which wraps the prompt with the chat template) was not called during evaluation, so the model saw a different prompt at eval time than it was trained on. With the fix applied, the `nodrop` config reaches **98% format adherence** in 15 epochs.

### RL training behavior

![RL reward EMA](assets/rl_reward_ema.png)
![RL reward components](assets/rl_reward_components.png)

EMA reward peaks early (~step 100), then plateaus. Format and schema components stay near their maximums throughout — RL maintains what SFT built. Content component carries the actual RL signal. Best-EMA checkpoint captures the early peak.

### Eval results

_Pending final RL run. Will be updated._

| Metric | SFT | RL |
|---|---|---|
| format_adherence_rate | — | — |
| full_structure_rate | — | — |
| total_match_rate | — | — |
| mean_key_coverage | — | — |
| mean_value_accuracy | — | — |
| mean_reward | — | — |

## Design Decisions

**Donut over CLIP/SigLIP.** Donut's encoder is pretrained on document images with a text-decoding objective.

**LayerNorm at the projector output.** It is added so that the component-wise magnitudes of the visual tokens sit in a range the frozen LM attention layers react to.

**CORD-native schema (verbatim keys).** Every visible field on the receipt maps to a target token. This is done to preserve the visual grounding signal across all section of the image.

**Dynamic reward grading.** The content reward scores only keys present in that sample's GT. Schema-agnostic: adding fields to the parser automatically changes what gets scored.

**Text hallucination penalty.** For text fields where both pred and GT have values, penalty = `-0.03 × (1 - token_overlap)`. Starts negative, erodes to zero at full overlap. Pushes the model to use visual signal rather than sampling from its prior.

**Batch size 4 (no gradient accumulation).** With ~720 training samples, smaller batches do ~4× more optimizer steps per epoch. No-dropout batch-4 reaches 98% format adherence in 15 epochs; larger effective batch sizes don't converge within the same epoch budget.

**Dropout in the projector.** No-dropout reaches a sharp SFT minimum — good for SFT metrics but RL destabilized it across multiple runs. Dropout (p=0.15) produces a flatter minimum that RL can refine without collapsing format adherence.

**Resume support.** Both SFT and RL support `--resume`. SFT saves optimizer and scheduler state per epoch checkpoint; RL saves optimizer state in the best-EMA checkpoint.

**GRPO with PPO clipping, `ppo_epochs=1`.** Clipped PPO surrogate implemented with snapshotted old log-probs. At `ppo_epochs=1` clipping is a no-op — tried `ppo_epochs=4`, no improvement.

### Alternatives explored

**Cross-attention resampler.** ~92M trainable params vs 12M for MLP. Val loss climbed from epoch 7 — a 128,000:1 parameter-to-sample ratio. The attention queries learn sample-specific visual fingerprints on 720 receipts. Bridge capacity is not the bottleneck.

**Qwen-2.5 as the language model.** Output shifted toward coherent Indonesian dish names. Names still didn't match the actual receipt — sampling from a better language prior, not reading the image better. Bottleneck is visual grounding, not vocabulary.

**No-dropout SFT.** Reaches 98% format adherence in 15 epochs. However, RL starting from this checkpoint degraded all metrics across multiple runs — the sharp SFT minimum was destabilized by noisy RL advantages. The dropout checkpoint is the more RL-stable starting point.

## Limitations

**Frozen base models.** The root cause of every limitation below. With both models frozen the LM can't learn to attend to visual positions; the encoder can't adapt to the training distribution.

**Visual grounding ceiling.** The projector must find the exact subspace of the frozen LM's embedding space where frozen attention responds. It can't teach the LM to attend to visual tokens — only approximate grounding through a fixed bottleneck. The model compensates by memorizing training-distribution patterns.

**Frozen layers amplify overfitting.** The projector solves two problems simultaneously: map features into a space the frozen LM reacts to, and produce correct output tokens. On small data it memorizes training receipts. LoRA on LM attention layers would allow the model to actually learn to attend to the visual prefix.

**Language mismatch.** TinyLlama is English-tuned; CORD is largely Indonesian. Qwen-2.5 improves vocabulary coverage but not grounding accuracy.

**Reward signal noise.** K=4 group-relative advantages — many steps skipped due to near-zero advantage variance when the policy is still learning.

## Scaling to Production

**Multi-page documents.** Donut at 640×960 isn't sufficient for dense forms. Context limits become a bottleneck at a few pages. Higher resolution, page tiling, or per-page encoding with merged extraction needed for bill of lading scale.

**Visual grounding.** Bills of lading have more structured layout than restaurant receipts. The grounding problem identified here gets worse. LoRA on LM attention layers is the minimum required change for production accuracy.

**Layout-aware encoder.** For complex Bill with multi-column tables fine-tuning a layout-aware encoder like Qwen-VL or LayoutLMv3 would likely be necessary to capture the spatial relationships between fields.

**Larger models.** TinyLlama at 1.1B is the main capacity constraint. A 7B+ instruction-tuned LM with LoRA adapters and the same projector architecture would likely close most of the grounding gap.

## Project Structure

```
vlm/
├── configs/
│   ├── paths.py
│   ├── training_configs.py     # named runs: debug, base, nodrop, long, tight, best, ca
│   └── training_schema.py      # dataclass schemas
├── data/
│   ├── collator.py
│   └── dataset.py              # CORD-native parser, drop tracking
├── models/
│   ├── language_model.py
│   ├── projector.py
│   ├── resampler.py
│   ├── receipt_vlm.py
│   └── vision_encoder.py
├── training/
│   ├── common.py
│   ├── generate.py
│   ├── rewards.py              # dynamic per-key grading, text hallucination penalty
│   ├── rl.py
│   ├── rl_utils.py
│   └── sft.py
├── evaluation/
│   └── evaluate.py             # format adherence, full structure rate, key coverage
├── notebooks/
│   └── walkthrough.py          # interactive Marimo walkthrough of the full pipeline
├── utils/
│   ├── device.py
│   ├── hf_login.py
│   ├── json_extractor.py
│   └── training.py
└── scripts/
    ├── train_sft.py            # --resume supported
    ├── train_rl.py             # --resume supported
    └── evaluate.py
```