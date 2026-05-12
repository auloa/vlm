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
python -m vlm.scripts.train_sft -c tlama_sp_n_accum_bs_4_drp

# RL alignment (starts from SFT best checkpoint automatically)
python -m vlm.scripts.train_rl  -c tlama_sp_n_accum_bs_4_drp

# Evaluate both SFT and RL checkpoints on the held-out test split
python -m vlm.scripts.evaluate  -c tlama_sp_n_accum_bs_4_drp
```

Outputs land in `training_runs/<config_name>/`:

```
training_runs/tlama_sp_n_accum_bs_4_drp/
├── checkpoints/
│   ├── sft/best.pt          # best val-loss SFT checkpoint
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

A `debug` config runs the full pipeline on 20 samples in a few minutes — useful for verifying the environment before a full run:

```bash
python -m vlm.scripts.train_sft -c debug
python -m vlm.scripts.train_rl  -c debug
python -m vlm.scripts.evaluate  -c debug
```

### TensorBoard

```bash
tensorboard --logdir training_runs/tlama_sp_n_accum_bs_4_drp/runs
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
    cfg.sft.learning_rate = 3e-5
    cfg.rl.kl_coef = 0.30
    return cfg
```

Then run with `-c my_run`. The config name becomes the directory under `training_runs/`.

**Key config fields:**

| Section | Field                       | Default  | What it controls                                                                             |
|-------|-----------------------------|----------|----------------------------------------------------------------------------------------------|
| `sft` | `epochs`                    | 15       | Number of SFT training epochs                                                                |
| `sft` | `batch_size`                | 4        | Effective batch size (no accumulation)                                                       |
| `sft` | `grad_accum_steps`          | 1        | Steps to accumulate before optimizing (effective batch size = batch_size × grad_accum_steps) |
| `sft` | `learning_rate`             | 5e-5     | AdamW LR for SFT                                                                             |
| `sft` | `max_target_length`         | 256      | Max target tokens; longer samples are dropped                                                |
| `rl`  | `completions_per_image`     | 4        | K for group-relative advantages (must be ≥ 2)                                                |
| `rl`  | `learning_rate`             | 5e-6     | AdamW LR for RL                                                                              |
| `rl`  | `kl_coef`                   | 0.02     | KL penalty weight against SFT reference                                                      |
| `rl`  | `temperature`               | 0.7      | Sampling temperature for RL rollouts                                                         |
| `vision` | `image_height, image_width` | 640, 960 | Resize target for Donut processor                                                            |
| `projector` | `cross_attention`           | False    | To use cross attention sampler instead of MLP                                                |
| `eval` | `num_samples`               | 50       | Test samples for evaluation                                                                  |

The `dataclass` definitions for each section are in `vlm/configs/training_schema.py`.

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

`naver-clova-ix/donut-base-finetuned-cord-v2`. Donut is published as a `VisionEncoderDecoderModel` — a Swin transformer encoder paired with a BART-style text decoder, fine-tuned end-to-end on CORD for receipt parsing. For this pipeline only the encoder is needed:

```python
full_model = VisionEncoderDecoderModel.from_pretrained(model_name, dtype=self.model_dtype)
self.model = full_model.encoder
del full_model
```

The image processor's resize is overridden to 640×960. The default Donut processor targets ~2560×1920, producing ~4800 visual tokens that blow past TinyLlama's context window. At 640×960 the count drops to a manageable 600.

### Visual token routing

The LM's standard forward pass expects token ids via an embedding lookup. Visual tokens have no ids — the projector output already lives in the LM's embedding space — so the lookup is bypassed:

```python
visual_embeddings = self._get_visual_embeddings(images)
text_embeddings   = self._embed_input_ids(input_ids)
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

Two-layer MLP mapping Donut features into the LM's embedding space. The output LayerNorm was added after early runs where the LM ignored the visual prefix entirely — pre-norm projector outputs sat at magnitudes the frozen LM attention didn't react to.

### Language model

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Small enough to fit alongside the vision encoder on a single consumer GPU, instruction-tuned so it follows the prompt format consistently.

`tokenizer.apply_chat_template` is used at runtime. An earlier hand-written prompt suffix caused the model to leak its own role tokens into outputs.

## Data

CORD-v2 — scanned restaurant receipts with parsed JSON ground truth.

### Schema

The training target mirrors CORD's native `gt_parse` structure verbatim. Every field visible on the receipt has a place in the label:

```json
{
  "menu": [
    {
      "nm": "Nasi Campur Bali",
      "cnt": "1 x",
      "price": "75,000"
    }
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

All fields are optional — emitted only when the receipt shows them. Single-item receipts with no subtotal produce short targets; full-detail receipts produce all sections.

Using CORD's native keys directly means the model is trained to extract every visible line on the receipt. Renaming or dropping fields teaches the model to ignore parts of the image it can see — the "ignore this visible region" signal weakens the visual grounding prior across all samples.

### Parser

Samples are filtered if they fail to parse, have no menu items, or have no `total.total_price`. Per-reason counts and sample previews are logged at startup via `CORDDataset.print_drop_summary()`.

Roughly 25 samples (~3%) are dropped because the raw `total` dict lacks `total_price` — these are receipts where the visible total is stored under `cashprice` or `creditcardprice` instead of the canonical key. Dropped rather than implementing a fallback chain.

#### Target length distribution (new schema, 800 samples)

| p75 | p85 | p90 | p99 | max |
|-----|-----|-----|-----|-----|
| 189 | 231 | 270 | 584 | 714 |

`max_target_length = 256` drops ~8% of samples.

## Training

Two stages: SFT to teach the projector basic extraction, then RL to align output format and content coverage.

### SFT

Cross-entropy over the target JSON tokens. Visual prefix and instruction masked from the loss.

| Epochs | 15 |
|---|---|
| Batch size | 4 (no gradient accumulation) |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01 |
| LR schedule | Cosine with warmup |
| Mixed precision | bf16 on CUDA |
| Dropout | 0.1 in projector |

Best val-loss checkpoint saved to `sft/best.pt`.

### RL

Starts from the SFT checkpoint. For each image:

1. Sample K=4 completions from the current policy
2. Score each with the reward function
3. Compute group-relative advantages `(R_i − μ) / (σ + ε)`
4. Skip if all rewards are equal — no preference signal
5. Optimize against a frozen copy of the SFT projector as reference

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
| format | 0–0.30 | 0.30 for strict JSON, 0.10 for parseable-but-wrapped, 0 otherwise |
| schema | 0–0.30 | menu is a list, items have nm+price, total dict has total_price |
| content | 0–0.40 | dynamic per-key grading against GT structure |
| hallucination | ≤ 0 | extra keys, duplicate items, leaked chat tokens, repeated-character runs |

Content grading walks the GT recursively and scores each leaf key against the prediction. Money fields (digit-heavy values) use tolerant numeric match (strip separators). Text fields use token overlap. The combined score is `coverage × value_accuracy` — the model is rewarded for both finding the right keys and getting the values right.

The reward is dynamic: for each sample, only the keys present in the GT are scored. A model that produces extra keys (hallucinated fields) is penalized regardless of whether those keys appear in other samples' GTs.

## Evaluation

```bash
python -m vlm.scripts.evaluate -c tlama_sp_n_accum_bs_4_drp
```

**Headline metric: `format_adherence_rate`** — parseable strict JSON with `menu` (non-empty list) and `total.total_price` present.

Additional metrics:

- `strict_json_rate` — parseable without extraction
- `extractable_json_rate` — parseable after regex extraction
- `total_match_rate` — digit-exact match with GT `total.total_price`
- `mean_key_coverage` — fraction of GT leaf keys present in prediction
- `mean_value_accuracy` — average value match score across matched keys
- `mean_extra_keys` — avg hallucinated keys per sample

Outputs written to `results/eval_{stage}/`:

- `summary.json` — all metrics
- `samples.jsonl` — per-sample scores and predictions
- `samples.md` — human-readable side-by-side
- `results/eval_comparison.json` — SFT vs RL delta

## Results

Final evaluation on 50 held-out test images. Submitted model: `tlama_sp_n_accum_bs_4_drp` (MLP projector, dropout, new CORD-native schema).

| Metric | SFT | RL |
|---|---|---|
| format_adherence_rate | 4% | **6%** |
| extractable_json_rate | 16% | **28%** |
| required_keys_rate | 6% | **22%** |
| total_match_rate | 2% | **4%** |
| mean_key_coverage | 4.6% | **14.7%** |
| mean_value_accuracy | 3.2% | **9.0%** |
| mean_reward | 0.056 | **0.104** |

RL improves over SFT on every metric. The absolute numbers are low because the CORD-native schema is significantly harder than a flat `{line_items, total}` target — the model must extract 8–12 leaf keys per sample vs 3–4 previously, including nested structures (`sub_total`, per-item `sub`, optional fields like `discountprice`). The direction is correct: format adherence, key coverage, and value accuracy all improve after RL.

### SFT training behavior

![SFT training loss](assets/sft_loss.png)

No-dropout (blue/green): val bottoms at ~0.47 around epoch 9, then climbs to 0.53 — overfitting. Dropout (pink/orange): val still declining at epoch 15 at ~0.47 — hasn't converged yet. The dropout run is the submitted model; its best-val-loss checkpoint captures the most generalizable state within the 15-epoch budget.

The low format adherence (4% SFT) reflects an undertrained model. The richer schema needs more epochs to converge than the old flat schema did — the projector is learning to produce a more complex target with the same capacity and epoch budget.

### RL training behavior

![RL reward EMA](assets/rl_reward_ema.png)
![RL reward components](assets/rl_reward_components.png)

The reward EMA peaks around step 100 (~0.57), dips sharply around step 300, then partially recovers to ~0.45. Format component stays near its max (0.15) throughout — RL maintains format. Content component is noisier, reflecting the harder task. The best-EMA checkpoint is selected around the early peak (~step 100).

The EMA ceiling of ~0.57 vs the theoretical max of ~0.90 reflects the architectural bottleneck: with both base models frozen, the projector can only redistribute mass over outputs it can already produce, and those outputs are limited by how much the frozen LM attends to the visual prefix.

### Behavioral shift: SFT → RL

SFT outputs are mostly unstructured text that happens to contain some JSON fragments. RL outputs are recognizable structured JSON with correct key names even when values are wrong. This is the intended behavioral shift — RL teaches the model to produce the right schema even when it can't yet read the receipt accurately.

Sample where RL demonstrates correct schema structure (index 13):

```
GT:  {"menu": [{"nm": "SAMGYOPSAL", "cnt": "2", "price": "194,000"}, ...],
      "sub_total": {"subtotal_price": "369,000", "service_price": "18,450", "tax_price": "38,745"},
      "total": {"total_price": "426,195"}}

SFT: "menu * {"menu": [{"nm": "SAMI KOP"..." (malformed, truncated)

RL:  {"menu": [{"nm": "SAMI KOP", "cnt": "2", "price": "194,000"}, ...],
      "sub_total": {"subtotal_price": "194,000", "service_price": "16,000", "tax_price": "19,700"},
      "total": {"total_price": "218,700"}}
```

RL produces clean CORD-native JSON with all three sections. Values are wrong (visual grounding bottleneck) but the structure is correct.

## Design Decisions

**Donut over CLIP/SigLIP.** Donut's encoder is pretrained on document images with a text-decoding objective, giving it dense text-aware features. CLIP/SigLIP encode for object semantics, which is the wrong inductive bias for receipt text.

**LayerNorm at the projector output.** Added after early runs where the LM produced outputs entirely unaffected by the input image. Without it, projector outputs sat at magnitudes the frozen LM attention didn't react to. After LayerNorm, outputs became image-conditioned within the first epoch.

**CORD-native schema (verbatim keys).** The training target uses CORD's `gt_parse` keys directly (`nm`, `cnt`, `price`, `sub_total`, `total_price`, etc.) with no renaming. Every field visible on the receipt maps to a target token. This means the model is never trained to ignore visible content — any field present on the receipt has a place in the label. Renaming or dropping fields introduces "ignore this visual region" signal that weakens grounding.

**Dynamic reward grading.** The content reward walks the GT structure recursively and scores only keys that exist in that sample's GT. A model producing extra keys is penalized. This makes the reward schema-agnostic: adding new fields to the parser automatically changes what gets scored without touching the reward function.

**Optional fields in the schema.** Fields are only emitted when the receipt shows them. A single-item receipt with no subtotal produces `{"menu": [...], "total": {...}}`. A detailed receipt produces all sections. This prevents the model from learning to hallucinate fields the receipt doesn't show.

**Batch size 4 (no gradient accumulation).** With ~720 training samples and a fixed 15-epoch budget, smaller batches do ~4× more optimizer steps per epoch. The SFT loss comparisons show no-dropout batch-4 reaching val 0.47 vs larger-batch runs that don't converge within the same epoch budget.

**Dropout in the projector.** With dropout, the val curve stays flat instead of rising after epoch 9, at the cost of slower convergence. For the richer CORD-native schema, 15 epochs with dropout is insufficient to fully converge — the model is still learning at epoch 15. This is the primary limitation of the submitted model.

**GRPO with PPO clipping, `ppo_epochs=1` default.** The loop implements the clipped PPO surrogate with snapshotted old log-probs. At `ppo_epochs=1` clipping is a no-op. Tried `ppo_epochs=4`, no improvement — the policy moves too little per step at this LR for clipping to change anything.

### Alternatives explored

**Cross-attention resampler.** 64 learned queries attending to visual features (~92M params). Val loss bottomed earlier but then climbed steeply — a 128,000:1 parameter-to-sample ratio produces aggressive memorization. The resampler's attention can learn sample-specific visual fingerprints on 720 receipts. The MLP can't. Bridge capacity is not the bottleneck; the MLP stayed.

**Qwen-2.5 as the language model.** Output distributions shifted toward coherent Indonesian dish names instead of TinyLlama's phonetic English guesses. But the names still didn't match the actual receipt — the model was sampling from a better-matched language prior, not reading the image. Confirmed the bottleneck is visual grounding, not vocabulary.

**No-dropout SFT.** Reaches 24% format adherence in 15 epochs vs 4% for the dropout run, but val loss climbs from epoch 9 onward (overfitting). RL starting from the no-dropout SFT checkpoint consistently degraded all metrics across multiple runs — the policy was already in a sharp minimum and RL destabilized it rather than refining it. The dropout checkpoint is a flatter, more RL-stable starting point, even at lower absolute performance.

## Limitations

**Frozen base models.** The root cause of every other limitation. With both models frozen, the LM can't learn to attend to visual positions and the encoder can't adapt to the training distribution. Every limitation below is a symptom.

**Visual grounding ceiling.** The peak RL EMA reward (~0.57) is well below the theoretical max (~0.90). The projector must find the one subspace of the frozen LM's embedding space where the frozen attention happens to respond usefully — it can't teach the LM to attend to visual tokens, only approximate visual grounding through a fixed bottleneck. The model compensates by memorizing training-distribution patterns rather than reading the image.

**Frozen layers and overfitting interaction.** Because only the projector updates, it has to solve two problems simultaneously: map visual features into a space the frozen LM will attend to, and learn to produce the right output tokens. On small data the projector memorizes the training receipts rather than learning generalizable visual grounding. LoRA on LM attention layers would allow the model to actually learn to attend to the visual prefix.

**Schema complexity vs epoch budget.** The CORD-native schema requires more training epochs than the flat schema. The submitted dropout model hasn't converged at 15 epochs — val loss is still declining. 25–30 epochs with dropout would likely reach a better SFT starting point for RL.

**Reward signal noise.** K=4 group-relative advantages on a model that produces mostly non-JSON output means many steps are skipped (all rewards near-equal). When updates do fire, the signal is from structurally unusual samples, which may not generalize.

**Language mismatch.** TinyLlama is English-tuned; CORD is largely Indonesian. Qwen-2.5 improves vocabulary coverage but not grounding accuracy.

## Scaling to Production

**Multi-page documents.** Donut at 640×960 isn't sufficient for dense forms. Concatenating visual tokens from all pages hits context limits just a few pages. Per-page encoding with merged extraction in post-processing is cheaper but loses cross-page references. Higher resolution or page tiling needed for bill of lading kind of documents.

**Visual grounding.** Bills of lading are more structures with many different sections. The grounding problem identified here gets worse. LoRA on LM attention layers and a cross-attention connector inside the LM are the minimum changes required for production-grade extraction accuracy.

**Layout-aware encoder.** For complex lading bill with multi-column tables, we can use LayoutLMv3 or DocFormerV2 instead of Donut — they encode bounding boxes, OCR text, and image jointly, which matters when document structure carries semantic meaning.
** Larger models.
**Schema flexibility.** Bill schemas can vary a lot based on the source. The CORD-native schema approach (extract whatever fields are present) generalizes better than a fixed schema. For production, include the target schema in the prompt at inference time and train the model to extract whatever schema it's shown.

## Project Structure

```
vlm/
├── configs/
│   ├── paths.py
│   ├── training_configs.py     # named runs
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
│   ├── rewards.py              # dynamic per-key grading
│   ├── rl.py
│   ├── rl_utils.py
│   └── sft.py
├── evaluation/
│   └── evaluate.py             # key coverage + value accuracy metrics
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