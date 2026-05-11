# vlm

Custom Vision-language model for structured data extraction from scanned receipts. Submitted for the Document AI Alignment take-home assignment.

## README Outline

- **Quick Start** — setup and how to run the pipeline
- **Architecture** — vision encoder choice, projector, visual token routing
- **Data** — CORD-v2, target schema, parser decisions
- **Training** — SFT and RL configs and procedures
- **Reward** — components and how they map to the eval metric
- **Evaluation** — what's measured and how
- **Results** — headline numbers and behavior notes
- **Design Decisions** — non-obvious choices, alternatives explored
- **Scaling to Production** — what would change for multi-page bills of lading
- **Limitations** — what's not addressed and why
- **Project Structure** — repo layout


## Quick Start

Requirements:

- Single GPU with 8–16 GB VRAM (tested on RTX 3090)
- Python 3.10+
- `uv` for dependency management
- HF token for downloading model weights

```bash
uv sync
python -m vlm.utils.hf_login

python -m vlm.scripts.train_sft -c tlama_sp # tlama_sp is run config name defined in vlm/configs/training_configs.py
python -m vlm.scripts.train_rl  -c tlama_sp
python -m vlm.scripts.evaluate  -c tlama_sp
```

TensorBoard:

```bash
tensorboard --logdir training_runs/tlama_sp/runs
```

A `-c debug` configuration runs the full pipeline on 20 samples in a few minutes for sanity-checking.


### Architecture

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
  instruction  ──► tokenize ──► LM embed ──► text embeddings
                                                                  ┘

  Frozen: vision encoder, language model. Trainable: projector.
```

### Vision encoder

`naver-clova-ix/donut-base-finetuned-cord-v2`. Donut is published as a `VisionEncoderDecoderModel` — a Swin transformer encoder paired with a BART-style text decoder, fine-tuned end-to-end on CORD for receipt parsing. For this pipeline only the encoder is needed, so the decoder is dropped right after loading:

```python
full_model = VisionEncoderDecoderModel.from_pretrained(model_name, dtype=self.model_dtype)
self.model = full_model.encoder
del full_model
```

Using the CORD-fine-tuned variant means the encoder features are already aware of the dataset's visual distribution rather than generic document layouts.

The image processor's resize step also gets overridden:

```python
processor = DonutProcessor.from_pretrained(model_name)
if not default_processor:
    processor.image_processor.size = {"height": img_shape[0], "width": img_shape[1]}
```

The default Donut processor targets a much higher resolution, producing close to 4800 visual tokens. Two problems with that:

- The vision encoder takes significant compute at that resolution
- The token sequence exceeds TinyLlama's context window once prompt and target are added

Resizing inputs to 640×960 cuts the count to a manageable 600. Attention pooling over the encoder output was also considered as a way to compress it, but the constraints around training make it unsuitable.

### Visual token routing

The standard LM forward pass expects token ids, which it feeds through an embedding lookup. Visual tokens have no corresponding ids — the projector output already lives in the LM's embedding space — so the lookup is bypassed entirely:

```python
visual_embeddings = self._get_visual_embeddings(images)
text_embeddings   = self._embed_input_ids(input_ids)

inputs_embeds = torch.cat([visual_embeddings, text_embeddings], dim=1)
```

The attention mask covers both halves:

```python
visual_attention_mask = torch.ones(batch_size, visual_len, device=self.device, dtype=torch.long)
full_attention_mask   = torch.cat([visual_attention_mask, attention_mask], dim=1)
```

Labels are masked to `-100` over the visual prefix and the instruction. Only the target JSON tokens contribute to the loss:

```python
visual_labels = torch.full((batch_size, visual_len), -100, device=self.device, dtype=torch.long)
full_labels   = torch.cat([visual_labels, labels], dim=1)
```

The LM is called with `inputs_embeds` instead of `input_ids`:

```python
self.lm.model(
    inputs_embeds=inputs_embeds,
    attention_mask=full_attention_mask,
    labels=full_labels,
)
```

### Projector

```python
nn.Sequential(
    nn.Linear(vis_dim, 2 * llm_dim),
    nn.GELU(),
    nn.Linear(2 * llm_dim, llm_dim),
    nn.LayerNorm(llm_dim),
)
```

A two-layer MLP that maps Donut features into the LM's embedding space. The output LayerNorm was added after early runs where the LM ignored the visual prefix entirely — normalizing the projector output brought it into a magnitude range the LM's frozen attention reacted to.

##### Note: A cross-attention resampler was also tried as a drop-in replacement: 64 learned queries attending to the visual features, compressing the sequence from 600 tokens to 64. It added significantly more trainable parameters but produced equivalent results, which pointed to the projector not being the bottleneck.

### Language model

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Small enough to fit alongside the vision encoder on a single consumer GPU, and instruction-tuned so it follows the prompt format consistently.

The instruction is wrapped at runtime through `tokenizer.apply_chat_template`, producing the role-tagged layout the model was trained on. An earlier hand-written prompt suffix caused the model to leak its own chat-template tokens into outputs.

##### Note: Qwen-2.5 was also tested — better multilingual coverage given CORD is largely Indonesian — but TinyLlama was kept as the primary model to stay within the assignment's small-LLM guidance and keep iteration fast.

## Data

CORD-v2 — scanned restaurant receipts with parsed JSON ground truth.

### Target schema

```json
{"line_items": [
  {"name": "GRILLED BABY POTATO ( R", "count": "1", "price": "50,500"},
  {"name": "HOT TUNA", "count": "1", "price": "67,000"}, 
  {"name": "HOT TUNA", "count": "1", "price": "67,000"}
], 
  "total": "148,127"}
```

- All fields are strings
- `count` is optional — only included when the receipt shows a quantity
- Sub-items like `"WELL DONE"` are folded into the parent name
- Prices are kept as-is; normalization happens at eval time

### Parser

Samples are filtered if they fail to parse, have no line items, have no total, or exceed the max target length. Per-reason counts are logged at startup.
Target length distribution across 800 samples shows a long tail — median around 74 tokens but p99 at 410 and a max of 654. Samples beyond the configured threshold are dropped to keep batches within the LM's context window and avoid truncating labels mid-JSON.

#### Target length distribution
| p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|
| 74 | 113 | 176 | 410 | 654 |

####  Maximum target tokens: 192 tokens — only 10% of samples are dropped at this point.


## Training

Two stages: SFT to teach the projector basic extraction, then RL to align the output format.

### SFT

Cross-entropy over the target JSON tokens. Visual prefix and instruction are masked out of the loss.

| Epochs | 15 |
|---|---|
| Batch size | 4 × 4 grad accumulation |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01 |
| LR schedule | Cosine with warmup |
| Mixed precision | bf16 on CUDA |

Best val-loss checkpoint saved to `sft/best.pt`. Training typically plateaus around epoch 10.

### RL

Starts from the SFT checkpoint. For each image:

1. Sample K=4 completions from the current policy
2. Score each with the reward function
3. Compute group-relative advantages `(R_i − μ) / (σ + ε)`
4. Skip if all rewards are equal — no preference signal
5. Optimize against a frozen copy of the SFT projector as reference

| Epochs | 2 |
|---|---|
| K (completions per image) | 4 |
| Optimizer | AdamW, lr 5e-6, weight decay 0.01 |
| KL coefficient β | 0.02 |
| Gradient clipping | 0.5 |

The KL term uses Schulman's k3 estimator applied per token, then masked-averaged. This is REINFORCE with group-relative advantages and a KL anchor — no PPO clipping, since with a single update step per batch and projector-only training it adds no real benefit.


## Reward

Four components, total clipped to [0, 1]:

| Component | Range | What it measures |
|---|---|---|
| format | 0–0.30 | 0.30 for strict JSON, 0.05 for parseable-but-wrapped, 0 otherwise |
| schema | 0–0.30 | Required top-level keys and well-formed line items |
| content | 0–0.30 | Line item count similarity, total digit match, name token overlap |
| hallucination | ≤ 0 | Duplicates, leaked chat tokens, repeated-character runs, excessive output |

The format reward is tied to the eval metric — the assignment defines format adherence as `json.loads` succeeding on the full output, so full credit only goes to strict JSON. Wrapped output gets a small partial credit so RL still has a gradient pointing at "drop the wrapper."

`count` is not among the required item keys. The dataset only emits it when the receipt shows a quantity, so requiring it would penalize outputs that correctly match the image.

## Evaluation

```bash
python -m vlm.scripts.evaluate -c tlama_sp
```

Runs both SFT and RL checkpoints on the held-out test split and writes a side-by-side comparison.

**Headline metric: `format_adherence_rate`** — percentage of outputs where `json.loads` succeeds on the full string, the result contains `line_items` and `total`, `line_items` is a list, and `total` is non-empty. This matches the assignment's required metric exactly.

Other metrics tracked per checkpoint:

- `strict_json_rate` — parseable without extraction
- `extractable_json_rate` — parseable after regex extraction
- `required_keys_rate`, `line_items_list_rate`, `total_present_rate`
- `total_match_rate` — digit-exact match with ground truth total
- `mean_reward` — same reward function used during RL

Outputs per stage written to `results/eval_{stage}/`:
- `summary.json` — all metrics
- `samples.jsonl` — per-sample scores and predictions
- `samples.md` — human-readable side-by-side for inspection
- `results/eval_comparison.json` — SFT vs RL delta

## Limitations

**Frozen base models.** The core constraint and the root cause of everything below. With both models frozen, the LM can't learn to attend to visual positions, adapt to Indonesian text, or update its representations for receipts. Every other limitation is a symptom.

**Visual grounding.** The LM samples from its prior over what a receipt looks like, lightly conditioned on visual features. Confirmed by two experiments: the resampler didn't help (bridge capacity isn't the issue), and Qwen shifted the output distribution but didn't fix the grounding. The real fix requires unfreezing part of the LM.

**Language mismatch.** TinyLlama is English-tuned; CORD is largely Indonesian. Qwen-2.5 improves vocabulary coverage but not grounding accuracy.

**Reward signal.** Format and schema saturate at SFT, leaving content as the only RL signal. Name token overlap is noisy; total digit match is reliable but fires on fewer samples.

**Hardware budget.** K=4 with batch=1. A significant fraction of RL steps get skipped due to near-zero advantage variance.

## Scaling to Production

**Multi-page documents.** Two options: concatenate visual tokens from all pages and run the LM once (simple, hits context limits around 5–6 pages); or encode each page independently and merge extractions in post-processing (cheaper, loses cross-page references). Either way, Donut at 640×960 isn't enough for dense forms — higher resolution or page tiling needed.

**Visual grounding.** Bills of lading are denser and more structured than restaurant receipts. The visual grounding problem identified here doesn't go away — it gets worse. LoRA on the LM's attention layers and a cross-attention connector are the minimum required changes for production-grade extraction accuracy.

**Layout-aware encoder.** For complex BoL with multi-column tables and many fields, swap Donut for LayoutLMv3 or DocFormerV2 — they encode bounding boxes, OCR text, and the image jointly, which matters when document structure carries semantic meaning.

**Schema flexibility.** BoL schemas vary by carrier and shipper. Include the target schema in the prompt at inference time, training the model to extract whatever schema it's shown. Or train per-document-type projector adapters routed by a lightweight document classifier upstream.