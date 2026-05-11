from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from vlm.configs.training_schema import TrainingConfig
from vlm.data.collator import CORDCollator
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.common import (
    build_instruction,
    ensure_dir,
    log_text,
    prepare_tokenizer,
    set_projector_only_trainable,
)
from vlm.utils.device import get_device
from vlm.utils.training import get_autocast, set_seed


def train_sft(cfg: TrainingConfig) -> None:
    data = cfg.data
    vision = cfg.vision
    model_cfg = cfg.model
    projector_cfg = cfg.projector
    sft = cfg.sft

    assert sft.grad_accum_steps >= 1, "grad_accum_steps must be >= 1"

    set_seed(42)
    device = get_device()
    checkpoint_dir = ensure_dir(cfg.sft_checkpoint_dir)

    print(f"device: {device}")
    print("loading model...")

    model = ReceiptVLM(
        device=device,
        vision_model_name=vision.model_name,
        default_vision_processor=vision.default_processor,
        image_height=vision.image_height,
        image_width=vision.image_width,
        lm_name=model_cfg.lm_name,
        cross_attention_projector=projector_cfg.cross_attention,
        cross_attention_projector_num_queries=projector_cfg.num_queries,
        cross_attention_projector_num_heads=projector_cfg.num_heads,
        cross_attention_projector_num_layers=projector_cfg.num_layers,
        cross_attention_projector_ffn_mult=projector_cfg.ffn_mult,
        projector_mult=projector_cfg.projector_mult,
    )

    tokenizer = prepare_tokenizer(model.lm.tokenizer)
    instruction = build_instruction(tokenizer, model_cfg.instruction)
    set_projector_only_trainable(model)

    print("loading datasets...")

    train_dataset = CORDDataset(
        split=data.train_split,
        max_samples=data.train_samples,
        dataset_name=data.dataset_name,
        tokenizer=tokenizer,
        max_target_length=sft.max_target_length,
    )

    val_dataset = CORDDataset(
        split=data.val_split,
        max_samples=data.val_samples,
        dataset_name=data.dataset_name,
        tokenizer=tokenizer,
        max_target_length=sft.max_target_length,
    )

    _print_dataset_stats("train", train_dataset)
    _print_dataset_stats("val", val_dataset)

    collator = CORDCollator(
        tokenizer=tokenizer,
        instruction=instruction,
        max_target_length=sft.max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=sft.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=sft.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        model.projector.parameters(),
        lr=sft.learning_rate,
        weight_decay=sft.weight_decay,
    )

    steps_per_epoch = (
        len(train_loader) + sft.grad_accum_steps - 1
    ) // sft.grad_accum_steps

    total_optimizer_steps = max(1, steps_per_epoch * sft.epochs)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_optimizer_steps // 10),
        num_training_steps=total_optimizer_steps,
    )

    global_step = 0
    best_val_loss = float("inf")

    with SummaryWriter(log_dir=str(cfg.sft_run_dir)) as writer:
        log_text(writer, "sft/log", f"device: {device}", step=0)

        for epoch in range(sft.epochs):
            model.projector.train()
            model.vision_encoder.eval()
            model.lm.model.eval()

            epoch_loss = 0.0
            valid_batches = 0

            optimizer.zero_grad(set_to_none=True)

            print(f"\nepoch {epoch + 1}/{sft.epochs}")

            for batch_idx, batch in enumerate(train_loader, start=1):
                with get_autocast(device):
                    output = model(
                        images=batch.images,
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        labels=batch.labels,
                    )

                output_loss: torch.Tensor = output.loss

                if not torch.isfinite(output_loss):
                    print(f"warning: invalid SFT loss at batch {batch_idx}; skipping")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Scale loss for gradient accumulation.
                (output_loss / sft.grad_accum_steps).backward()

                # Store the real, unscaled loss for logging.
                epoch_loss += output_loss.item()
                valid_batches += 1

                should_step = batch_idx % sft.grad_accum_steps == 0
                is_last_batch = batch_idx == len(train_loader)

                if not (should_step or is_last_batch):
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.projector.parameters(),
                    max_norm=sft.grad_clip_norm,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % sft.log_every == 0:
                    avg_loss = epoch_loss / max(1, valid_batches)
                    lr = scheduler.get_last_lr()[0]

                    writer.add_scalars(
                        "sft/loss",
                        {
                            "step": output_loss.item(),
                            "running_avg": avg_loss,
                        },
                        global_step,
                    )
                    writer.add_scalar("sft/optim/lr", lr, global_step)
                    writer.add_scalar(
                        "sft/optim/grad_norm",
                        float(grad_norm),
                        global_step,
                    )

                    print(
                        f"step {global_step:4d} | "
                        f"batch {batch_idx}/{len(train_loader)} | "
                        f"loss {output_loss.item():.4f} | "
                        f"avg {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"grad {float(grad_norm):.2f}"
                    )

                if global_step % sft.sample_every == 0:
                    _log_sample(
                        model=model,
                        batch=batch,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        device=device,
                        global_step=global_step,
                        writer=writer,
                        max_completion_tokens=128,
                    )

            train_loss = epoch_loss / max(1, valid_batches)
            val_loss = _validate(model, val_loader)

            writer.add_scalars(
                "sft/epoch_loss",
                {
                    "train": train_loss,
                    "val": val_loss,
                },
                epoch + 1,
            )

            epoch_msg = (
                f"epoch {epoch + 1} complete | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )
            log_text(writer, "sft/log", epoch_msg, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:02d}.pt"

                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    val_loss=val_loss,
                    checkpoint_path=epoch_checkpoint_path,
                )

                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    val_loss=val_loss,
                    checkpoint_path=cfg.sft_best_checkpoint,
                )

                log_text(
                    writer,
                    "sft/log",
                    (
                        f"checkpoint saved: {cfg.sft_best_checkpoint} "
                        f"(val loss {val_loss:.4f})"
                    ),
                    step=global_step,
                )

    print("SFT training complete")


def _print_dataset_stats(name: str, dataset: CORDDataset) -> None:
    print(
        f"{name} dataset: {len(dataset)} samples | "
        f"parse failed: {dataset.num_parse_failed} | "
        f"too long: {dataset.num_too_long} | "
        f"empty items: {dataset.num_empty_items}"
    )


def _validate(model, val_loader) -> float:
    was_training = model.projector.training

    model.projector.eval()
    model.vision_encoder.eval()
    model.lm.model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            with get_autocast(model.device):
                output = model(
                    images=batch.images,
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )

            total_loss += output.loss.item()

    if was_training:
        model.projector.train()

    return total_loss / max(1, len(val_loader))


def _log_sample(
    model,
    batch,
    tokenizer,
    instruction: str,
    device,
    global_step: int,
    writer,
    max_completion_tokens: int,
) -> None:
    was_training = model.projector.training

    model.projector.eval()
    model.vision_encoder.eval()
    model.lm.model.eval()

    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = prompt_tokens["input_ids"].to(device)
    attention_mask = prompt_tokens["attention_mask"].to(device)

    with torch.no_grad():
        inputs_embeds, full_attention_mask = model.prepare_inputs_embeds(
            images=[batch.images[0]],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # For inputs_embeds, max_length is total prefix length + new tokens.
        generated = model.lm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_length=inputs_embeds.shape[1] + max_completion_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
    )

    # Recover the ground-truth target string from labels.
    # Labels have -100 over the instruction prefix and padding; the remaining
    # positions are the target JSON token ids.
    label_ids = batch.labels[0]
    target_ids = label_ids[label_ids != -100]
    ground_truth = tokenizer.decode(target_ids, skip_special_tokens=True)

    writer.add_text(
        "sft/samples/output",
        f"**ground truth:**\n```\n{ground_truth}\n```\n\n"
        f"**prediction:**\n```\n{decoded}\n```",
        global_step,
    )

    print(f"\nsample @ step {global_step}:")
    print(f"  gt:   {ground_truth[:200]}")
    print(f"  pred: {decoded[:200]}")

    if was_training:
        model.projector.train()


def _save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    val_loss: float,
    checkpoint_path: str | Path,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "projector_state_dict": model.projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )