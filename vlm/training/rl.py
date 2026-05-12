from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vlm.configs.training_schema import RLConfig, TrainingConfig
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.common import (
    build_instruction,
    ensure_dir,
    log_text,
    prepare_tokenizer,
    set_projector_only_trainable,
)
from vlm.training.generate import GenerationOutput, generate_k_outputs
from vlm.training.rewards import RewardBreakdown, compute_reward
from vlm.training.rl_utils import (
    clone_reference_projector,
    compute_completion_token_log_probs,
    compute_grpo_loss,
)
from vlm.utils.device import get_device
from vlm.utils.training import get_autocast, set_seed


def train_rl(cfg: TrainingConfig, resume: bool = False) -> None:
    data = cfg.data
    vision = cfg.vision
    model_cfg = cfg.model
    projector_cfg = cfg.projector
    rl = cfg.rl

    device = get_device()
    set_seed(42)

    assert rl.completions_per_image >= 2, (
        "completions_per_image must be >= 2 because this RL loop uses "
        "group-relative advantages."
    )

    ensure_dir(cfg.rl_checkpoint_dir)

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
        projector_dropout=projector_cfg.dropout,
    )

    # Always start from SFT best checkpoint as the base weights.
    ckpt = torch.load(cfg.sft_best_checkpoint, map_location=device)
    model.projector.load_state_dict(ckpt["projector_state_dict"])
    print(f"loaded SFT checkpoint: {cfg.sft_best_checkpoint}")

    set_projector_only_trainable(model)

    # Frozen SFT projector used as KL/reference policy.
    ref_projector = clone_reference_projector(model.projector)

    print("loading dataset...")

    dataset = CORDDataset(
        split=data.train_split,
        max_samples=data.train_samples,
        dataset_name=data.dataset_name,
    )

    print(f"dataset ready: {len(dataset)} samples")

    tokenizer = prepare_tokenizer(model.lm.tokenizer)
    instruction = build_instruction(tokenizer, model_cfg.instruction)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=_rl_collate,
    )

    optimizer = AdamW(
        model.projector.parameters(),
        lr=rl.learning_rate,
        weight_decay=rl.weight_decay,
    )

    global_step = 0
    ema_reward: float | None = None
    best_ema_reward = -float("inf")
    best_ema_for_stopping = -float("inf")
    steps_since_improvement = 0
    stop_training = False

    # ── Resume ──────────────────────────────────────────────────────────
    if resume:
        rl_ckpt_path = Path(cfg.rl_best_checkpoint)
        if not rl_ckpt_path.exists():
            print("[resume] no RL checkpoint found — starting from SFT checkpoint")
        else:
            print(f"[resume] loading {rl_ckpt_path}")
            rl_ckpt = torch.load(rl_ckpt_path, map_location=device)
            # Overwrite projector weights with the RL checkpoint
            # (SFT weights loaded above are replaced here).
            model.projector.load_state_dict(rl_ckpt["projector_state_dict"])
            optimizer.load_state_dict(rl_ckpt["optimizer_state_dict"])
            global_step = rl_ckpt.get("step", 0)
            best_ema_reward = rl_ckpt.get("mean_reward", -float("inf"))
            best_ema_for_stopping = best_ema_reward
            print(
                f"[resume] step {global_step} | "
                f"best ema reward {best_ema_reward:.4f}"
            )
    # ────────────────────────────────────────────────────────────────────

    with SummaryWriter(log_dir=str(cfg.rl_run_dir)) as writer:
        log_text(writer, "rl/log", f"device: {device}", step=0)

        for epoch in range(rl.epochs):
            if stop_training:
                break

            model.projector.train()
            epoch_rewards: list[float] = []

            print(f"\nepoch {epoch + 1}/{rl.epochs}")

            for step, sample in enumerate(loader):
                if global_step >= rl.max_steps:
                    print(f"max_steps ({rl.max_steps}) reached — stopping.")
                    stop_training = True
                    break

                image = sample["image"][0]
                ground_truth = sample["label"][0]

                # 1. Generate K completions from the current policy.
                model.projector.eval()

                with torch.no_grad():
                    gen = generate_k_outputs(
                        model=model,
                        image=image,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        k=rl.completions_per_image,
                        max_completion_tokens=rl.max_completion_tokens,
                        temperature=rl.temperature,
                        do_sample=True,
                    )

                model.projector.train()

                # 2. Compute rewards.
                breakdowns = [
                    compute_reward(text, ground_truth)
                    for text in gen.texts
                ]

                rewards = torch.tensor(
                    [b.total for b in breakdowns],
                    dtype=torch.float32,
                    device=device,
                )

                epoch_rewards.extend(rewards.tolist())

                # 3. Group-relative advantages.
                reward_mean = rewards.mean()
                reward_std = rewards.std(unbiased=False)
                advantages = (rewards - reward_mean) / (reward_std + 1e-8)

                # If all completions receive the same reward, there is no
                # useful preference signal for this image.
                if torch.allclose(
                    advantages,
                    torch.zeros_like(advantages),
                    atol=1e-6,
                ):
                    global_step += 1

                    ema_reward, best_ema_reward, best_ema_for_stopping, steps_since_improvement, stop_training = _step_bookkeeping(
                        model=model, optimizer=optimizer, writer=writer, cfg=cfg, rl=rl,
                        epoch=epoch, global_step=global_step,
                        step_reward=rewards.mean().item(),
                        ema_reward=ema_reward, best_ema_reward=best_ema_reward,
                        best_ema_for_stopping=best_ema_for_stopping,
                        steps_since_improvement=steps_since_improvement,
                    )

                    if global_step % rl.log_every == 0:
                        _log_metrics(
                            writer=writer,
                            global_step=global_step,
                            rewards=rewards,
                            breakdowns=breakdowns,
                            policy_loss=None,
                            kl_loss=None,
                            loss=None,
                            step=step,
                            total_steps=len(loader),
                            skipped=True,
                        )

                    if global_step % rl.sample_every == 0:
                        _log_sample(
                            gen=gen,
                            rewards=rewards,
                            ground_truth=ground_truth,
                            global_step=global_step,
                            writer=writer,
                        )

                    if stop_training:
                        break

                    continue

                # 4. Score completions under frozen SFT reference projector.
                with torch.no_grad():
                    ref_token_log_probs, token_mask = compute_completion_token_log_probs(
                        model=model,
                        image=image,
                        completions=gen.texts,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        projector=ref_projector,
                        max_completion_length=rl.max_completion_tokens,
                        require_grad=False,
                    )

                # 5. Snapshot old policy log-probs before any gradient step.
                with torch.no_grad():
                    old_token_log_probs, _ = compute_completion_token_log_probs(
                        model=model,
                        image=image,
                        completions=gen.texts,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        projector=model.projector,
                        max_completion_length=rl.max_completion_tokens,
                        require_grad=False,
                    )

                # 6. PPO inner loop.
                valid_step = True
                for _ppo_step in range(rl.ppo_epochs):
                    with get_autocast(device):
                        policy_token_log_probs, token_mask = compute_completion_token_log_probs(
                            model=model,
                            image=image,
                            completions=gen.texts,
                            tokenizer=tokenizer,
                            instruction=instruction,
                            projector=model.projector,
                            max_completion_length=rl.max_completion_tokens,
                            require_grad=True,
                        )

                        loss, policy_loss, kl_loss = compute_grpo_loss(
                            policy_token_log_probs=policy_token_log_probs,
                            old_token_log_probs=old_token_log_probs,
                            ref_token_log_probs=ref_token_log_probs,
                            token_mask=token_mask,
                            advantages=advantages,
                            beta=rl.kl_coef,
                            clip_eps=rl.clip_eps,
                        )

                    if not torch.isfinite(loss) or loss.abs() > 100:
                        print(
                            f"warning: invalid RL loss at step {global_step}; "
                            f"loss={loss.item() if torch.isfinite(loss) else loss}"
                        )
                        optimizer.zero_grad(set_to_none=True)
                        valid_step = False
                        break

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.projector.parameters(),
                        max_norm=rl.grad_clip_norm,
                    )

                    optimizer.step()

                global_step += 1

                if not valid_step:
                    continue

                ema_reward, best_ema_reward, best_ema_for_stopping, steps_since_improvement, stop_training = _step_bookkeeping(
                    model=model, optimizer=optimizer, writer=writer, cfg=cfg, rl=rl,
                    epoch=epoch, global_step=global_step,
                    step_reward=rewards.mean().item(),
                    ema_reward=ema_reward, best_ema_reward=best_ema_reward,
                    best_ema_for_stopping=best_ema_for_stopping,
                    steps_since_improvement=steps_since_improvement,
                )

                # 7. Logging.
                if global_step % rl.log_every == 0:
                    _log_metrics(
                        writer=writer,
                        global_step=global_step,
                        rewards=rewards,
                        breakdowns=breakdowns,
                        policy_loss=policy_loss,
                        kl_loss=kl_loss,
                        loss=loss,
                        step=step,
                        total_steps=len(loader),
                        skipped=False,
                    )

                    writer.add_scalar(
                        "rl/optim/grad_norm",
                        float(grad_norm),
                        global_step,
                    )

                if global_step % rl.sample_every == 0:
                    _log_sample(
                        gen=gen,
                        rewards=rewards,
                        ground_truth=ground_truth,
                        global_step=global_step,
                        writer=writer,
                    )

                if stop_training:
                    break

            mean_epoch_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
            writer.add_scalar("rl/epoch_mean_reward", mean_epoch_reward, epoch + 1)
            log_text(
                writer, "rl/log",
                f"epoch {epoch + 1} complete | mean reward {mean_epoch_reward:.3f} | ema {ema_reward:.3f}",
                step=global_step,
            )

    print("RL training complete")


def _step_bookkeeping(
    model: ReceiptVLM,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    cfg: TrainingConfig,
    rl: RLConfig,
    epoch: int,
    global_step: int,
    step_reward: float,
    ema_reward: float | None,
    best_ema_reward: float,
    best_ema_for_stopping: float,
    steps_since_improvement: int,
) -> tuple[float, float, float, int, bool]:
    """Update EMA, checkpoint, and check stopping conditions.

    Returns updated (ema_reward, best_ema_reward, best_ema_for_stopping,
    steps_since_improvement, stop_training).
    """
    # EMA update.
    if ema_reward is None:
        ema_reward = step_reward
    else:
        ema_reward = rl.ema_alpha * step_reward + (1 - rl.ema_alpha) * ema_reward

    writer.add_scalar("rl/reward/ema", ema_reward, global_step)

    # Save on EMA improvement.
    if ema_reward > best_ema_reward:
        best_ema_reward = ema_reward
        _save_rl_checkpoint(
            model=model, optimizer=optimizer, epoch=epoch, step=global_step,
            mean_reward=ema_reward, checkpoint_path=cfg.rl_best_checkpoint,
            completions_per_image=rl.completions_per_image, kl_coef=rl.kl_coef,
        )
        log_text(
            writer, "rl/log",
            f"checkpoint saved: step {global_step} | ema_reward {ema_reward:.3f}",
            step=global_step,
        )

    # Periodic step checkpoint.
    if global_step % rl.save_every_n_steps == 0:
        step_path = Path(cfg.rl_checkpoint_dir) / f"step_{global_step:06d}.pt"
        _save_rl_checkpoint(
            model=model, optimizer=optimizer, epoch=epoch, step=global_step,
            mean_reward=ema_reward, checkpoint_path=step_path,
            completions_per_image=rl.completions_per_image, kl_coef=rl.kl_coef,
        )

    # Early stopping.
    if ema_reward > best_ema_for_stopping + rl.early_stop_min_delta:
        best_ema_for_stopping = ema_reward
        steps_since_improvement = 0
    else:
        steps_since_improvement += 1

    stop_training = steps_since_improvement >= rl.early_stop_patience
    if stop_training:
        print(
            f"early stopping: no EMA improvement > {rl.early_stop_min_delta} "
            f"for {rl.early_stop_patience} steps."
        )

    return ema_reward, best_ema_reward, best_ema_for_stopping, steps_since_improvement, stop_training


def _log_metrics(
    writer: SummaryWriter,
    global_step: int,
    rewards: torch.Tensor,
    breakdowns: list[RewardBreakdown],
    policy_loss: torch.Tensor | None,
    kl_loss: torch.Tensor | None,
    loss: torch.Tensor | None,
    step: int,
    total_steps: int,
    skipped: bool = False,
) -> None:
    mean_reward = rewards.mean().item()

    mean_format = sum(b.format for b in breakdowns) / max(1, len(breakdowns))
    mean_schema = sum(b.schema for b in breakdowns) / max(1, len(breakdowns))
    mean_content = sum(b.content for b in breakdowns) / max(1, len(breakdowns))
    mean_hallucination = sum(b.hallucination for b in breakdowns) / max(
        1, len(breakdowns)
    )

    clean_json_rate = sum(
        b.format >= 0.15 for b in breakdowns
    ) / max(1, len(breakdowns))

    parseable_or_wrapped_json_rate = sum(
        b.format >= 0.05 for b in breakdowns
    ) / max(1, len(breakdowns))

    schema_rate = sum(
        b.schema >= 0.15 for b in breakdowns
    ) / max(1, len(breakdowns))

    writer.add_scalar("rl/reward/mean", mean_reward, global_step)
    writer.add_scalar("rl/reward/clean_json_rate", clean_json_rate, global_step)
    writer.add_scalar(
        "rl/reward/parseable_or_wrapped_json_rate",
        parseable_or_wrapped_json_rate,
        global_step,
    )
    writer.add_scalar("rl/reward/schema_rate", schema_rate, global_step)

    writer.add_scalar("rl/reward_components/format", mean_format, global_step)
    writer.add_scalar("rl/reward_components/schema", mean_schema, global_step)
    writer.add_scalar("rl/reward_components/content", mean_content, global_step)
    writer.add_scalar(
        "rl/reward_components/hallucination",
        mean_hallucination,
        global_step,
    )

    if skipped:
        writer.add_scalar("rl/update/skipped", 1.0, global_step)
        print(
            f"step {global_step:4d} | "
            f"batch {step + 1}/{total_steps} | "
            f"reward {mean_reward:.3f} | "
            f"json {clean_json_rate:.0%} | "
            f"schema {schema_rate:.0%} | "
            f"skipped update: identical rewards"
        )
        return

    writer.add_scalar("rl/update/skipped", 0.0, global_step)
    writer.add_scalar("rl/loss/policy", policy_loss.item(), global_step)
    writer.add_scalar("rl/loss/kl", kl_loss.item(), global_step)
    writer.add_scalar("rl/loss/total", loss.item(), global_step)

    print(
        f"step {global_step:4d} | "
        f"batch {step + 1}/{total_steps} | "
        f"reward {mean_reward:.3f} | "
        f"json {clean_json_rate:.0%} | "
        f"schema {schema_rate:.0%} | "
        f"format {mean_format:.3f} | "
        f"content {mean_content:.3f} | "
        f"hallucination {mean_hallucination:.3f} | "
        f"policy {policy_loss.item():.4f} | "
        f"kl {kl_loss.item():.4f} | "
        f"loss {loss.item():.4f}"
    )


def _log_sample(
    gen: GenerationOutput,
    rewards: torch.Tensor,
    ground_truth: str,
    global_step: int,
    writer: SummaryWriter,
) -> None:
    best_idx = rewards.argmax().item()
    worst_idx = rewards.argmin().item()

    log_text_str = (
        f"**ground truth:** {ground_truth[:300]}\n\n"
        f"**best reward={rewards[best_idx]:.3f}:**\n"
        f"{gen.texts[best_idx][:500]}\n\n"
        f"**worst reward={rewards[worst_idx]:.3f}:**\n"
        f"{gen.texts[worst_idx][:500]}"
    )

    writer.add_text("rl/samples", log_text_str, global_step)

    print(
        f"\nbest @ step {global_step} (reward={rewards[best_idx]:.3f}):\n"
        f"  gt:   {ground_truth[:200]}\n"
        f"  pred: {gen.texts[best_idx][:200]}"
    )


def _save_rl_checkpoint(
    model: ReceiptVLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    mean_reward: float,
    checkpoint_path: str | Path,
    completions_per_image: int,
    kl_coef: float,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "mean_reward": mean_reward,
            "projector_state_dict": model.projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "algorithm": "group_relative_policy_gradient_projector_only",
            "completions_per_image": completions_per_image,
            "kl_coef": kl_coef,
        },
        checkpoint_path,
    )


def _rl_collate(batch: list[dict]) -> dict[str, list]:
    return {
        "image": [sample["image"] for sample in batch],
        "label": [sample["label"] for sample in batch],
    }