import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vlm.models.receipt_vlm import ReceiptVLM


def clone_reference_projector(projector: nn.Module) -> nn.Module:
    ref_projector = copy.deepcopy(projector)
    ref_projector.to(next(projector.parameters()).device)
    ref_projector.eval()
    ref_projector.requires_grad_(False)
    return ref_projector


def get_visual_embeddings_with_projector(
    model: ReceiptVLM,
    image: Image.Image,
    projector: nn.Module,
    require_grad: bool,
) -> torch.Tensor:
    """Encode image with frozen vision encoder, then apply selected projector.

    Gradients flow only through the projector when require_grad=True.
    """
    with torch.no_grad():
        visual_features = model.vision_encoder([image])

    if require_grad:
        visual_embeddings = projector(visual_features)
    else:
        with torch.no_grad():
            visual_embeddings = projector(visual_features)

    return visual_embeddings.float()


def compute_completion_token_log_probs(
    model: ReceiptVLM,
    image: Image.Image,
    completions: list[str],
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    projector: nn.Module | None = None,
    max_completion_length: int = 128,
    require_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probs for a batch of sampled completions.

    Sequence layout fed to the LM:
        visual tokens | prompt tokens | completion[:-1]

    The last prompt position predicts completion[0]; each completion
    position predicts the next completion token.

    Returns:
        token_log_probs: (K, T) — log-probs at each completion position
        target_mask:     (K, T) — 1 for real tokens, 0 for padding
    """
    device = model.device
    projector = projector or model.projector
    k = len(completions)

    tokens = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_completion_length,
        add_special_tokens=False,
    )

    completion_ids = tokens["input_ids"].to(device)
    completion_mask = tokens["attention_mask"].to(device).float()

    if completion_ids.shape[1] < 1:
        return (
            torch.zeros(k, 1, device=device),
            torch.zeros(k, 1, device=device),
        )

    if completion_ids.shape[1] > 1:
        completion_input_ids = completion_ids[:, :-1]
        completion_input_mask = completion_mask[:, :-1].long()
    else:
        completion_input_ids = completion_ids[:, :0]
        completion_input_mask = completion_mask[:, :0].long()

    target_ids = completion_ids
    target_mask = completion_mask

    visual_embeds = get_visual_embeddings_with_projector(
        model=model,
        image=image,
        projector=projector,
        require_grad=require_grad,
    )

    visual_embeds = visual_embeds.expand(k, -1, -1)

    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )

    prompt_ids = prompt_tokens["input_ids"].to(device)
    prompt_mask = prompt_tokens["attention_mask"].to(device)

    prompt_ids = prompt_ids.expand(k, -1)
    prompt_mask = prompt_mask.expand(k, -1)

    with torch.no_grad():
        prompt_embeds = model.lm.model.get_input_embeddings()(prompt_ids).float()

        if completion_input_ids.shape[1] > 0:
            completion_embeds = model.lm.model.get_input_embeddings()(
                completion_input_ids
            ).float()
        else:
            completion_embeds = torch.empty(
                k,
                0,
                prompt_embeds.shape[-1],
                device=device,
                dtype=prompt_embeds.dtype,
            )

    inputs_embeds = torch.cat(
        [visual_embeds, prompt_embeds, completion_embeds],
        dim=1,
    ).to(dtype=model.lm.model_dtype)

    visual_mask = torch.ones(
        k,
        visual_embeds.shape[1],
        device=device,
        dtype=torch.long,
    )

    attention_mask = torch.cat(
        [visual_mask, prompt_mask, completion_input_mask],
        dim=1,
    )

    if require_grad:
        outputs = model.lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    else:
        with torch.no_grad():
            outputs = model.lm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

    logits = outputs.logits

    visual_len = visual_embeds.shape[1]
    prompt_len = prompt_embeds.shape[1]

    start = visual_len + prompt_len - 1
    end = start + target_ids.shape[1]

    completion_logits = logits[:, start:end, :]
    log_probs = F.log_softmax(completion_logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * target_mask

    return token_log_probs, target_mask


def compute_grpo_loss(
    policy_token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    ref_token_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GRPO loss: clipped policy gradient with group-relative advantages and KL regularization.

    Policy term: PPO clipped surrogate. The ratio π_new / π_old is clipped to
    [1 - clip_eps, 1 + clip_eps], where π_old is the policy that generated the
    completions (snapshotted before any gradient step this batch). With
    ppo_epochs=1 the ratio is always 1 on the first step and clipping has no
    effect — it only bites on subsequent inner-loop steps.

    KL term: Schulman k3 estimator (exp(r) - r - 1) per token, masked-averaged,
    where r = log π_ref - log π_new. Computed per token rather than on sequence
    means because k3 is convex.

    Returns:
        loss:        total loss (policy + beta * kl)
        policy_loss: clipped policy gradient term
        kl_loss:     KL regularization term
    """
    advantages = advantages.detach()

    token_counts = token_mask.sum(dim=1).clamp_min(1.0)
    total_tokens = token_mask.sum().clamp_min(1.0)

    # Clipped policy gradient.
    log_ratio_old = policy_token_log_probs - old_token_log_probs.detach()
    ratio = torch.exp(log_ratio_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    adv = advantages.unsqueeze(-1)  # (K,) -> (K, 1) for broadcast over tokens
    per_token_obj = torch.min(ratio * adv, clipped_ratio * adv)

    policy_seq = (per_token_obj * token_mask).sum(dim=1) / token_counts
    policy_loss = -policy_seq.mean()

    # KL loss: per-token k3, masked mean.
    log_ratio_ref = ref_token_log_probs - policy_token_log_probs
    kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    kl_loss = (kl_per_token * token_mask).sum() / total_tokens

    loss = policy_loss + beta * kl_loss

    return loss, policy_loss, kl_loss