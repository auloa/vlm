import random
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)

    np_random: Any = np.random
    np_random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[seed] set to {seed}")


def assert_only_projector_trainable(model: nn.Module) -> None:
    trainable = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]

    bad = [name for name in trainable if not name.startswith("projector.")]

    if bad:
        raise RuntimeError(f"❌ Non-projector params are trainable: {bad}")

    total = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[params] total: {total:,}")
    print(f"[params] trainable: {trainable_count:,}")
    print("✅ Only projector is trainable")


def get_autocast(device: torch.device) -> torch.autocast:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # MPS / CPU → no AMP (safer)
    return torch.autocast(device_type="cpu", enabled=False)