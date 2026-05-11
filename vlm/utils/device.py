import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    # bf16 is preferred on cuda ampere+ and on mps (m-series handles it well)
    # fall back to fp32 on cpu
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32
