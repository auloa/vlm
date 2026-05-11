import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class CausalLM(nn.Module):
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: torch.device | None = None,
        freeze: bool = True,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.model_dtype,
        )

        self.model.to(self.device)
        self.model.requires_grad_(not freeze)
        if freeze:
            self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size
