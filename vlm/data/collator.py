from dataclasses import dataclass

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase


@dataclass
class CollatorOutput:
    images: list[Image.Image]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class CORDCollator:
    """Builds the language-side SFT batch for receipt JSON generation.

    input_ids:
        [instruction tokens] + [target JSON tokens]

    labels:
        [-100 for instruction] + [target JSON token ids]

    Padding positions are also set to -100.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        instruction: str,
        max_target_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_target_length = max_target_length

        self.instruction_ids = tokenizer(
            instruction,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"][0]

    def __call__(self, batch: list[dict]) -> CollatorOutput:
        images = [sample["image"] for sample in batch]

        eos = self.tokenizer.eos_token or ""
        label_strings = [sample["label"] + eos for sample in batch]

        target_tokens = self.tokenizer(
            label_strings,
            return_tensors="pt",
            padding="max_length",
            truncation=False, # length is controlled by max_target_length,
                              # and we want to ensure truncation happens at the end of the target, not the instruction
            max_length=self.max_target_length,
            add_special_tokens=False,
        )

        target_ids = target_tokens["input_ids"]
        target_mask = target_tokens["attention_mask"]

        batch_size = len(batch)
        instruction_len = self.instruction_ids.shape[0]

        instruction_ids = self.instruction_ids.unsqueeze(0).expand(batch_size, -1)
        instruction_mask = torch.ones(
            batch_size,
            instruction_len,
            dtype=torch.long,
        )

        input_ids = torch.cat([instruction_ids, target_ids], dim=1)
        attention_mask = torch.cat([instruction_mask, target_mask], dim=1)

        instruction_labels = torch.full(
            (batch_size, instruction_len),
            -100,
            dtype=torch.long,
        )

        target_labels = target_ids.clone()
        target_labels[target_mask == 0] = -100

        labels = torch.cat([instruction_labels, target_labels], dim=1)

        return CollatorOutput(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )