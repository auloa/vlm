from dataclasses import dataclass

import torch
from PIL import Image
from vlm.models.receipt_vlm import ReceiptVLM


@dataclass
class GenerationOutput:
    texts: list[str]


def generate_k_outputs(
    model: ReceiptVLM,
    image: Image.Image,
    tokenizer,
    instruction: str,
    k: int = 4,
    max_completion_tokens: int = 128,
    temperature: float = 0.8,
    do_sample: bool = True,
) -> GenerationOutput:
    device = model.device

    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = prompt_tokens["input_ids"].to(device)
    attention_mask = prompt_tokens["attention_mask"].to(device)

    if k > 1:
        input_ids = input_ids.expand(k, -1)
        attention_mask = attention_mask.expand(k, -1)

    images = [image] * k

    with torch.no_grad():
        inputs_embeds, full_attention_mask = model.prepare_inputs_embeds(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        max_length = inputs_embeds.shape[1] + max_completion_tokens # input embeding stay the same as we are not generating new tokens for the input, but we want to generate up to max_completion_tokens new tokens

        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention_mask,
            "max_length": max_length,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": True,
            "repetition_penalty": 1.1,
        }

        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95

        output = model.lm.model.generate(**generate_kwargs)

    texts = tokenizer.batch_decode(
        output.sequences,
        skip_special_tokens=True,
    )

    return GenerationOutput(texts=texts)