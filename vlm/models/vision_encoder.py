import torch
import torch.nn as nn
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


class DonutVisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        device: torch.device | None = None,
        default_processor: bool = False,
        img_shape: tuple[int, int] = (640, 960),
        freeze: bool = True,
    ):
        super().__init__()

        self.device = device or torch.device("cpu")

        processor = DonutProcessor.from_pretrained(model_name)

        if not default_processor:
            processor.image_processor.size = {
                "height": img_shape[0],
                "width": img_shape[1],
            }

        self.processor = processor.image_processor
        self.model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        full_model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            dtype=self.model_dtype,
        )

        self.model = full_model.encoder
        del full_model

        self.model.to(self.device)
        self.model.requires_grad_(not freeze)

        if freeze:
            self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: (
                value.to(self.device, dtype=self.model_dtype)
                if value.is_floating_point()
                else value.to(self.device)
            )
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)
        return outputs.last_hidden_state