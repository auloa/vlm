import torch
import torch.nn as nn
from PIL import Image
from vlm.models.language_model import CausalLM
from vlm.models.resampler import PerceiverResampler
from vlm.models.projector import Projector
from vlm.models.vision_encoder import DonutVisionEncoder


class ReceiptVLM(nn.Module):
    """Frozen Donut encoder + trainable projector + frozen causal LM."""

    def __init__(
        self,
        device: torch.device | None = None,
        vision_model_name: str = "naver-clova-ix/donut-base",
        default_vision_processor: bool = False,
        image_height: int = 640,
        image_width: int = 960,
        lm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cross_attention_projector: bool = False,
        cross_attention_projector_num_queries: int = 64,
        cross_attention_projector_num_heads: int = 8,
        cross_attention_projector_num_layers: int = 2,
        cross_attention_projector_ffn_mult: int = 4,
        projector_mult: int = 4,
        freeze_vision: bool = True,
        freeze_lm: bool = True,
    ):
        super().__init__()

        self.freeze_vision = freeze_vision
        self.freeze_lm = freeze_lm

        self.device = device or torch.device("cpu")

        self.vision_encoder = DonutVisionEncoder(
            model_name=vision_model_name,
            device=self.device,
            default_processor=default_vision_processor,
            img_shape=(image_height, image_width),
            freeze=freeze_vision,
        )

        self.lm = CausalLM(
            model_name=lm_name,
            device=self.device,
            freeze=freeze_lm,
        )

        if cross_attention_projector:
            self.projector = PerceiverResampler(
                vis_dim=self.vision_encoder.hidden_size,
                llm_dim=self.lm.hidden_size,
                num_queries=64,
                num_heads=8,
                num_layers=2,
                ffn_mult=4,
            ).to(self.device)
        else:
            self.projector = Projector(
                vis_dim=self.vision_encoder.hidden_size,
                llm_dim=self.lm.hidden_size,
                projector_ffn_mult = projector_mult,
            ).to(self.device)

    def _get_visual_embeddings(self, images: list[Image.Image]) -> torch.Tensor:
        if self.freeze_vision:
            with torch.no_grad():
                visual_features = self.vision_encoder(images)
        else:
            visual_features = self.vision_encoder(images)
        return self.projector(visual_features)

    def _embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.freeze_lm:
            with torch.no_grad():
                embeddings = self.lm.model.get_input_embeddings()(input_ids)
        else:
            embeddings = self.lm.model.get_input_embeddings()(input_ids)
        return embeddings.float()

    def _verify_sequence_in_context_length(self, seq_len: int):
        max_pos = getattr(self.lm.model.config, "max_position_embeddings", None)

        if max_pos is not None and seq_len > max_pos:
            raise ValueError(
                f"Sequence length {seq_len} exceeds LLM context window {max_pos}."
            )
    def prepare_inputs_embeds(
        self,
        images: list[Image.Image],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(images)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        visual_embeddings = self._get_visual_embeddings(images)
        visual_len = visual_embeddings.shape[1]

        text_embeddings = self._embed_input_ids(input_ids)

        inputs_embeds = torch.cat(
            [visual_embeddings, text_embeddings],
            dim=1,
        )

        visual_attention_mask = torch.ones(
            batch_size,
            visual_len,
            device=self.device,
            dtype=torch.long,
        )

        full_attention_mask = torch.cat(
            [visual_attention_mask, attention_mask],
            dim=1,
        )

        self._verify_sequence_in_context_length(inputs_embeds.shape[1])

        inputs_embeds = inputs_embeds.to(dtype=self.lm.model_dtype)

        return inputs_embeds, full_attention_mask

    def forward(
        self,
        images: list[Image.Image],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        batch_size = len(images)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if labels is not None:
            labels = labels.to(self.device)

        inputs_embeds, full_attention_mask = self.prepare_inputs_embeds(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        visual_len = full_attention_mask.shape[1] - attention_mask.shape[1]

        if labels is not None:
            visual_labels = torch.full(
                (batch_size, visual_len),
                -100,
                device=self.device,
                dtype=torch.long,
            )

            full_labels = torch.cat(
                [visual_labels, labels],
                dim=1,
            )
        else:
            full_labels = None

        return self.lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )