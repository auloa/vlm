import torch
import torch.nn as nn


class Projector(nn.Module):
    """Maps image embeddings into the LLM embedding space."""

    def __init__(self, vis_dim: int, llm_dim: int, projector_ffn_mult: int = 4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(vis_dim, llm_dim * projector_ffn_mult),
            nn.GELU(),
            nn.Linear(llm_dim * projector_ffn_mult, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())
