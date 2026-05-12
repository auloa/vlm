import torch
import torch.nn as nn


class Projector(nn.Module):
    """Maps image embeddings into the LLM embedding space."""

    def __init__(self, vis_dim: int, llm_dim: int, projector_ffn_mult: int = 2, dropout:float|None=None):
        super().__init__()

        layers = [
            nn.Linear(vis_dim, llm_dim * projector_ffn_mult),
            nn.GELU(),
        ]
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(llm_dim * projector_ffn_mult, llm_dim),
        ])
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.LayerNorm(llm_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())
