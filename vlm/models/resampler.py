
import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            kdim=kv_dim,
            vdim=kv_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Cross-attention with residual.
        attn_out, _ = self.cross_attn(
            query=self.norm_q(queries),
            key=self.norm_kv(kv),
            value=self.norm_kv(kv),
            need_weights=False,
        )
        queries = queries + attn_out

        # FFN with residual.
        queries = queries + self.ffn(self.norm_ffn(queries))

        return queries


class PerceiverResampler(nn.Module):
    """Compresses N visual tokens into num_queries tokens via cross-attention.

    Output is (batch, num_queries, llm_dim), which can be directly fed into the LLM as prefix tokens.
    """

    def __init__(
        self,
        vis_dim: int,
        llm_dim: int,
        num_queries: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_mult: int = 4,
    ):
        super().__init__()

        self.num_queries = num_queries

        # Learned query tokens. Small init so they don't dominate before training.
        self.queries = nn.Parameter(torch.randn(num_queries, llm_dim) * 0.02)

        self.blocks = nn.ModuleList(
            CrossAttentionBlock(
                dim=llm_dim,
                kv_dim=vis_dim,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
            )
            for _ in range(num_layers)
        )

        self.final_norm = nn.LayerNorm(llm_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        visual_features = visual_features.float()

        batch_size = visual_features.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        for block in self.blocks:
            queries = block(queries, visual_features)

        return self.final_norm(queries)