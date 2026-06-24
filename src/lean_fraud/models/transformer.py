"""Heavy Transformer baseline (TranAD-like) — the model our lean TCN is benchmarked against.

Intentionally larger: the point is to show the TCN matches its quality with far fewer params
and lower latency. `dim_feedforward` is set explicitly (PyTorch defaults to 2048, a 16x FFN ratio
that bloats this small d_model to ~1.8M params / 27x the TCN — an unfairly heavy baseline). At
dim_feedforward=256 (2x ratio) the encoder is ~400K params, ~6x the TCN, mirroring the ~6x gap
between TranAD and TCN+HMM in Alejandro's FDI paper.
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> logits: (batch,)."""
        h = self.encoder(self.input_proj(x))
        return self.head(h[:, -1, :]).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
