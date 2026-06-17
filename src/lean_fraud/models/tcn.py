"""Lean Temporal Convolutional Network (TCN) for sequence fraud scoring.

Causal, dilated 1D convolutions capture long transaction histories with few parameters and
fast, parallelizable inference — the efficiency thesis of this repo.
"""

from __future__ import annotations

import torch
from torch import nn


class _Chomp1d(nn.Module):
    """Trim the right padding so convolutions stay causal (no future leakage)."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    """Stacked temporal blocks + a binary fraud head.

    Args:
        n_features: number of features per transaction step.
        channels: hidden channels per temporal block (depth = len(channels)).
    """

    def __init__(
        self,
        n_features: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [64, 64, 64]
        layers: list[nn.Module] = []
        c_in = n_features
        for i, c_out in enumerate(channels):
            layers.append(_TemporalBlock(c_in, c_out, kernel_size, dilation=2**i, dropout=dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> logits: (batch,)."""
        x = x.transpose(1, 2)  # -> (batch, features, seq_len) for Conv1d
        feats = self.tcn(x)[:, :, -1]  # last timestep representation
        return self.head(feats).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
