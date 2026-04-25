"""Tiny MLP for the Neural Spacetime Field (training-time only).

The model is intentionally small (≈ 9k parameters) so CPU training
finishes in seconds. Position is rescaled by ``r_char`` and
acceleration by ``a_char`` *inside the forward pass* so the exported
ONNX speaks raw SI units — exactly what
:class:`cosmic_engine.ai.ONNXSpacetimeField` expects.
"""

from __future__ import annotations

import torch
from torch import nn


class SpacetimeMLP(nn.Module):
    """6-input → 3-output MLP with built-in input/output rescaling."""

    def __init__(
        self,
        r_char: float = 1.0e12,
        a_char: float = 6.674e-5,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.r_char = float(r_char)
        self.a_char = float(a_char)
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split position (raw m) and direction (already unit) and
        # normalize position so the MLP works in O(1)-magnitude land.
        position = x[..., :3] / self.r_char
        direction = x[..., 3:6]
        scaled_input = torch.cat([position, direction], dim=-1)
        scaled_output = self.net(scaled_input)
        return scaled_output * self.a_char
