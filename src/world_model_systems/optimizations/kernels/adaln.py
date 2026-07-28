"""Stable public import path for the model-agnostic AdaLN kernel."""

from matrixgame_systems.kernels.adaln import (
    adaln_reference,
    fused_adaln,
    fused_gated_residual,
    gated_residual_reference,
)

__all__ = [
    "adaln_reference",
    "fused_adaln",
    "fused_gated_residual",
    "gated_residual_reference",
]
