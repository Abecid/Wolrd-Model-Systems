from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WanDiTShape:
    sequence_length: int
    hidden_size: int = 5120
    ffn_hidden_size: int = 13824
    num_layers: int = 40
    text_length: int = 512


def estimate_wan_dit_forward_flops(shape: WanDiTShape) -> float:
    """Estimate dense DiT forward FLOPs.

    The estimate counts multiply-add as two FLOPs and includes self-attention,
    cross-attention, Q/K/V/O projections, and the two FFN projections. Elementwise
    normalization, modulation, RoPE, activation, and VAE FLOPs are omitted, so the
    result should be labeled an estimate rather than a hardware-counter truth.
    """

    length = shape.sequence_length
    width = shape.hidden_size
    ffn = shape.ffn_hidden_size
    text = shape.text_length

    self_projection = 8.0 * length * width * width
    self_attention = 4.0 * length * length * width
    cross_projection = 2.0 * (length * width * width + 2.0 * text * width * width)
    cross_attention = 4.0 * length * text * width
    feed_forward = 4.0 * length * width * ffn
    per_layer = self_projection + self_attention + cross_projection + cross_attention + feed_forward
    return per_layer * shape.num_layers


def estimate_training_flops(forward_flops: float, backward_multiplier: float = 2.0) -> float:
    return forward_flops * (1.0 + backward_multiplier)
