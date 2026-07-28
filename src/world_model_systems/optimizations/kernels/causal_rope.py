from __future__ import annotations

import torch


def _apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: int, dtype: torch.dtype) -> torch.Tensor:
    heads = x.size(2)
    complex_width = x.size(3) // 2
    split = [complex_width - 2 * (complex_width // 3), complex_width // 3, complex_width // 3]
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    freq_parts = freqs.to(dtype=complex_dtype).split(split, dim=1)
    outputs = []
    for sample_index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        frames, height, width = int(frames), int(height), int(width)
        sequence_length = frames * height * width
        sample = torch.view_as_complex(
            x[sample_index, :sequence_length]
            .to(dtype=dtype)
            .reshape(sequence_length, heads, -1, 2)
        )
        sample_freqs = torch.cat(
            [
                freq_parts[0][start_frame : start_frame + frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                freq_parts[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                freq_parts[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(sequence_length, 1, -1)
        rotated = torch.view_as_real(sample * sample_freqs).flatten(2)
        outputs.append(torch.cat([rotated, x[sample_index, sequence_length:].to(rotated.dtype)]))
    return torch.stack(outputs).to(dtype=x.dtype)


def causal_rope_reference(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor:
    """LingBot-compatible complex128 reference."""
    return _apply(x, grid_sizes, freqs, start_frame, torch.float64)


def causal_rope_apply_fp32(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor:
    """Complex64 causal 3D RoPE.

    This removes the complex128 conversion in the upstream causal path. It is
    opt-in until the user validates long-horizon quality on real checkpoints.
    """
    return _apply(x, grid_sizes, freqs, start_frame, torch.float32)
