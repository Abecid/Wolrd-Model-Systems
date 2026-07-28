from __future__ import annotations

from world_model_systems.core.spec import Capability, OptimizationSpec

STANDARD_OPTIMIZATIONS = {
    "fused-adaln": OptimizationSpec(
        id="fused-adaln",
        category="kernel",
        objective=("latency", "throughput"),
        description="Fuse affine-free LayerNorm and timestep modulation into one Triton kernel.",
    ),
    "fp32-causal-rope": OptimizationSpec(
        id="fp32-causal-rope",
        category="kernel",
        objective=("latency",),
        description="Use complex64 rather than complex128 for causal 3D RoPE.",
        experimental=True,
        capabilities=(Capability.CAUSAL,),
    ),
    "sequence-parallel": OptimizationSpec(
        id="sequence-parallel",
        category="parallelism",
        objective=("memory", "throughput", "latency"),
        description="Shard video tokens across Ulysses sequence-parallel ranks.",
        capabilities=(Capability.SEQUENCE_PARALLEL,),
    ),
    "few-step": OptimizationSpec(
        id="few-step",
        category="algorithm",
        objective=("latency", "throughput"),
        description="Use an upstream distilled few-step checkpoint.",
        capabilities=(Capability.FEW_STEP,),
    ),
    "kv-cache": OptimizationSpec(
        id="kv-cache",
        category="runtime",
        objective=("latency", "throughput"),
        description="Reuse causal attention keys and values across generated chunks.",
        capabilities=(Capability.KV_CACHE,),
    ),
    "async-vae": OptimizationSpec(
        id="async-vae",
        category="runtime",
        objective=("latency", "throughput"),
        description="Overlap VAE decoding with the next diffusion chunk.",
        capabilities=(Capability.ASYNC_VAE,),
    ),
    "quantization": OptimizationSpec(
        id="quantization",
        category="precision",
        objective=("memory", "latency", "throughput"),
        description="Use the model adapter's supported INT8/FP8 path.",
        capabilities=(Capability.QUANTIZATION,),
    ),
}
