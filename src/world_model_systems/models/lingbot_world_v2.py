from __future__ import annotations

from pathlib import Path

from world_model_systems.core.spec import (
    Capability,
    LaunchRequest,
    LicensePolicy,
    ModelSpec,
    OptimizationSpec,
)
from world_model_systems.integrations.patching import SourcePatch, apply_source_patches
from world_model_systems.optimizations.catalog import STANDARD_OPTIMIZATIONS

from .base import BaseAdapter


class LingBotWorldV2Adapter(BaseAdapter):
    spec = ModelSpec(
        id="lingbot-world-v2",
        display_name="LingBot-World 2.0 14B causal-fast",
        family="Wan2.2 causal interactive world model",
        organization="Robbyant / Ant Group",
        repository="https://github.com/Robbyant/lingbot-world-v2.git",
        pinned_revision="2648877f763a06cc743bcd919936da4d25f12e7b",
        model_size="14B",
        license=LicensePolicy(
            "CC-BY-NC-SA-4.0",
            commercial_use=False,
            redistribution=True,
            hosted_service=False,
            notes="Non-commercial and share-alike. Upstream code and weights are never vendored.",
            acceptance_required=True,
        ),
        capabilities=frozenset(
            {
                Capability.CAUSAL,
                Capability.STREAMING,
                Capability.ACTION_CONDITIONED,
                Capability.CAMERA_CONDITIONED,
                Capability.KV_CACHE,
                Capability.FEW_STEP,
                Capability.SEQUENCE_PARALLEL,
                Capability.FSDP,
            }
        ),
        default_checkpoint="robbyant/lingbot-world-v2-14b-causal-fast",
        default_resolution=(480, 832),
        default_frames=361,
        optimizations=(
            STANDARD_OPTIMIZATIONS["few-step"],
            STANDARD_OPTIMIZATIONS["kv-cache"],
            STANDARD_OPTIMIZATIONS["fused-adaln"],
            STANDARD_OPTIMIZATIONS["fp32-causal-rope"],
            STANDARD_OPTIMIZATIONS["sequence-parallel"],
            OptimizationSpec(
                id="prewarm",
                category="runtime",
                objective=("latency",),
                description="Run LingBot's shape-matched prewarm outside the measured generation window.",
            ),
        ),
        notes="Released July 2026. Built on Alibaba Wan2.2; the repository is Robbyant/Ant Group, not Alibaba Research.",
    )

    def build_inference_command(self, request: LaunchRequest) -> list[str]:
        height, width = self.spec.default_resolution or (480, 832)
        action_path = request.image.parent
        command = [
            "torchrun",
            f"--nproc_per_node={request.num_gpus}",
            str(request.upstream / "generate.py"),
            "--task",
            "i2v-A14B",
            "--size",
            f"{height}*{width}",
            "--ckpt_dir",
            str(request.checkpoint),
            "--image",
            str(request.image),
            "--action_path",
            str(action_path),
            "--frame_num",
            str(request.frames or self.spec.default_frames),
            "--local_attn_size",
            "18",
            "--sink_size",
            "6",
            "--prompt",
            request.prompt,
        ]
        if request.num_gpus > 1:
            command += ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(request.num_gpus)]
        command += list(request.extra_args)
        return command

    def apply_optimization(self, optimization_id: str, upstream: Path, *, allow_unpinned: bool = False) -> list[Path]:
        if optimization_id == "fused-adaln":
            return apply_source_patches(
                model_id=self.spec.id,
                optimization_id=optimization_id,
                upstream=upstream,
                expected_revision=self.spec.pinned_revision,
                marker_name=".wms_fused_adaln.json",
                allow_unpinned=allow_unpinned,
                patches=(
                    SourcePatch(
                        path="wan/modules/model_fast.py",
                        import_anchor="import torch.nn.functional as torch_F\n",
                        import_line="from world_model_systems.optimizations.kernels.adaln import fused_adaln\n",
                        replacements=(
                            (
                                "self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)",
                                "fused_adaln(x, e[1].squeeze(2), e[0].squeeze(2), eps=self.norm1.eps)",
                            ),
                            (
                                "self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)",
                                "fused_adaln(x, e[4].squeeze(2), e[3].squeeze(2), eps=self.norm2.eps)",
                            ),
                            (
                                "self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)",
                                "fused_adaln(x, e[1].squeeze(2), e[0].squeeze(2), eps=self.norm.eps)",
                            ),
                        ),
                    ),
                ),
            )
        if optimization_id == "fp32-causal-rope":
            return apply_source_patches(
                model_id=self.spec.id,
                optimization_id=optimization_id,
                upstream=upstream,
                expected_revision=self.spec.pinned_revision,
                marker_name=".wms_fp32_causal_rope.json",
                allow_unpinned=allow_unpinned,
                patches=(
                    SourcePatch(
                        path="wan/modules/model_fast.py",
                        import_anchor="from .attention import flash_attention\n",
                        import_line="from world_model_systems.optimizations.kernels.causal_rope import causal_rope_apply_fp32\n",
                        replacements=((
                            "def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):",
                            "def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):\n    return causal_rope_apply_fp32(x, grid_sizes, freqs, start_frame)\n\ndef _wms_original_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):",
                        ),),
                    ),
                ),
            )
        return super().apply_optimization(optimization_id, upstream, allow_unpinned=allow_unpinned)
