from __future__ import annotations

from pathlib import Path

from world_model_systems.core.spec import Capability, LaunchRequest, LicensePolicy, ModelSpec
from world_model_systems.integrations.patching import SourcePatch, apply_source_patches
from world_model_systems.optimizations.catalog import STANDARD_OPTIMIZATIONS

from .base import BaseAdapter


class MatrixGameAdapter(BaseAdapter):
    spec = ModelSpec(
        id="matrix-game-3",
        display_name="Matrix-Game 3.0 5B",
        family="Wan2.2 interactive world model",
        organization="SkyworkAI",
        repository="https://github.com/SkyworkAI/Matrix-Game.git",
        pinned_revision="71c3cd7f741311f8100f6cf9cde942b6c1378d11",
        model_size="5B",
        license=LicensePolicy("Apache-2.0", True, True, True),
        capabilities=frozenset(
            {
                Capability.CAUSAL,
                Capability.STREAMING,
                Capability.ACTION_CONDITIONED,
                Capability.CAMERA_CONDITIONED,
                Capability.FEW_STEP,
                Capability.SEQUENCE_PARALLEL,
                Capability.FSDP,
                Capability.ASYNC_VAE,
                Capability.QUANTIZATION,
            }
        ),
        default_checkpoint="Skywork/Matrix-Game-3.0",
        default_resolution=(704, 1280),
        default_frames=97,
        optimizations=tuple(
            STANDARD_OPTIMIZATIONS[key]
            for key in ("few-step", "fused-adaln", "sequence-parallel", "async-vae", "quantization")
        ),
    )

    def build_inference_command(self, request: LaunchRequest) -> list[str]:
        subdir = request.upstream / "Matrix-Game-3"
        root = subdir if subdir.exists() else request.upstream
        iterations = max(1, ((request.frames or 57) - 57 + 39) // 40 + 1)
        command = [
            "torchrun",
            f"--nproc_per_node={request.num_gpus}",
            str(root / "generate.py"),
            "--size",
            f"{self.spec.default_resolution[0]}*{self.spec.default_resolution[1]}",
            "--ckpt_dir",
            str(request.checkpoint),
            "--image",
            str(request.image),
            "--prompt",
            request.prompt,
            "--output_dir",
            str(request.output_dir),
            "--num_iterations",
            str(iterations),
            "--num_inference_steps",
            "3",
            "--fa_version",
            "3",
            "--compile_vae",
        ]
        if request.num_gpus > 1:
            command += ["--ulysses_size", str(request.num_gpus), "--dit_fsdp", "--t5_fsdp"]
        command += list(request.extra_args)
        return command

    def apply_optimization(self, optimization_id: str, upstream: Path, *, allow_unpinned: bool = False) -> list[Path]:
        if optimization_id != "fused-adaln":
            return super().apply_optimization(optimization_id, upstream, allow_unpinned=allow_unpinned)
        root = upstream / "Matrix-Game-3" if (upstream / "Matrix-Game-3").exists() else upstream
        return apply_source_patches(
            model_id=self.spec.id,
            optimization_id=optimization_id,
            upstream=root,
            expected_revision=self.spec.pinned_revision,
            marker_name=".wms_fused_adaln.json",
            allow_unpinned=allow_unpinned,
            patches=(
                SourcePatch(
                    path="wan/modules/model.py",
                    import_anchor="import torch.nn.functional as torch_F\n",
                    import_line="from world_model_systems.optimizations.kernels.adaln import fused_adaln\n",
                    replacements=(
                        (
                            "(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(x.dtype)",
                            "fused_adaln(x, e[1].squeeze(2), e[0].squeeze(2), eps=self.norm1.eps)",
                        ),
                        (
                            "(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(self.ffn[0].weight.dtype)",
                            "fused_adaln(x, e[4].squeeze(2), e[3].squeeze(2), eps=self.norm2.eps).to(self.ffn[0].weight.dtype)",
                        ),
                    ),
                ),
            ),
        )
