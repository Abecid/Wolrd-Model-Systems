from __future__ import annotations

from world_model_systems.core.spec import Capability, LaunchRequest, LicensePolicy, ModelSpec
from world_model_systems.optimizations.catalog import STANDARD_OPTIMIZATIONS

from .base import BaseAdapter


class HYWorldPlayAdapter(BaseAdapter):
    spec = ModelSpec(
        id="hy-worldplay-1.5",
        display_name="HY-World 1.5 / WorldPlay",
        family="HunyuanVideo causal interactive world model",
        organization="Tencent Hunyuan",
        repository="https://github.com/Tencent-Hunyuan/HY-WorldPlay.git",
        pinned_revision="1588e1336e842b03b0a7860c654ebd7c46bb065e",
        model_size="5B / 8B",
        license=LicensePolicy(
            "Tencent-HY-WorldPlay-Community",
            commercial_use=True,
            redistribution=True,
            hosted_service=True,
            notes=(
                "Custom territorial and use restrictions apply; the published license excludes "
                "the EU, UK, and South Korea and has additional hosted-service/commercial terms."
            ),
            acceptance_required=True,
        ),
        capabilities=frozenset(
            {
                Capability.CAUSAL,
                Capability.STREAMING,
                Capability.ACTION_CONDITIONED,
                Capability.CAMERA_CONDITIONED,
                Capability.FEW_STEP,
                Capability.TRAINING,
                Capability.SEQUENCE_PARALLEL,
                Capability.FSDP,
                Capability.QUANTIZATION,
            }
        ),
        default_checkpoint="tencent/HY-WorldPlay",
        default_resolution=(480, 832),
        default_frames=125,
        optimizations=tuple(
            STANDARD_OPTIMIZATIONS[key]
            for key in ("few-step", "sequence-parallel", "quantization")
        ),
        notes="Adapter is command/profiling only; no Tencent source or weights are redistributed.",
    )

    def build_inference_command(self, request: LaunchRequest) -> list[str]:
        # Upstream centralizes paths and model selection in run.sh. Environment
        # overrides let users keep the source tree untouched.
        return [
            "env",
            f"MODEL_PATH={request.checkpoint}",
            f"IMAGE_PATH={request.image}",
            f"PROMPT={request.prompt}",
            f"NUM_FRAMES={request.frames or self.spec.default_frames}",
            f"N_INFERENCE_GPU={request.num_gpus}",
            "bash",
            str(request.upstream / "run.sh"),
            *request.extra_args,
        ]
