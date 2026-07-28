from __future__ import annotations

import asyncio
import builtins
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .backend import InferenceBackend, ProgressCallback
from .models import GenerationSpec, RequestState


class MatrixGameBackend(InferenceBackend):
    """Long-lived single-request adapter for the official Matrix-Game 3 pipeline.

    The upstream memory/control code currently contains batch-size-one assumptions,
    so `max_batch_size` is intentionally one. Queueing is real; fake tensor batching
    is not advertised.
    """

    max_batch_size = 1

    def __init__(
        self,
        *,
        upstream_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        precision: str = "bf16",
        output_dir: str = "outputs",
        use_int8: bool = False,
        fa_version: str = "3",
        compile_vae: bool = True,
        warmup_image: str | None = None,
        warmup_prompt: str = "A navigable scene.",
    ) -> None:
        self.upstream_path = Path(upstream_path).resolve()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.precision = precision
        self.output_dir = Path(output_dir)
        self.use_int8 = use_int8
        self.fa_version = fa_version
        self.compile_vae = compile_vae
        self.warmup_image = warmup_image
        self.warmup_prompt = warmup_prompt
        self.pipeline = None
        self.args = None
        self._lock = asyncio.Lock()

    def _load_sync(self) -> None:
        if self.pipeline is not None:
            return
        if not (self.upstream_path / "generate.py").exists():
            raise FileNotFoundError(f"Matrix-Game-3 not found at {self.upstream_path}")
        sys.path.insert(0, str(self.upstream_path))
        from pipeline.inference_pipeline import MatrixGame3Pipeline
        from wan.configs import WAN_CONFIGS

        local_rank = int(self.device.split(":")[-1]) if ":" in self.device else 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.args = SimpleNamespace(
            output_dir=str(self.output_dir),
            use_int8=self.use_int8,
            verify_quant=False,
            use_async_vae=False,
            compile_vae=self.compile_vae,
            lightvae_pruning_rate=0.75,
            vae_type="mg_lightvae_v2",
            async_vae_warmup_iters=0,
            num_iterations=2,
            size="704*1280",
            save_name="service",
        )
        config = None
        for key in ("matrix_game3", "matrix_game_3", "matrix-game-3", "matrixgame3"):
            if key in WAN_CONFIGS:
                config = WAN_CONFIGS[key]
                break
        if config is None:
            candidates = [key for key in WAN_CONFIGS if "matrix" in key.lower() and "game" in key.lower()]
            if len(candidates) != 1:
                raise KeyError(f"Could not uniquely select Matrix-Game config from {list(WAN_CONFIGS)}")
            config = WAN_CONFIGS[candidates[0]]
        candidate_kwargs = {
            "config": config,
            "checkpoint_dir": self.checkpoint_path,
            "ckpt_dir": self.checkpoint_path,
            "device_id": local_rank,
            "rank": 0,
            "t5_fsdp": False,
            "dit_fsdp": False,
            "use_sp": False,
            "use_usp": False,
            "t5_cpu": False,
            "init_on_cpu": False,
            "convert_model_dtype": False,
            "args": self.args,
            "fa_version": self.fa_version,
            "use_base_model": False,
        }
        signature = inspect.signature(MatrixGame3Pipeline.__init__)
        kwargs = {key: value for key, value in candidate_kwargs.items() if key in signature.parameters}
        self.pipeline = MatrixGame3Pipeline(**kwargs)
        if self.precision == "fp8":
            from matrixgame_systems.kernels.fp8 import convert_named_linears_to_fp8

            changed = convert_named_linears_to_fp8(self.pipeline.model)
            if not changed:
                raise RuntimeError("FP8 requested but no Q/K/V/O Linear modules matched")

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    async def warmup(self) -> None:
        if not self.warmup_image:
            return
        state = RequestState(
            spec=GenerationSpec(
                prompt=self.warmup_prompt,
                image_path=self.warmup_image,
                num_iterations=1,
                num_inference_steps=1,
                precision=self.precision,
                output_name="warmup",
            )
        )

        async def ignore_progress(*args, **kwargs):
            return None

        await self.generate_batch([state], ignore_progress)

    def _generate_sync(self, state: RequestState) -> str:
        assert self.pipeline is not None and self.args is not None
        spec = state.spec
        output_name = spec.output_name or state.request_id
        request_dir = self.output_dir / state.request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        self.args.output_dir = str(request_dir)
        self.args.save_name = output_name
        self.args.num_iterations = spec.num_iterations
        self.args.size = f"{spec.height}*{spec.width}"
        self.pipeline.output_dir = str(request_dir)
        image = Image.open(spec.image_path).convert("RGB")
        original_exit = builtins.exit

        def intercepted_exit(code: int | None = 0) -> None:
            raise SystemExit(code)

        builtins.exit = intercepted_exit
        try:
            candidate_kwargs = {
                "max_area": spec.height * spec.width,
                "shift": 5.0,
                "num_inference_steps": spec.num_inference_steps,
                "guide_scale": 5.0,
                "seed": spec.seed,
                "use_base_model": False,
                "args": self.args,
            }
            signature = inspect.signature(self.pipeline.generate)
            kwargs = {key: value for key, value in candidate_kwargs.items() if key in signature.parameters}
            self.pipeline.generate(spec.prompt, image, **kwargs)
        except SystemExit as exc:
            if int(exc.code or 0) != 0:
                raise
        finally:
            builtins.exit = original_exit
        output = request_dir / f"{output_name}.mp4"
        if not output.exists():
            candidates = sorted(request_dir.glob("*.mp4"))
            if not candidates:
                raise RuntimeError(f"Matrix-Game completed without an MP4 in {request_dir}")
            output = candidates[-1]
        return str(output)

    async def generate_batch(
        self, requests: list[RequestState], progress: ProgressCallback
    ) -> list[str]:
        if len(requests) != 1:
            raise ValueError("The official Matrix-Game 3 adapter is batch-size one")
        state = requests[0]
        async with self._lock:
            await progress(state, "model_progress", {"stage": "conditioning", "fraction": 0.02})
            output = await asyncio.to_thread(self._generate_sync, state)
            await progress(state, "model_progress", {"stage": "saved", "fraction": 1.0})
            return [output]
