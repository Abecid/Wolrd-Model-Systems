import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from matrixgame_systems.common.config import load_yaml

from .admission import HBMAdmissionController
from .backend import MockBackend
from .matrixgame_backend import MatrixGameBackend
from .metrics import render_metrics
from .models import GenerationSpec, RequestState
from .scheduler import DynamicBatchScheduler, QueueFullError


def create_app(config: dict[str, Any]):
    try:
        from fastapi import FastAPI, HTTPException, Response
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install world-model-systems[service]") from exc

    class GenerationBody(BaseModel):
        prompt: str = Field(min_length=1, max_length=4096)
        image_path: str
        height: int = Field(default=704, ge=128, le=2160)
        width: int = Field(default=1280, ge=128, le=3840)
        num_iterations: int = Field(default=2, ge=1, le=100)
        num_inference_steps: int = Field(default=3, ge=1, le=100)
        seed: int = 42
        precision: str = "bf16"
        output_name: str | None = None

    backend_name = str(config.get("backend", "matrix-game-3"))
    if backend_name == "mock":
        backend = MockBackend(
            delay_s=float(config.get("mock_delay_s", 0.01)),
            max_batch_size=int(config.get("max_batch_size", 8)),
            output_dir=str(config.get("output_dir", "/tmp")),
        )
    elif backend_name in {"matrixgame", "matrix-game-3"}:
        upstream = str(config.get("upstream_path", ""))
        checkpoint = str(config.get("checkpoint_path", ""))
        if not upstream or upstream.startswith("${"):
            raise ValueError("Set MATRIX_GAME_UPSTREAM or upstream_path")
        if not checkpoint or checkpoint.startswith("${"):
            raise ValueError("Set MATRIX_GAME_CKPT or checkpoint_path")
        precision = str(config.get("precision", "bf16"))
        backend = MatrixGameBackend(
            upstream_path=upstream,
            checkpoint_path=checkpoint,
            device=str(config.get("device", "cuda:0")),
            precision=precision,
            output_dir=str(config.get("output_dir", "outputs")),
            use_int8=precision == "int8",
            fa_version=str(config.get("fa_version", "3")),
            compile_vae=bool(config.get("compile_vae", True)),
            warmup_image=config.get("warmup_image"),
            warmup_prompt=str(config.get("warmup_prompt", "A navigable scene.")),
        )
    else:
        raise ValueError(f"Unknown backend {backend_name}")

    total_override = config.get("total_hbm_bytes_override")
    admission = HBMAdmissionController(
        device=str(config.get("device", "cuda:0")),
        estimated_model_bytes=int(config.get("estimated_model_bytes", 0)),
        estimated_request_bytes=int(config.get("estimated_request_bytes", 0)),
        safety_fraction=float(config.get("hbm_safety_fraction", 0.12)),
        total_bytes_override=int(total_override) if total_override is not None else None,
    )
    scheduler = DynamicBatchScheduler(
        backend,
        admission,
        max_queue_size=int(config.get("max_queue_size", 32)),
        max_batch_size=int(config.get("max_batch_size", 1)),
        max_batch_wait_ms=int(config.get("max_batch_wait_ms", 15)),
        request_timeout_s=float(config.get("request_timeout_s", 900)),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await scheduler.start()
        if config.get("warmup", False):
            await backend.warmup()
        app.state.scheduler = scheduler
        yield
        await scheduler.close()

    app = FastAPI(title="World Model Systems", version="0.2.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        return {
            "status": "ready",
            "backend": backend_name,
            "queue_depth": scheduler._queue.qsize() + len(scheduler._deferred),
            "hbm_capacity_bytes": admission.capacity_bytes(),
            "hbm_reserved_bytes": admission.reserved_bytes,
        }

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(render_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/generations/{request_id}")
    async def generation_status(request_id: str) -> dict[str, Any]:
        state = scheduler.get(request_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown request")
        return state.snapshot()

    @app.delete("/v1/generations/{request_id}")
    async def cancel_generation(request_id: str) -> dict[str, Any]:
        if not await scheduler.cancel(request_id):
            raise HTTPException(status_code=404, detail="unknown or completed request")
        return {"request_id": request_id, "status": "cancellation_requested"}

    @app.post("/v1/generations")
    async def generate(body: GenerationBody) -> StreamingResponse:
        image = Path(body.image_path)
        if backend_name != "mock" and not image.exists():
            raise HTTPException(status_code=400, detail=f"image_path does not exist: {image}")
        spec = GenerationSpec(**body.model_dump())
        state = RequestState(spec=spec)
        try:
            await scheduler.submit(state)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        async def stream():
            yield f"data: {json.dumps({'kind': 'accepted', 'request_id': state.request_id})}\n\n"
            while True:
                event = await state.events.get()
                yield f"data: {json.dumps(event.asdict(), default=str)}\n\n"
                if state.done_event.is_set() and state.events.empty():
                    break

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the World Model Systems inference service")
    parser.add_argument("--config", default="configs/service.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    host = args.host or str(config.get("host", "0.0.0.0"))
    port = args.port or int(config.get("port", 8000))
    app = create_app(config)
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
