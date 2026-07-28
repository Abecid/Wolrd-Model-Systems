"""Minimal integration pattern for an existing FastVideo/DiT training loop."""

from pathlib import Path

import torch

from matrixgame_systems.common.manifest import environment_manifest, write_json_atomic
from matrixgame_systems.distributed.checkpoint import AsyncDistributedCheckpointer
from matrixgame_systems.profiling.events import PhaseRecorder


def train(model, optimizer, loader, max_steps: int, run_dir: str = "runs/train"):
    run_path = Path(run_dir)
    write_json_atomic(run_path / "manifest.json", environment_manifest(config={"max_steps": max_steps}))
    profiler = PhaseRecorder(run_path, frames_per_step=16, num_gpus=1)
    checkpointer = AsyncDistributedCheckpointer(run_path / "checkpoints")
    iterator = iter(loader)
    for step in range(max_steps):
        profiler.mark_step(step)
        with profiler.phase("step"):
            with profiler.phase("dataloader"):
                batch = next(iterator)
            with profiler.phase("forward"):
                loss = model(batch)
            with profiler.phase("backward"):
                loss.backward()
            with profiler.phase("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step and step % 1000 == 0:
            with profiler.phase("checkpoint"):
                pause_ms = checkpointer.save(
                    step,
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    {"foreground_pause_ms": None},
                )
            print(f"checkpoint foreground pause: {pause_ms:.3f} ms")
    checkpointer.close()
    profiler.close()
