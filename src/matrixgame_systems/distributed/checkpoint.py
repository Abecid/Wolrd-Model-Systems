from __future__ import annotations

import io
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

try:
    import torch.distributed.checkpoint as dcp
except ImportError:  # pragma: no cover
    dcp = None  # type: ignore[assignment]


def serialize_object(value: Any) -> torch.Tensor:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return torch.tensor(list(buffer.getvalue()), dtype=torch.uint8)


def deserialize_object(payload: torch.Tensor) -> Any:
    buffer = io.BytesIO(bytes(payload.cpu().tolist()))
    return torch.load(buffer, weights_only=False)


@dataclass
class PendingCheckpoint:
    step: int
    temporary: Path
    destination: Path
    future: Any
    foreground_pause_ms: float
    metadata: dict[str, Any]


def capture_rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state["generator"] = generator.get_state()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if generator is not None and "generator" in state:
        generator.set_state(state["generator"])


def _rank(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_rank(group) if dist.is_available() and dist.is_initialized() else 0


def _barrier(group: dist.ProcessGroup | None = None) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=group)


class AsyncDistributedCheckpointer:
    """One-in-flight, atomically committed distributed checkpointer.

    `torch.distributed.checkpoint.async_save` performs staging and writing without
    blocking the training stream for the full I/O duration. A checkpoint is not
    considered resumable until `wait()` writes `COMMITTED` and atomically renames
    the temporary directory.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        process_group: dist.ProcessGroup | None = None,
        max_pending: int = 1,
    ) -> None:
        if max_pending != 1:
            raise ValueError("This implementation intentionally permits exactly one in-flight checkpoint")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.process_group = process_group
        self.pending: PendingCheckpoint | None = None

    def save(self, step: int, state: dict[str, Any], metadata: dict[str, Any] | None = None) -> float:
        if self.pending is not None:
            self.wait()
        temporary = self.root / f"step-{step:08d}.incomplete"
        destination = self.root / f"step-{step:08d}"
        if temporary.exists():
            shutil.rmtree(temporary)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite committed checkpoint {destination}")
        temporary.mkdir(parents=True)
        start = time.perf_counter()
        if dcp is not None and hasattr(dcp, "async_save"):
            future = dcp.async_save(
                state_dict=state,
                checkpoint_id=str(temporary / "shards"),
                process_group=self.process_group,
            )
        elif dcp is not None:
            dcp.save(state_dict=state, checkpoint_id=str(temporary / "shards"))
            future = None
        else:  # pragma: no cover - modern supported torch includes DCP
            torch.save(state, temporary / f"rank-{_rank(self.process_group):05d}.pt")
            future = None
        pause_ms = (time.perf_counter() - start) * 1000.0
        self.pending = PendingCheckpoint(
            step=step,
            temporary=temporary,
            destination=destination,
            future=future,
            foreground_pause_ms=pause_ms,
            metadata=metadata or {},
        )
        return pause_ms

    def wait(self) -> Path | None:
        pending = self.pending
        if pending is None:
            return None
        if pending.future is not None:
            pending.future.result()
        _barrier(self.process_group)
        if _rank(self.process_group) == 0:
            manifest = {
                "step": pending.step,
                "foreground_pause_ms": pending.foreground_pause_ms,
                "metadata": pending.metadata,
                "committed_unix_s": time.time(),
            }
            (pending.temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            (pending.temporary / "COMMITTED").write_text("ok\n", encoding="utf-8")
            os.replace(pending.temporary, pending.destination)
        _barrier(self.process_group)
        self.pending = None
        return pending.destination

    def close(self) -> None:
        self.wait()

    def load(self, checkpoint: str | Path, state: dict[str, Any]) -> dict[str, Any]:
        path = Path(checkpoint)
        if not (path / "COMMITTED").exists():
            raise RuntimeError(f"Checkpoint is not atomically committed: {path}")
        if dcp is None:
            loaded = torch.load(path / f"rank-{_rank(self.process_group):05d}.pt", weights_only=False)
            state.update(loaded)
        else:
            dcp.load(
                state_dict=state,
                checkpoint_id=str(path / "shards"),
                process_group=self.process_group,
            )
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    def __enter__(self) -> AsyncDistributedCheckpointer:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
