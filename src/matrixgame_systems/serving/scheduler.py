from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import Any

from . import metrics
from .admission import HBMAdmissionController
from .backend import InferenceBackend
from .models import RequestState, RequestStatus


class QueueFullError(RuntimeError):
    pass


class DynamicBatchScheduler:
    def __init__(
        self,
        backend: InferenceBackend,
        admission: HBMAdmissionController,
        *,
        max_queue_size: int = 32,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 15,
        request_timeout_s: float = 900.0,
    ) -> None:
        self.backend = backend
        self.admission = admission
        self.max_queue_size = max_queue_size
        self.max_batch_size = min(max_batch_size, backend.max_batch_size)
        self.max_batch_wait_s = max_batch_wait_ms / 1000.0
        self.request_timeout_s = request_timeout_s
        self._queue: asyncio.Queue[RequestState] = asyncio.Queue(maxsize=max_queue_size)
        self._deferred: deque[RequestState] = deque()
        self._requests: dict[str, RequestState] = {}
        self._worker: asyncio.Task | None = None
        self._closing = False

    async def start(self) -> None:
        if self._worker is None:
            await self.backend.load()
            self._worker = asyncio.create_task(self._loop(), name="mgs-dynamic-batcher")

    async def close(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def submit(self, state: RequestState) -> RequestState:
        if self._queue.full():
            raise QueueFullError(f"Queue capacity {self.max_queue_size} reached")
        self._requests[state.request_id] = state
        await state.emit("queued", queue_depth=self._queue.qsize() + 1)
        await self._queue.put(state)
        metrics.QUEUE_DEPTH.set(self._queue.qsize() + len(self._deferred))
        return state

    def get(self, request_id: str) -> RequestState | None:
        return self._requests.get(request_id)

    async def cancel(self, request_id: str) -> bool:
        state = self._requests.get(request_id)
        if state is None or state.done_event.is_set():
            return False
        state.cancel_event.set()
        await state.emit("cancellation_requested")
        return True

    async def _next(self) -> RequestState:
        if self._deferred:
            return self._deferred.popleft()
        return await self._queue.get()

    async def _form_batch(self) -> list[RequestState]:
        first = await self._next()
        batch = [first]
        deadline = asyncio.get_running_loop().time() + self.max_batch_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                candidate = await asyncio.wait_for(self._next(), timeout=remaining)
            except TimeoutError:
                break
            if candidate.spec.compatibility_key() == first.spec.compatibility_key():
                batch.append(candidate)
            else:
                self._deferred.append(candidate)
        metrics.QUEUE_DEPTH.set(self._queue.qsize() + len(self._deferred))
        return batch

    async def _progress(self, state: RequestState, kind: str, payload: dict[str, Any]) -> None:
        await state.emit(kind, **payload)

    async def _run_batch(self, batch: list[RequestState]) -> None:
        active = [state for state in batch if not state.cancel_event.is_set()]
        cancelled = [state for state in batch if state.cancel_event.is_set()]
        for state in cancelled:
            await self._finish_cancelled(state)
        if not active:
            return
        reservation = None
        while reservation is None:
            reservation = await self.admission.reserve(len(active))
            if reservation is None:
                if any(state.cancel_event.is_set() for state in active):
                    active = [state for state in active if not state.cancel_event.is_set()]
                    if not active:
                        return
                await asyncio.sleep(0.01)
        metrics.HBM_RESERVED.set(self.admission.reserved_bytes)
        now = time.time()
        for state in active:
            state.status = RequestStatus.RUNNING
            state.started_s = now
            metrics.QUEUE_WAIT.observe(now - state.created_s)
            await state.emit("started", batch_size=len(active))
        metrics.ACTIVE.inc(len(active))
        metrics.BATCH_SIZE.observe(len(active))
        try:
            outputs = await asyncio.wait_for(
                self.backend.generate_batch(active, self._progress), timeout=self.request_timeout_s
            )
            if len(outputs) != len(active):
                raise RuntimeError("Backend returned the wrong number of outputs")
            for state, output in zip(active, outputs):
                if state.cancel_event.is_set():
                    await self._finish_cancelled(state)
                    continue
                state.output_path = output
                state.status = RequestStatus.SUCCEEDED
                state.completed_s = time.time()
                await state.emit("completed", output_path=output)
                state.done_event.set()
                metrics.REQUESTS.labels("succeeded").inc()
                metrics.LATENCY.observe(state.completed_s - state.created_s)
        except asyncio.TimeoutError:
            for state in active:
                await self._finish_failed(state, f"request exceeded {self.request_timeout_s}s")
        except Exception as exc:
            for state in active:
                await self._finish_failed(state, repr(exc))
        finally:
            metrics.ACTIVE.dec(len(active))
            await self.admission.release(reservation)
            metrics.HBM_RESERVED.set(self.admission.reserved_bytes)

    async def _finish_cancelled(self, state: RequestState) -> None:
        if state.done_event.is_set():
            return
        state.status = RequestStatus.CANCELLED
        state.completed_s = time.time()
        await state.emit("cancelled")
        state.done_event.set()
        metrics.REQUESTS.labels("cancelled").inc()

    async def _finish_failed(self, state: RequestState, error: str) -> None:
        if state.done_event.is_set():
            return
        state.status = RequestStatus.FAILED
        state.error = error
        state.completed_s = time.time()
        await state.emit("failed", error=error)
        state.done_event.set()
        metrics.REQUESTS.labels("failed").inc()
        metrics.LATENCY.observe(state.completed_s - state.created_s)

    async def _loop(self) -> None:
        while not self._closing:
            batch = await self._form_batch()
            await self._run_batch(batch)
