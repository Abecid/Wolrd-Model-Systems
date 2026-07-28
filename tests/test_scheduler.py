import pytest

from matrixgame_systems.serving.admission import HBMAdmissionController
from matrixgame_systems.serving.backend import MockBackend
from matrixgame_systems.serving.models import GenerationSpec, RequestState, RequestStatus
from matrixgame_systems.serving.scheduler import DynamicBatchScheduler


@pytest.mark.asyncio
async def test_compatible_requests_are_batched(tmp_path):
    backend = MockBackend(delay_s=0.001, max_batch_size=4, output_dir=str(tmp_path))
    admission = HBMAdmissionController(
        total_bytes_override=10_000,
        estimated_request_bytes=100,
        safety_fraction=0,
    )
    scheduler = DynamicBatchScheduler(
        backend,
        admission,
        max_batch_size=4,
        max_batch_wait_ms=20,
        request_timeout_s=1,
    )
    await scheduler.start()
    first = RequestState(GenerationSpec(prompt="a", image_path="x"))
    second = RequestState(GenerationSpec(prompt="b", image_path="y"))
    await scheduler.submit(first)
    await scheduler.submit(second)
    await first.done_event.wait()
    await second.done_event.wait()
    assert first.status == RequestStatus.SUCCEEDED
    assert second.status == RequestStatus.SUCCEEDED
    assert first.output_path and second.output_path
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancellation_before_dispatch(tmp_path):
    backend = MockBackend(delay_s=0.01, max_batch_size=1, output_dir=str(tmp_path))
    admission = HBMAdmissionController(
        total_bytes_override=10_000, estimated_request_bytes=100, safety_fraction=0
    )
    scheduler = DynamicBatchScheduler(backend, admission, max_batch_size=1)
    state = RequestState(GenerationSpec(prompt="a", image_path="x"))
    await scheduler.submit(state)
    assert await scheduler.cancel(state.request_id)
    await scheduler.start()
    await state.done_event.wait()
    assert state.status == RequestStatus.CANCELLED
    await scheduler.close()
