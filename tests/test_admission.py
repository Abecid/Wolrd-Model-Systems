import pytest

from matrixgame_systems.serving.admission import HBMAdmissionController


@pytest.mark.asyncio
async def test_hbm_reservations():
    admission = HBMAdmissionController(
        total_bytes_override=1000,
        estimated_model_bytes=200,
        estimated_request_bytes=250,
        safety_fraction=0.1,
    )
    first = await admission.reserve(2)
    assert first is not None
    assert admission.reserved_bytes == 500
    assert await admission.reserve(1) is None
    await admission.release(first)
    assert admission.reserved_bytes == 0
