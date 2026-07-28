import pytest
import torch

from matrixgame_systems.distributed.checkpoint import AsyncDistributedCheckpointer


@pytest.mark.skipif(
    not hasattr(torch.distributed, "checkpoint"), reason="torch.distributed.checkpoint unavailable"
)
def test_async_distributed_checkpoint_roundtrip(tmp_path):
    checkpointer = AsyncDistributedCheckpointer(tmp_path)
    source = {"tensor": torch.arange(16, dtype=torch.float32)}
    pause = checkpointer.save(3, source, {"test": True})
    path = checkpointer.wait()
    assert pause >= 0
    assert path is not None
    assert (path / "COMMITTED").exists()
    target = {"tensor": torch.empty(16, dtype=torch.float32)}
    metadata = checkpointer.load(path, target)
    torch.testing.assert_close(target["tensor"], source["tensor"])
    assert metadata["step"] == 3
