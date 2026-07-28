from pathlib import Path

from world_model_systems import get_model


MODEL_SOURCE = """import torch.nn.functional as torch_F
from .attention import flash_attention

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    return x

class Block:
    def f(self, x, e):
        a = self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        b = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        c = self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        return a, b, c
"""


def _fake_upstream(tmp_path: Path) -> Path:
    target = tmp_path / "wan" / "modules" / "model_fast.py"
    target.parent.mkdir(parents=True)
    target.write_text(MODEL_SOURCE)
    return tmp_path


def test_fused_adaln_patch_is_idempotent(tmp_path):
    upstream = _fake_upstream(tmp_path)
    adapter = get_model("lingbot-world-v2")
    changed = adapter.apply_optimization("fused-adaln", upstream, allow_unpinned=True)
    assert changed == [upstream / "wan" / "modules" / "model_fast.py"]
    source = changed[0].read_text()
    assert source.count("from world_model_systems.optimizations.kernels.adaln import fused_adaln") == 1
    assert source.count("fused_adaln(") == 3
    assert adapter.apply_optimization("fused-adaln", upstream, allow_unpinned=True) == []
