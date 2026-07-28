from pathlib import Path

from world_model_systems import get_model, list_models
from world_model_systems.core.spec import Capability, LaunchRequest


def test_builtin_models_are_registered():
    ids = {adapter.spec.id for adapter in list_models()}
    assert ids == {"hy-worldplay-1.5", "lingbot-world-v2", "matrix-game-3"}


def test_lingbot_command_is_reproducible():
    adapter = get_model("lingbot-world-v2")
    command = adapter.build_inference_command(
        LaunchRequest(
            upstream=Path("/src/lingbot"),
            checkpoint=Path("/weights/lingbot"),
            prompt="walk forward",
            image=Path("/data/example/image.jpg"),
            output_dir=Path("/out"),
            frames=361,
            num_gpus=8,
        )
    )
    assert command[:2] == ["torchrun", "--nproc_per_node=8"]
    assert "--ulysses_size" in command
    assert "--local_attn_size" in command
    assert Capability.KV_CACHE in adapter.spec.capabilities
