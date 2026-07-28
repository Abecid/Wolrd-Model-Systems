from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import torch

from matrixgame_systems.distributed.checkpoint import (
    AsyncDistributedCheckpointer,
    capture_rng_state,
    deserialize_object,
    restore_rng_state,
    serialize_object,
)

try:
    from torch.distributed.checkpoint.stateful import Stateful
except ImportError:
    class Stateful:  # type: ignore[no-redef]
        pass


class ToyTrainingState(Stateful):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.step = 0

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": torch.tensor(self.step, dtype=torch.int64),
            "rng": serialize_object(capture_rng_state()),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.step = int(state_dict["step"].item())
        restore_rng_state(deserialize_object(state_dict["rng"]))


def make_components(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run(total_steps: int, interrupt_step: int, root: Path) -> None:
    generator = torch.Generator().manual_seed(1234)
    inputs = torch.randn(total_steps, 8, 16, generator=generator)
    targets = torch.randn(total_steps, 8, 4, generator=generator)

    baseline_model, baseline_optimizer = make_components(7)
    initial_model = copy.deepcopy(baseline_model.state_dict())
    baseline_losses = [
        train_step(baseline_model, baseline_optimizer, inputs[step], targets[step])
        for step in range(total_steps)
    ]

    model, optimizer = make_components(7)
    model.load_state_dict(initial_model)
    state = ToyTrainingState(model, optimizer)
    resumed_losses: list[float] = []
    for step in range(interrupt_step):
        resumed_losses.append(train_step(model, optimizer, inputs[step], targets[step]))
        state.step = step + 1

    checkpointer = AsyncDistributedCheckpointer(root)
    pause_ms = checkpointer.save(state.step, {"training": state}, {"kind": "toy-restart"})
    checkpoint = checkpointer.wait()
    assert checkpoint is not None

    restored_model, restored_optimizer = make_components(999)
    restored = ToyTrainingState(restored_model, restored_optimizer)
    checkpointer.load(checkpoint, {"training": restored})
    for step in range(restored.step, total_steps):
        resumed_losses.append(
            train_step(restored_model, restored_optimizer, inputs[step], targets[step])
        )

    torch.testing.assert_close(torch.tensor(resumed_losses), torch.tensor(baseline_losses), rtol=0, atol=0)
    for baseline, resumed in zip(
        baseline_model.parameters(), restored_model.parameters(), strict=True
    ):
        torch.testing.assert_close(baseline, resumed, rtol=0, atol=0)
    print(
        f"Exact restart passed: {total_steps} steps, interruption at {interrupt_step}, "
        f"foreground checkpoint pause {pause_ms:.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--interrupt-step", type=int, default=4)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    if args.checkpoint_dir:
        run(args.steps, args.interrupt_step, Path(args.checkpoint_dir))
    else:
        with tempfile.TemporaryDirectory() as directory:
            run(args.steps, args.interrupt_step, Path(directory))


if __name__ == "__main__":
    main()
