from matrixgame_systems.distributed.validation import (
    deterministic_seed,
    deterministic_validation_subset,
)


def test_deterministic_validation():
    ids = list(range(100))
    assert deterministic_validation_subset(ids, 5, seed=4) == deterministic_validation_subset(
        ids, 5, seed=4
    )
    assert deterministic_seed(1, "clip-4", 9) == deterministic_seed(1, "clip-4", 9)
