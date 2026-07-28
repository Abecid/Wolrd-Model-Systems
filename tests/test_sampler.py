from matrixgame_systems.distributed.sampler import StatefulDistributedSampler


def test_no_duplicates_across_eight_ranks():
    dataset = list(range(103))
    shards = [
        set(StatefulDistributedSampler(dataset, num_replicas=8, rank=rank, seed=7).rank_indices())
        for rank in range(8)
    ]
    for left in range(8):
        for right in range(left + 1, 8):
            assert shards[left].isdisjoint(shards[right])
    assert sum(map(len, shards)) == 96


def test_exact_cursor_resume():
    dataset = list(range(40))
    sampler = StatefulDistributedSampler(dataset, num_replicas=4, rank=2, seed=9)
    iterator = iter(sampler)
    prefix = [next(iterator) for _ in range(3)]
    state = sampler.state_dict()
    expected_suffix = list(iterator)
    restored = StatefulDistributedSampler(dataset, num_replicas=4, rank=2, seed=9)
    restored.load_state_dict(state)
    assert list(restored) == expected_suffix
    assert prefix + expected_suffix == StatefulDistributedSampler(
        dataset, num_replicas=4, rank=2, seed=9
    ).rank_indices()
