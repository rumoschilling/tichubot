"""Action-vector cache correctness and accounting."""

import numpy as np

from training.rollout_embed import (
    _action_vec,
    get_action_vec_cache_stats,
    reset_action_vec_cache_stats,
    set_action_vec_cache_enabled,
)


def test_action_vec_cache_uses_lru_stats_without_hot_path_profiling():
    action = ([9, 3, 7], True)
    set_action_vec_cache_enabled(True)
    reset_action_vec_cache_stats()
    try:
        first = _action_vec(action)
        second = _action_vec(action)

        np.testing.assert_array_equal(first, second)
        assert not first.flags.writeable
        assert get_action_vec_cache_stats() == {
            "calls": 2,
            "hits": 1,
            "misses": 1,
            "unique_actions": 1,
            "hit_rate": 0.5,
        }

        set_action_vec_cache_enabled(False)
        uncached = _action_vec(action)
        assert uncached.flags.writeable
        assert get_action_vec_cache_stats()["calls"] == 0
    finally:
        set_action_vec_cache_enabled(True)


def test_phoenix_semantic_bit_participates_in_cache_key():
    low = _action_vec(([1, 2, 3, 4, 54], False, False))
    high = _action_vec(([1, 2, 3, 4, 54], False, True))

    np.testing.assert_array_equal(low[:-1], high[:-1])
    assert (low[-1], high[-1]) == (0, 1)
