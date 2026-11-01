"""Conformance checks for consumers of the public PettingZoo environment."""

import pytest
from pettingzoo.test import api_test

from tichu_env.env import TichuEnv


@pytest.mark.parametrize("track_history", [False, True])
def test_last_can_skip_observation_encoding(monkeypatch, track_history):
    env = TichuEnv(seed=0, track_history=track_history)
    env.reset(seed=0)

    def unexpected_observation(_agent):
        pytest.fail("last(observe=False) must not encode an observation")

    monkeypatch.setattr(env, "observe", unexpected_observation)
    assert env.last(observe=False) == (None, 0.0, False, False, {})
    assert env.last(False) == (None, 0.0, False, False, {})


@pytest.mark.filterwarnings("ignore:Observation space for each agent probably should be:UserWarning")
@pytest.mark.filterwarnings("ignore:Observation is not a NumPy array:UserWarning")
def test_static_environment_passes_pettingzoo_api_contract():
    api_test(TichuEnv(seed=0), num_cycles=300, verbose_progress=False)


def test_terminal_rewards_remain_visible_until_next_step():
    env = TichuEnv(seed=0)
    env.reset(seed=0)
    env.game.hands = [{0}, {1}, set(), {2}]
    env.game.finish_order = [2]
    env.game.finished = {2}
    env.game.current_player = 0
    env.game._invalidate_cache()
    env._sync_agent_to_game()
    env._update_legal_actions()

    env.step(env._legal_actions.index(([0], False)))

    expected = {"player_0": 0.5, "player_1": -0.5,
                "player_2": 0.5, "player_3": -0.5}
    assert env.rewards == expected
    assert env._cumulative_rewards == expected
    assert env.last(observe=False)[1:4] == (0.5, True, False)

    env.step(None)

    assert "player_0" not in env.agents
    assert all(reward == 0 for reward in env.rewards.values())
    assert env.last(observe=False)[1] == expected[env.agent_selection]
