"""Behavioral coverage for the fixed-head rollout collector."""

from types import SimpleNamespace

import numpy as np
import torch

from tichu_env.cards import MAHJONG_INDEX
from tichu_env.encoding import STATIC_OBS_SIZE
from tichu_env.game_state import GameState
from training import rollout
from training.model import TichuPolicy


class FirstLegalPolicy:
    def eval(self):
        return self

    def act(self, obs, mask):
        actions = mask.to(torch.int64).argmax(dim=1)
        zeros = torch.zeros(len(obs), device=obs.device)
        return actions, zeros, zeros


class ImmediatePool:
    def map_async(self, function, arguments):
        self.sizes = [args[1] for args in arguments]
        buffers = [function(args) for args in arguments]
        return SimpleNamespace(ready=lambda: True, get=lambda: buffers)


def test_fixed_rollout_records_mahjong_wish_in_next_observation(monkeypatch):
    instances = []

    class OpeningMahjongGame(GameState):
        def reset(self):
            super().reset(seed=0)
            self.hands = [{MAHJONG_INDEX, 0}, {1, 3}, {2, 4}, {5, 6}]
            self.current_player = 0
            self._invalidate_cache()
            instances.append(self)

    rng = SimpleNamespace(
        random=lambda: 0.0,
        randint=lambda low, high: 5,
        choice=lambda options: options[0],
    )
    monkeypatch.setattr(rollout, "GameState", OpeningMahjongGame)
    monkeypatch.setattr(rollout, "random", SimpleNamespace(Random=lambda: rng))

    buffer = rollout._collect_batched(FirstLegalPolicy(), 1, torch.device("cpu"))

    assert instances[0].round_over
    assert buffer["obs"][0, 185] == 0
    assert buffer["obs"][1, 185] == 1
    assert buffer["obs"][1, 186] == np.float32((5 - 2) / 12)


def test_fixed_rollout_with_more_workers_than_games(monkeypatch):
    monkeypatch.setattr(rollout.os, "cpu_count", lambda: 2)
    pool = ImmediatePool()
    policy = TichuPolicy(hidden_dim=16, n_layers=1)

    buffer = rollout.collect_rollouts(
        policy, 1, torch.device("cpu"), pool=pool, n_workers=4,
    )

    assert pool.sizes == [1]
    assert buffer["obs"].ndim == 2
    assert buffer["obs"].shape[1] == STATIC_OBS_SIZE
    assert len(buffer["obs"]) > 0
    assert all(len(values) == len(buffer["obs"]) for values in buffer.values())
