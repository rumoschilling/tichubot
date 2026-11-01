"""Wall-clock cap: run_play_training stops gracefully when cfg.max_hours elapses."""
import numpy as np
import torch

from training.config import TrainConfig
from training.play_training import PlayTrainingBundle, PlayTrainingHooks, run_play_training


def _make_bundle():
    policy = torch.nn.Linear(4, 2)
    return PlayTrainingBundle(policy=policy, optimizer=torch.optim.SGD(policy.parameters(), lr=0.1))


def _make_hooks(updates_done, checkpoints):
    def start_rollout(_blocking):
        return None

    def finish_rollout(_job):
        return {"obs": np.zeros((8, 4), dtype=np.float32), "returns": np.zeros(8, dtype=np.float32)}

    def ppo_update(_rollout):
        updates_done.append(1)
        return {}

    def save_checkpoint(games_played, path):
        checkpoints.append((games_played, path))

    return PlayTrainingHooks(
        start_rollout=start_rollout,
        finish_rollout=finish_rollout,
        ppo_update=ppo_update,
        save_checkpoint=save_checkpoint,
        metric_fields=(),
    )


def _base_cfg(tmp_path, **overrides):
    cfg = TrainConfig()
    cfg.total_games = 10_000
    cfg.games_per_update = 100
    cfg.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.log_file = str(tmp_path / "train.csv")
    cfg.checkpoint_every = 1_000_000  # never on the periodic schedule
    cfg.pass_every = 1_000_000
    cfg.log_every = 1_000_000
    cfg.train_gt = False
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_max_hours_zero_stops_after_first_update(tmp_path):
    updates, checkpoints = [], []
    cfg = _base_cfg(tmp_path, max_hours=0.0)
    run_play_training(cfg, _make_bundle(), _make_hooks(updates, checkpoints), torch.device("cpu"))
    # the in-flight rollout still gets its update, then a graceful stop
    assert len(updates) == 1
    assert [path.endswith("latest.pt") for _games, path in checkpoints] == [True]
    assert checkpoints[0][0] == 100


def test_max_hours_none_runs_to_games_target(tmp_path):
    updates, checkpoints = [], []
    cfg = _base_cfg(tmp_path, total_games=300, max_hours=None)
    run_play_training(cfg, _make_bundle(), _make_hooks(updates, checkpoints), torch.device("cpu"))
    assert len(updates) == 3
    # clean finish saves final.pt, not the interrupt path
    assert any(path.endswith("final.pt") for _games, path in checkpoints)
