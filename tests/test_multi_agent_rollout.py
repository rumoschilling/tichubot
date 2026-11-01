"""Mixed-agent rollout collection: each seat acts with its owner's heads and
only learner decisions are recorded, one buffer per learner."""

import numpy as np
import torch

from training.model_gt import TichuGrandTichuPolicy
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.model_passing import TichuPassingPolicy
from training.rollout_history import (
    RolloutAgent,
    _collect_batched_multi,
    collect_rollouts_multi,
)

DEVICE = torch.device("cpu")
ENC_CFG = dict(d_model=32, nhead=2, n_layers=1, ffn_dim=64)
MDL_CFG = dict(hidden_dim=32, n_layers=2)
SAME_WEIGHTS_BIT = 195

# Team A = seats 0, 2; team B = seats 1, 3.
OWNERS = [
    (0, 2, 1, 2),
    (0, 2, 0, 2),
    (1, 2, 1, 3),
    (2, 0, 3, 1),
    (2, 1, 2, 1),
    (3, 0, 3, 0),
]


def _agents(n: int = 4) -> list[RolloutAgent]:
    agents = []
    for seed in range(n):
        torch.manual_seed(seed)
        policy = TichuPolicyHistory(TransformerHistoryEncoder(**ENC_CFG), **MDL_CFG)
        agents.append(RolloutAgent(policy, TichuGrandTichuPolicy(), TichuPassingPolicy()))
    return agents


def _check_ownership(buffer: dict, learner: int, owners: list[tuple]) -> None:
    games = buffer["history_game_ids"]
    observers = buffer["history_observers"]
    assert len(games) == len(buffer["obs"]) > 0
    for game, seat, obs in zip(games, observers, buffer["obs"]):
        assert owners[game][seat] == learner
        partner_same = owners[game][(seat + 2) % 4] == learner
        assert obs[SAME_WEIGHTS_BIT] == float(partner_same)


def test_only_learner_seats_are_recorded_with_truthful_partner_bit():
    torch.manual_seed(7)
    buffers, _ = _collect_batched_multi(
        _agents(), len(OWNERS), DEVICE,
        seat_owners=OWNERS, learners=(0, 1), with_calling=True, seed=3,
    )

    assert set(buffers) == {0, 1}
    for learner, buffer in buffers.items():
        _check_ownership(buffer, learner, OWNERS)
        seats = sum(owner == learner for owners in OWNERS for owner in owners)
        # Each owned seat makes exactly one GT decision and one pass.
        assert len(buffer["gt_h8"]) == seats
        assert len(buffer["pass_train_obs"]) == seats


def test_single_agent_path_still_returns_self_play_buffer():
    torch.manual_seed(7)
    buffers, _ = _collect_batched_multi(_agents(1), 3, DEVICE, with_calling=True, seed=3)

    buffer = buffers[0]
    assert set(np.unique(buffer["history_game_ids"])) == {0, 1, 2}
    assert np.all(buffer["obs"][:, SAME_WEIGHTS_BIT] == 1.0)


class ImmediatePool:
    def map(self, function, arguments):
        self.slices = [len(args[1]) for args in arguments]
        return [function(args) for args in arguments]


def test_worker_dispatch_merges_per_learner_with_global_game_ids():
    pool = ImmediatePool()
    torch.manual_seed(11)
    buffers = collect_rollouts_multi(
        _agents(), [(ENC_CFG, MDL_CFG)] * 4, OWNERS, (2, 3), DEVICE,
        pool=pool, n_workers=4, with_calling=True, seed=5,
    )

    assert pool.slices == [2, 2, 1, 1]
    assert set(buffers) == {2, 3}
    for learner, buffer in buffers.items():
        _check_ownership(buffer, learner, OWNERS)


def test_recorded_log_probs_and_values_match_reevaluation():
    """Batched read-back keeps each row's outputs with its own decision.

    Forward batches are sorted by history length within each owner, so a
    misplaced row would pair one decision with another's log-prob/value.
    """
    from training.ppo_history import _rebuild_shared_history_numpy

    agents = _agents()
    torch.manual_seed(7)
    buffers, _ = _collect_batched_multi(
        agents, len(OWNERS), DEVICE,
        seat_owners=OWNERS, learners=(0, 1), with_calling=True, seed=3,
    )
    for learner, buf in buffers.items():
        n = len(buf["obs"])
        lens = buf["legal_lens"].astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        legal = np.zeros((n, int(lens.max()), buf["legal_flat"].shape[1]), dtype=np.float32)
        for row in range(n):
            legal[row, : lens[row]] = buf["legal_flat"][starts[row] : starts[row] + lens[row]]
        hist = _rebuild_shared_history_numpy(buf, np.arange(n)).astype(np.float32)
        with torch.no_grad():
            log_prob, value, _ = agents[learner].policy.evaluate(
                torch.as_tensor(buf["obs"]),
                torch.as_tensor(legal),
                torch.as_tensor(lens),
                torch.as_tensor(buf["action_idxs"]),
                torch.as_tensor(hist),
                torch.as_tensor(buf["history_lens"]),
                torch.as_tensor(buf["pass_source_ids"]),
                torch.as_tensor(buf["pass_target_ids"]),
            )
        np.testing.assert_allclose(log_prob.numpy(), buf["log_probs"], atol=1e-4)
        np.testing.assert_allclose(value.numpy(), buf["values"], atol=1e-4)


def test_batched_gt_log_probs_match_per_hand_evaluation():
    """GT heads run batched per owner; each stored log-prob must still be the
    behaviour probability of that row's own hand under its owner's head."""
    from training.model_gt import GT_DERIVE_TEMP
    from training.rollout_history import GT_EXPLORE_EPS

    agents = _agents()
    # A fresh head is a constant "never call"; spread it so hands differ.
    torch.manual_seed(5)
    for agent in agents:
        torch.nn.init.normal_(agent.gt_policy.adv_head.weight, std=0.5 * GT_DERIVE_TEMP)
        torch.nn.init.zeros_(agent.gt_policy.adv_head.bias)
    torch.manual_seed(7)
    buffers, _ = _collect_batched_multi(
        agents, len(OWNERS), DEVICE,
        seat_owners=OWNERS, learners=(0, 1), with_calling=True, seed=3,
    )
    for learner, buf in buffers.items():
        gt = agents[learner].gt_policy
        assert np.ptp(buf["gt_log_probs"]) > 0.1  # probabilities really vary
        expected = []
        for h8, declare in zip(buf["gt_h8"], buf["gt_declare"]):
            with torch.no_grad():
                p = float(torch.sigmoid(gt.call_logits(torch.as_tensor(h8[None, :]))).item())
            p_act = p if declare else 1.0 - p
            expected.append(np.log((1.0 - GT_EXPLORE_EPS) * p_act + GT_EXPLORE_EPS * 0.5))
        np.testing.assert_allclose(buf["gt_log_probs"], expected, atol=1e-5)
