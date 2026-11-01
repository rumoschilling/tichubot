"""Team-signed proxy rewards in the bootstrap passing pipeline.

With a purely additive value function V(hand) = Σ_c w_c the team-signed
exchange score has a closed form: intra-team passes cancel exactly and

    R = 2 · [w(cards B passed to A) − w(cards A passed to B)]

The tests verify the pipeline's rewards against that closed form,
reconstructed independently from the harvested choices — which also pins
down the direction semantics (left = p+1, partner = p+2, right = p+3)
and the ±R team sign convention.
"""
import numpy as np
import torch

from training.model_passing import TichuPassingPolicy
from training.rollout_passing import _collect_passing_batched

DEVICE = torch.device("cpu")
N_DEALS = 6


class _LinearValuePolicy(torch.nn.Module):
    """V(obs) = Σ w_c over the 56 own-hand bits; blind to everything else."""

    def __init__(self):
        super().__init__()
        self.w = torch.arange(56, dtype=torch.float32) / 56.0

    def trunk(self, obs):
        return obs

    def value_head(self, feat):
        return (feat[:, :56] * self.w).sum(dim=1, keepdim=True)


def test_team_signed_rewards_match_linear_closed_form():
    torch.manual_seed(0)
    rollout = _collect_passing_batched(
        TichuPassingPolicy(), _LinearValuePolicy(), N_DEALS, DEVICE)

    hand_cards = rollout["hand_cards"].reshape(N_DEALS, 4, 14)
    choices = rollout["choices"].reshape(N_DEALS, 4, 3)
    rewards = rollout["rewards"].reshape(N_DEALS, 4)
    w = np.arange(56, dtype=np.float64) / 56.0

    for i in range(N_DEALS):
        w_a_to_b = 0.0
        w_b_to_a = 0.0
        for p in range(4):
            receivers = ((p + 1) % 4, (p + 2) % 4, (p + 3) % 4)
            for slot, receiver in enumerate(receivers):
                if p % 2 == receiver % 2:
                    continue  # intra-team: cancels under additive V
                card = hand_cards[i, p, choices[i, p, slot]]
                if p % 2 == 0:
                    w_a_to_b += w[card]
                else:
                    w_b_to_a += w[card]
        expected = 2.0 * (w_b_to_a - w_a_to_b)
        assert np.allclose(
            rewards[i], [expected, -expected, expected, -expected],
            atol=1e-4), (
            f"deal {i}: rewards {rewards[i]} != team-signed ±{expected:.4f}")


def test_rewards_are_team_antisymmetric_within_each_deal():
    torch.manual_seed(1)
    rollout = _collect_passing_batched(
        TichuPassingPolicy(), _LinearValuePolicy(), N_DEALS, DEVICE)
    per_deal = rollout["rewards"].reshape(N_DEALS, 4)
    assert np.allclose(per_deal[:, 0], per_deal[:, 2], atol=1e-6)
    assert np.allclose(per_deal[:, 1], per_deal[:, 3], atol=1e-6)
    assert np.allclose(per_deal[:, 0], -per_deal[:, 1], atol=1e-6)
    assert np.any(per_deal != 0.0)
