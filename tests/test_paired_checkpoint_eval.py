from __future__ import annotations

import numpy as np

from training.paired_checkpoint_eval import _holm_adjust, _summarize_comparison


def test_holm_adjustment_preserves_ordered_familywise_bounds():
    adjusted = _holm_adjust([0.01, 0.04, 0.20])

    np.testing.assert_allclose(adjusted, [0.03, 0.08, 0.20])


def test_paired_summary_sign_corrects_the_swapped_orientation():
    candidate_team_a = np.full(8, 10.0)
    # Raw scores favor the team-A opponent, so -10 means the candidate on
    # team B also won by 10 points.
    opponent_team_a = np.full(8, -10.0)

    summary = _summarize_comparison(
        candidate_team_a,
        opponent_team_a,
        block_size=2,
        bootstrap_samples=200,
        sign_flip_samples=200,
        alpha=0.05,
        comparisons=2,
        seed=7,
    )

    assert summary["candidate_mean_score_advantage"] == 10.0
    assert summary["candidate_team_a_mean"] == 10.0
    assert summary["candidate_team_b_mean"] == 10.0
    assert summary["candidate_win_rate"] == 1.0
    assert summary["candidate_loss_rate"] == 0.0
    assert summary["direction"] == "candidate_stronger"
