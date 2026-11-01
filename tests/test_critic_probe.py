import numpy as np
import torch

from training.critic_probe import (
    RawObservationCritic,
    build_lambda_targets,
    cohort_mask,
    critic_regression_loss,
    evaluate_critics,
    forced_double_score_oracle,
    model_fingerprint,
    train_critic_batch,
)
from training.model_embed import TichuPolicyEmbed
from training.rollout_embed import collect_rollouts_embed


def test_raw_critic_variants_have_expected_shape_and_residual():
    obs = torch.randn(7, 200)
    for activation, residual in (
        ("linear", False),
        ("tanh", False),
        ("silu", False),
        ("silu", True),
    ):
        critic = RawObservationCritic(
            200,
            hidden_dim=16,
            n_layers=2,
            activation=activation,
            raw_residual=residual,
        )
        assert critic(obs).shape == (7,)


def test_raw_critic_width_changes_capacity_without_changing_output_shape():
    obs = torch.randn(7, 200)
    narrow = RawObservationCritic(200, hidden_dim=256, activation="tanh")
    wide = RawObservationCritic(200, hidden_dim=512, activation="tanh")

    assert narrow(obs).shape == wide(obs).shape == (7,)
    assert sum(p.numel() for p in wide.parameters()) > 3 * sum(
        p.numel() for p in narrow.parameters()
    )


def test_heteroscedastic_critic_predicts_positive_variance_and_trains_both_heads():
    critic = RawObservationCritic(
        200,
        hidden_dim=16,
        n_layers=2,
        activation="tanh",
        heteroscedastic=True,
    )
    obs = torch.randn(32, 200)
    targets = torch.linspace(-0.5, 0.5, 32)
    value, log_variance = critic.distribution(obs)

    assert value.shape == log_variance.shape == (32,)
    assert torch.all(torch.isfinite(log_variance))
    assert torch.all(torch.exp(log_variance) > 0.0)

    loss = critic_regression_loss(critic, obs, targets)
    loss.backward()
    assert critic.value_head.weight.grad is not None
    assert critic.log_variance_head.weight.grad is not None
    assert torch.count_nonzero(critic.value_head.weight.grad).item() > 0
    assert torch.count_nonzero(critic.log_variance_head.weight.grad).item() > 0


def test_heteroscedastic_anchor_floors_mean_gradient_without_changing_variance_gradient():
    pure = RawObservationCritic(
        3,
        activation="linear",
        heteroscedastic=True,
    )
    anchored = RawObservationCritic(
        3,
        activation="linear",
        heteroscedastic=True,
        mse_anchor_alpha=0.2,
    )
    anchored.load_state_dict(pure.state_dict())
    with torch.no_grad():
        pure.log_variance_head.bias.fill_(100.0)
        anchored.log_variance_head.bias.fill_(100.0)

    obs = torch.ones(8, 3)
    targets = torch.ones(8)
    pure_loss = critic_regression_loss(pure, obs, targets)
    anchored_loss = critic_regression_loss(anchored, obs, targets)
    pure_loss.backward()
    anchored_loss.backward()

    # The surrogate reports the same Gaussian NLL and leaves uncertainty
    # learning unchanged, while retaining at least 20% mean weight.
    torch.testing.assert_close(anchored_loss, pure_loss)
    torch.testing.assert_close(
        anchored.log_variance_head.bias.grad,
        pure.log_variance_head.bias.grad,
    )
    assert abs(anchored.value_head.bias.grad.item()) > 10 * abs(
        pure.value_head.bias.grad.item()
    )


def test_heteroscedastic_evaluation_reports_calibration_metrics():
    critic = RawObservationCritic(
        200,
        hidden_dim=8,
        n_layers=1,
        activation="tanh",
        heteroscedastic=True,
    )
    data = {
        "obs": np.zeros((3, 200), dtype=np.float32),
        "returns": np.array([0.0, 0.25, -0.25], dtype=np.float32),
        "values": np.zeros(3, dtype=np.float32),
        "steps_to_end": np.array([1, 5, 10], dtype=np.int16),
        "steps_from_start": np.array([65, 2, 3], dtype=np.int16),
        "legal_lens": np.array([1, 2, 2], dtype=np.int32),
        "double_win": np.array([False, False, False]),
    }

    report = evaluate_critics(
        {"tanh_hetero": critic}, data, torch.device("cpu")
    )
    metrics = report["all"]["tanh_hetero"]

    assert metrics["sigma"] > 0.0
    assert 0.0 <= metrics["coverage_1sigma"] <= 1.0
    assert np.isfinite(metrics["nll"])


def test_critic_update_does_not_modify_actor():
    actor = TichuPolicyEmbed(obs_size=200, hidden_dim=16, embed_dim=8)
    before = model_fingerprint(actor)
    critic = RawObservationCritic(200, hidden_dim=16, activation="silu")
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    critic_before = model_fingerprint(critic)

    rng = np.random.default_rng(3)
    train_critic_batch(
        {"silu": critic},
        {"silu": optimizer},
        rng.normal(size=(32, 200)).astype(np.float32),
        rng.normal(size=32).astype(np.float32),
        device=torch.device("cpu"),
        batch_size=16,
        n_epochs=2,
        max_grad_norm=1.0,
    )

    assert model_fingerprint(actor) == before
    assert model_fingerprint(critic) != critic_before


def test_embed_rollout_labels_one_terminal_transition_per_round():
    games = 4
    policy = TichuPolicyEmbed(obs_size=200, hidden_dim=16, embed_dim=8)
    rollout = collect_rollouts_embed(
        policy,
        games,
        torch.device("cpu"),
        counted_trick_points=True,
    )

    assert rollout["steps_to_end"].dtype == np.int16
    assert rollout["steps_from_start"].dtype == np.int16
    assert rollout["double_win"].dtype == np.bool_
    assert rollout["players"].dtype == np.int8
    assert len(rollout["steps_to_end"]) == len(rollout["obs"])
    assert int((rollout["steps_to_end"] == 1).sum()) == games
    starts = np.flatnonzero(rollout["steps_from_start"] == 1)
    assert len(starts) == games
    for start, end in zip(starts, [*starts[1:], len(rollout["obs"])]):
        np.testing.assert_array_equal(
            rollout["steps_from_start"][start:end],
            np.arange(1, end - start + 1, dtype=np.int16),
        )


def test_lambda_targets_handle_episode_boundaries_and_team_perspective():
    critic = RawObservationCritic(200, activation="linear")
    with torch.no_grad():
        critic.value_head.weight.zero_()
        critic.value_head.weight[0, 0] = 1.0
        critic.value_head.bias.zero_()

    obs = np.zeros((5, 200), dtype=np.float32)
    obs[:, 0] = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    data = {
        "obs": obs,
        "returns": np.array([0.5, -0.5, 0.5, -0.25, -0.25], dtype=np.float32),
        "steps_to_end": np.array([3, 2, 1, 2, 1], dtype=np.int16),
        "players": np.array([0, 1, 2, 3, 1], dtype=np.int8),
    }

    td0 = build_lambda_targets(
        {"critic": critic}, data, 0.0, device=torch.device("cpu")
    )["critic"]
    monte_carlo = build_lambda_targets(
        {"critic": critic}, data, 1.0, device=torch.device("cpu")
    )["critic"]

    np.testing.assert_allclose(td0, [-0.2, -0.3, 0.5, 0.5, -0.25])
    np.testing.assert_allclose(monte_carlo, data["returns"])


def test_forced_terminal_cohort_selects_only_unique_final_actions():
    data = {
        "returns": np.zeros(5, dtype=np.float32),
        "steps_to_end": np.array([1, 1, 2, 1, 10], dtype=np.int16),
        "steps_from_start": np.array([65, 12, 64, 65, 4], dtype=np.int16),
        "legal_lens": np.array([1, 2, 1, 1, 1], dtype=np.int32),
        "double_win": np.array([False, False, False, True, False]),
    }

    assert cohort_mask(data, "terminal").tolist() == [True, True, False, True, False]
    assert cohort_mask(data, "step_10").tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert cohort_mask(data, "start_65").tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]
    assert cohort_mask(data, "forced_terminal").tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]
    assert cohort_mask(data, "forced_double").tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]


def test_forced_double_score_oracle_uses_relative_declarations():
    obs = np.zeros((4, 200), dtype=np.float32)
    obs[1, 168] = 1.0  # Self Tichu fails: -100.
    obs[2, 170] = 1.0  # Partner Tichu succeeds: +100.
    obs[3, 173] = 1.0  # Next opponent Grand Tichu fails: opponent -200.

    assert (forced_double_score_oracle(obs) * 400.0).tolist() == [
        200.0,
        100.0,
        300.0,
        400.0,
    ]
