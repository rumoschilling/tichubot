import itertools
import random

import pytest

from training.ring import (
    PHASE_PAIRS,
    RingState,
    entropy_coef_at,
    learner_seat_counts,
    matchups,
    parity_reached,
    phase_split,
    sample_seat_owners,
)


def test_schedule_trains_every_pair_and_alternates_within_each_split():
    assert {frozenset(p) for p in PHASE_PAIRS} == {frozenset(p) for p in itertools.combinations(range(4), 2)}
    for k in range(0, 6, 2):
        trainees, frozen = phase_split(k)
        # The frozen pair of one phase trains in the next, against the pair that caught up.
        assert phase_split(k + 1) == (frozen, trainees)
    assert phase_split(6) == phase_split(0)


def test_nine_matchups_cover_all_partnerships():
    ms = matchups((0, 1), (2, 3))
    assert len(ms) == 9
    assert {t for t, _ in ms} == {(0, 1), (0, 0), (1, 1)}
    assert {f for _, f in ms} == {(2, 3), (2, 2), (3, 3)}


def test_seat_owners_are_balanced_and_trainees_never_oppose_each_other():
    trainees, frozen = (0, 2), (1, 3)
    owners = sample_seat_owners(trainees, frozen, 18 * 40, random.Random(0))

    cells = {}
    for o in owners:
        team_a, team_b = {o[0], o[2]}, {o[1], o[3]}
        assert (team_a <= set(trainees) and team_b <= set(frozen)) or (
            team_a <= set(frozen) and team_b <= set(trainees))
        key = (tuple(sorted((o[0], o[2]))), tuple(sorted((o[1], o[3]))))
        cells[key] = cells.get(key, 0) + 1
    assert len(cells) == 18 and set(cells.values()) == {40}
    counts = learner_seat_counts(owners, trainees)
    assert counts[0] == counts[2] == len(owners)  # one seat per game on average


def test_parity_needs_a_clear_pooled_lead_and_no_clear_deficit():
    # Nine matchups with se 6 each: pooled se = 6 * 3 / 9 = 2.
    def matchups(mean, z=0.1, n=9):
        return [{"mean": mean, "se": 6.0, "z": z}] * n

    assert parity_reached(matchups(4.1))            # pooled z 2.05
    assert not parity_reached(matchups(3.9))        # pooled z 1.95: not clearly ahead
    assert not parity_reached(matchups(0.5))        # merely even
    one_clear_loss = matchups(8.0, z=1.3, n=8) + [{"mean": -16.0, "se": 6.0, "z": -2.7}]
    assert not parity_reached(one_clear_loss)
    # min_pooled_z=0 restores the original "even or better" rule.
    assert parity_reached(matchups(0.5), min_pooled_z=0)
    assert not parity_reached(matchups(-0.5, z=-0.1), min_pooled_z=0)


def test_entropy_schedule_reproduces_lineage_values():
    # The 250,880-game checkpoint stored 0.02750144: the coefficient of its
    # last update, computed before that 1,024-game update was counted.
    assert entropy_coef_at(250_880 - 1024, 0.03, 0.01, 2_000_000) == pytest.approx(0.02750144)
    assert entropy_coef_at(5_000_000, 0.03, 0.01, 2_000_000) == pytest.approx(0.01)


def test_state_round_trips(tmp_path):
    state = RingState(["a", "b", "c", "d"], ["1", "2", "3", "4"], [1.0, 2.0, 3.0, 4.0], phase_index=3)
    state.checks.append({"pooled_mean": 1.5})
    path = tmp_path / "ring_state.json"
    state.save(path)
    assert RingState.load(path) == state


class _ImmediatePool:
    def map(self, function, jobs):
        return [function(job) for job in jobs]


def test_pooled_matchup_evaluation_matches_serial():
    import torch

    from tests.test_multi_agent_rollout import ENC_CFG, MDL_CFG, _agents
    from training.ring import evaluate_matchups
    from training.rollout_history import _agent_payload

    payloads = [_agent_payload(a, ENC_CFG, MDL_CFG) for a in _agents()]
    kwargs = dict(deals=4, block_size=2, seed=3, device=torch.device("cpu"))
    serial = evaluate_matchups(payloads, (0, 1), (2, 3), **kwargs)
    pooled = evaluate_matchups(payloads, (0, 1), (2, 3), pool=_ImmediatePool(), n_workers=1, **kwargs)

    assert len(serial) == 9
    assert [r["mean"] for r in serial] == [r["mean"] for r in pooled]
    assert [tuple(r["trainee_team"]) for r in serial][:3] == [(0, 1)] * 3


def test_history_games_pit_trainees_against_snapshot_teams():
    from training.ring import sample_seat_owners_with_history

    trainees, frozen, history = (0, 1), (2, 3), [4, 5]
    owners = sample_seat_owners_with_history(trainees, frozen, history, 1000, 0.2, random.Random(1))

    assert len(owners) == 1000
    vs_history = 0
    for o in owners:
        teams = [{o[0], o[2]}, {o[1], o[3]}]
        trainee_team = [t for t in teams if t <= set(trainees)]
        assert len(trainee_team) == 1  # exactly one trainee partnership per game
        other = teams[1 - teams.index(trainee_team[0])]
        assert other <= set(frozen) or other <= set(history)
        vs_history += other <= set(history)
    assert vs_history == 200


def test_no_snapshots_means_no_history_games():
    from training.ring import sample_seat_owners_with_history

    owners = sample_seat_owners_with_history((0, 1), (2, 3), [], 36, 0.5, random.Random(2))
    assert all(set(o) <= {0, 1, 2, 3} for o in owners)


def test_state_from_before_snapshot_pool_loads(tmp_path):
    import json

    path = tmp_path / "ring_state.json"
    legacy = RingState(["a", "b", "c", "d"], ["1", "2", "3", "4"], [0.0] * 4)
    data = legacy.__dict__.copy()
    del data["snapshots"]
    path.write_text(json.dumps(data))
    assert RingState.load(path).snapshots == []


def test_snapshot_weights_favour_stronger_and_skip_excluded():
    import math

    from training.ring import snapshot_weights

    pool = ["old", "mid", "new", "live"]
    strengths = {"old": -120.0, "mid": -80.0, "new": -40.0, "live": -35.0}
    w = snapshot_weights(pool, strengths, exclude={"live"}, temp=40.0)

    assert w[3] == 0.0
    assert w[2] == 1.0
    assert w[1] == pytest.approx(math.exp(-1.0))
    assert w[0] == pytest.approx(math.exp(-2.0))
    assert snapshot_weights(pool, strengths, exclude=set(), temp=0) == [1.0] * 4
    # A missing strength falls back to uniform rather than guessing.
    assert snapshot_weights(pool, {"old": -1.0}, exclude=set(), temp=40.0) == [1.0] * 4


def test_draw_snapshots_is_distinct_and_follows_weights():
    from training.ring import draw_snapshots

    rng = random.Random(3)
    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(4000):
        (first,) = draw_snapshots(["a", "b", "c"], [1.0, 2.0, 0.0], 1, rng)
        counts[first] += 1
    assert counts["c"] == 0
    assert 0.62 < counts["b"] / 4000 < 0.71
    two = draw_snapshots(["a", "b", "c"], [1.0, 1.0, 1.0], 2, rng)
    assert len(set(two)) == 2


def test_latest_phase_checkpoints_tracks_each_agents_newest():
    from training.ring import latest_phase_checkpoints

    phases = [
        {"trainees": [0, 1], "checkpoints": ["p0a0", "p0a1"]},
        {"trainees": [2, 3], "checkpoints": ["p1a2", "p1a3"]},
        {"trainees": [0, 2], "checkpoints": ["p2a0", "p2a2"]},
    ]
    assert latest_phase_checkpoints(phases) == {"p2a0", "p0a1", "p2a2", "p1a3"}
