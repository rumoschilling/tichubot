import pytest
from types import SimpleNamespace

from tichu_env.cards import DRAGON_INDEX
from training.critic_value_eval import (
    TERMINAL_COHORTS,
    ValueTraceCollector,
    summarize,
)


class _Game:
    def __init__(self, score_a, score_b):
        self.round_scores = (score_a, score_b)
        self.round_over = False
        self.finish_order = [0, 1, 2, 3]
        self.tichu_declarations = [0, 0, 0, 0]
        self.played_cards = set()
        self.trick = SimpleNamespace(cards_on_table=[])


def _entry(game_index, player, legal=None):
    legal = legal or [([0], False)]
    return (game_index, player, None, None, None, len(legal), legal)


def test_trace_collector_assigns_team_targets_and_steps_to_end():
    game = _Game(135, 65)
    collector = ValueTraceCollector({"a": "static", "b": "history"})
    collector.begin_batch([game], None)
    collector.record_decisions([game], "a", [_entry(0, 0)], [0.1])
    collector.record_action(game, "a", 0, ([0], False))
    collector.record_decisions([game], "b", [_entry(0, 1)], [-0.2])
    collector.record_action(game, "b", 1, ([0], False))
    collector.record_decisions([game], "a", [_entry(0, 2)], [0.15])
    game.round_over = True
    collector.record_action(game, "a", 2, ([0], False))
    collector.finish_batch([game], None)

    assert [sample.steps_to_end for sample in collector.samples] == [3, 2, 1]
    assert [sample.target_points for sample in collector.samples] == [70, -70, 70]
    assert [sample.prediction_points for sample in collector.samples] == pytest.approx(
        [40, -80, 60]
    )
    assert [sample.ended_immediately for sample in collector.samples] == [
        False,
        False,
        True,
    ]


def test_summary_reports_signed_bias_and_absolute_error():
    game = _Game(100, 0)
    collector = ValueTraceCollector({"a": "model", "b": "other"})
    collector.begin_batch([game], None)
    collector.record_decisions([game], "a", [_entry(0, 0)], [0.1])
    collector.record_action(game, "a", 0, ([0], False))
    collector.record_decisions([game], "a", [_entry(0, 2)], [0.15])
    collector.record_action(game, "a", 2, ([0], False))
    collector.finish_batch([game], None)

    stats = summarize(collector.samples)

    assert stats["bias"] == pytest.approx(-50.0)
    assert stats["mae"] == pytest.approx(50.0)
    assert stats["rmse"] == pytest.approx(((40**2 + 60**2) / 2) ** 0.5)


def test_observable_terminal_cohort_requires_immediate_end():
    game = _Game(80, 20)
    collector = ValueTraceCollector({"a": "model", "b": "other"})
    collector.begin_batch([game], None)
    collector.record_decisions([game], "a", [_entry(0, 0)], [0.1])
    game.round_over = True
    collector.record_action(game, "a", 0, ([0], False))
    collector.finish_batch([game], None)

    sample = collector.samples[0]
    predicates = dict(TERMINAL_COHORTS)
    assert predicates["immediate end"](sample)
    assert predicates["forced + no prior Dragon give"](sample)


def test_terminal_dragon_is_not_mistaken_for_a_prior_give():
    game = _Game(80, 20)
    collector = ValueTraceCollector({"a": "model", "b": "other"})
    collector.begin_batch([game], None)
    legal = [([DRAGON_INDEX], False)]
    collector.record_decisions([game], "a", [_entry(0, 0, legal)], [0.1])
    game.round_over = True
    collector.record_action(game, "a", 0, legal[0])
    collector.finish_batch([game], None)

    sample = collector.samples[0]
    predicates = dict(TERMINAL_COHORTS)
    assert not sample.prior_dragon_give
    assert predicates["forced + no prior Dragon give"](sample)


def test_selected_tichu_call_is_recorded():
    game = _Game(180, -80)
    collector = ValueTraceCollector({"a": "model", "b": "other"})
    collector.begin_batch([game], None)
    legal = [([0], True)]
    collector.record_decisions([game], "a", [_entry(0, 0, legal)], [0.1])
    game.round_over = True
    collector.record_action(game, "a", 0, legal[0])
    collector.finish_batch([game], None)

    sample = collector.samples[0]
    assert sample.had_declaration


def test_prior_dragon_give_is_split_from_observable_terminal_cohort():
    game = _Game(80, 20)
    game.played_cards.add(DRAGON_INDEX)
    collector = ValueTraceCollector({"a": "model", "b": "other"})
    collector.begin_batch([game], None)
    collector.record_decisions([game], "a", [_entry(0, 0)], [0.1])
    game.round_over = True
    collector.record_action(game, "a", 0, ([0], False))
    collector.finish_batch([game], None)

    sample = collector.samples[0]
    predicates = dict(TERMINAL_COHORTS)
    assert sample.prior_dragon_give
    assert not predicates["forced + no prior Dragon give"](sample)
    assert predicates["forced + prior Dragon give"](sample)


def test_forced_double_win_is_a_terminal_cohort_even_after_dragon_give():
    game = _Game(200, 0)
    game.finish_order = [0, 2, 1, 3]
    game.played_cards.add(DRAGON_INDEX)
    collector = ValueTraceCollector({"a": "model", "b": "other"})
    collector.begin_batch([game], None)
    collector.record_decisions([game], "a", [_entry(0, 0)], [0.1])
    game.round_over = True
    collector.record_action(game, "a", 0, ([0], False))
    collector.finish_batch([game], None)

    sample = collector.samples[0]
    predicates = dict(TERMINAL_COHORTS)
    assert sample.scoring_kind == "double"
    assert sample.prior_dragon_give
    assert predicates["forced double win"](sample)
