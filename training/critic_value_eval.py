"""Compare critic calibration by main-play decisions remaining.

The two policies play against each other. At every main-play decision, the
acting policy's pre-action value is recorded. After the round, that value is
compared with the realized team-relative score difference. By default the
match is repeated with teams swapped.

``steps_to_end == 1`` is the state immediately before the round-ending main
play. Errors are reported in game points (network values are multiplied by
400).

Usage:
    python -m training.critic_value_eval \
      checkpoints/embed_counted_points_local_20260806/latest.pt play \
      --label-a static --label-b history --games 2048 --gt --self-play
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

import numpy as np
import torch

from tichu_env.cards import DRAGON_INDEX
from tichu_env.game_state import TEAM_A
from training.eval import evaluate


@dataclass(frozen=True)
class ValueSample:
    model: str
    steps_to_end: int
    prediction_points: float
    target_points: float
    legal_action_count: int
    ended_immediately: bool
    had_declaration: bool
    prior_dragon_give: bool
    scoring_kind: str


@dataclass
class _PendingValueSample:
    model: str
    player: int
    step_index: int
    prediction_points: float
    legal_action_count: int
    had_declaration: bool
    prior_dragon_give: bool
    ended_immediately: bool = False


class ValueTraceCollector:
    def __init__(self, team_labels: dict[str, str]):
        self.team_labels = team_labels
        self.samples: list[ValueSample] = []
        self._steps: dict[int, int] = {}
        self._records: dict[int, list[_PendingValueSample]] = {}
        self._awaiting_action: dict[int, _PendingValueSample] = {}

    def begin_batch(self, games, _gt_wanted) -> None:
        for game in games:
            key = id(game)
            self._steps[key] = 0
            self._records[key] = []

    def record_decisions(self, games, team, active, values) -> None:
        model = self.team_labels[team]
        for entry, value in zip(active, values, strict=True):
            game_index, player = entry[0], entry[1]
            game = games[game_index]
            key = id(game)
            step_index = self._steps[key]
            self._steps[key] = step_index + 1
            legal = entry[6]
            # An awarded Dragon's recipient is absent from play tokens. A
            # Dragon still on the table has not had a hidden give decision.
            prior_dragon_give = (
                DRAGON_INDEX in game.played_cards
                and DRAGON_INDEX not in game.trick.cards_on_table
            )
            record = _PendingValueSample(
                model=model,
                player=player,
                step_index=step_index,
                prediction_points=float(value) * 400.0,
                legal_action_count=len(legal),
                had_declaration=any(game.tichu_declarations),
                prior_dragon_give=prior_dragon_give,
            )
            self._records[key].append(record)
            self._awaiting_action[key] = record

    def record_action(self, game, _team, _player, action) -> None:
        record = self._awaiting_action.pop(id(game))
        _cards, call_tichu = action[:2]
        record.had_declaration = record.had_declaration or bool(call_tichu)
        record.ended_immediately = bool(game.round_over)

    def finish_batch(self, games, _gt_wanted) -> None:
        for game in games:
            key = id(game)
            total_steps = self._steps.pop(key)
            score_a, score_b = game.round_scores
            first, second = game.finish_order[:2]
            scoring_kind = "double" if (first - second) % 2 == 0 else "normal"
            for record in self._records.pop(key):
                target = score_a - score_b
                if record.player not in TEAM_A:
                    target = -target
                self.samples.append(
                    ValueSample(
                        model=record.model,
                        steps_to_end=total_steps - record.step_index,
                        prediction_points=record.prediction_points,
                        target_points=float(target),
                        legal_action_count=record.legal_action_count,
                        ended_immediately=record.ended_immediately,
                        had_declaration=record.had_declaration,
                        prior_dragon_give=record.prior_dragon_give,
                        scoring_kind=scoring_kind,
                    )
                )


DISTANCE_BINS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-5", 4, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
    ("41-80", 41, 80),
    ("81+", 81, math.inf),
)


def summarize(samples: list[ValueSample]) -> dict[str, float]:
    predictions = np.array(
        [sample.prediction_points for sample in samples], dtype=np.float64
    )
    targets = np.array(
        [sample.target_points for sample in samples], dtype=np.float64
    )
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    correlation = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if len(samples) > 1
        and predictions.std() > 0
        and targets.std() > 0
        else float("nan")
    )
    return {
        "n": len(samples),
        "bias": float(errors.mean()),
        "mae": float(absolute_errors.mean()),
        "mae_se": float(absolute_errors.std(ddof=1) / math.sqrt(len(samples)))
        if len(samples) > 1
        else 0.0,
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "correlation": correlation,
    }


TERMINAL_COHORTS = (
    ("all t-1", lambda sample: True),
    ("immediate end", lambda sample: sample.ended_immediately),
    (
        "immediate + one legal",
        lambda sample: sample.ended_immediately and sample.legal_action_count == 1,
    ),
    (
        "forced + no prior Dragon give",
        lambda sample: sample.ended_immediately
        and sample.legal_action_count == 1
        and not sample.prior_dragon_give,
    ),
    (
        "forced + prior Dragon give",
        lambda sample: sample.ended_immediately
        and sample.legal_action_count == 1
        and sample.prior_dragon_give,
    ),
    (
        "forced double win",
        lambda sample: sample.ended_immediately
        and sample.legal_action_count == 1
        and sample.scoring_kind == "double",
    ),
    (
        "forced normal + no prior Dragon",
        lambda sample: sample.ended_immediately
        and sample.legal_action_count == 1
        and sample.scoring_kind == "normal"
        and not sample.prior_dragon_give,
    ),
)


def print_terminal_report(
    samples: list[ValueSample], labels: tuple[str, str]
) -> None:
    terminal = [sample for sample in samples if sample.steps_to_end == 1]
    print("\nTerminal-state cohorts")
    header = (
        f"{'cohort':<38} {'model':<14} {'n':>6} {'share':>7} {'bias':>8} "
        f"{'MAE':>13} {'RMSE':>8} {'corr':>7}"
    )
    print(header)
    print("-" * len(header))
    for cohort, predicate in TERMINAL_COHORTS:
        for model in labels:
            model_terminal = [sample for sample in terminal if sample.model == model]
            selected = [sample for sample in model_terminal if predicate(sample)]
            if not selected:
                continue
            stats = summarize(selected)
            share = len(selected) / len(model_terminal) if model_terminal else 0.0
            print(
                f"{cohort:<38} {model:<14} {stats['n']:6d} {share:6.1%} "
                f"{stats['bias']:+8.1f} "
                f"{stats['mae']:6.1f}+/-{stats['mae_se']:<4.1f} "
                f"{stats['rmse']:8.1f} {stats['correlation']:7.3f}"
            )


def print_report(samples: list[ValueSample], labels: tuple[str, str]) -> None:
    print("\nCritic error in points (bias = prediction - realized score)")
    header = (
        f"{'steps':>7}  {'model':<14} {'n':>7} {'bias':>8} "
        f"{'MAE':>13} {'RMSE':>8} {'corr':>7}"
    )
    print(header)
    print("-" * len(header))
    for bin_label, low, high in DISTANCE_BINS:
        for model in labels:
            selected = [
                sample
                for sample in samples
                if sample.model == model
                and low <= sample.steps_to_end <= high
            ]
            if not selected:
                continue
            stats = summarize(selected)
            print(
                f"{bin_label:>7}  {model:<14} {stats['n']:7d} "
                f"{stats['bias']:+8.1f} "
                f"{stats['mae']:6.1f}+/-{stats['mae_se']:<4.1f} "
                f"{stats['rmse']:8.1f} {stats['correlation']:7.3f}"
            )

    print("\nOverall")
    for model in labels:
        stats = summarize([sample for sample in samples if sample.model == model])
        print(
            f"  {model}: n={stats['n']:,}  bias={stats['bias']:+.1f}  "
            f"MAE={stats['mae']:.1f}  RMSE={stats['rmse']:.1f}  "
            f"corr={stats['correlation']:.3f}"
        )
    print_terminal_report(samples, labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two critics by decisions remaining to round end"
    )
    parser.add_argument("policy_a")
    parser.add_argument("policy_b")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument(
        "--games",
        type=int,
        default=2048,
        help="Rounds per seat orientation",
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gt", action="store_true", help="Enable both bundled GT heads"
    )
    parser.add_argument(
        "--no-seat-swap",
        action="store_true",
        help="Run only policy A on team A and policy B on team B",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Evaluate each checkpoint against itself instead of head-to-head",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    collector = ValueTraceCollector({"a": args.label_a, "b": args.label_b})
    if args.self_play:
        matchups = (
            (args.policy_a, args.policy_a, args.label_a),
            (args.policy_b, args.policy_b, args.label_b),
        )
        for policy_a, policy_b, label in matchups:
            collector.team_labels = {"a": label, "b": label}
            evaluate(
                policy_a,
                policy_b,
                n_games=args.games,
                batch=args.batch,
                device=device,
                progress_every=0,
                gt_a=args.gt,
                gt_b=args.gt,
                post_pass_hook=collector.begin_batch,
                decision_hook=collector.record_decisions,
                post_action_hook=collector.record_action,
                post_round_hook=collector.finish_batch,
            )
    else:
        evaluate(
            args.policy_a,
            args.policy_b,
            n_games=args.games,
            batch=args.batch,
            device=device,
            progress_every=0,
            gt_a=args.gt,
            gt_b=args.gt,
            post_pass_hook=collector.begin_batch,
            decision_hook=collector.record_decisions,
            post_action_hook=collector.record_action,
            post_round_hook=collector.finish_batch,
        )
        if not args.no_seat_swap:
            collector.team_labels = {"a": args.label_b, "b": args.label_a}
            evaluate(
                args.policy_b,
                args.policy_a,
                n_games=args.games,
                batch=args.batch,
                device=device,
                progress_every=0,
                gt_a=args.gt,
                gt_b=args.gt,
                post_pass_hook=collector.begin_batch,
                decision_hook=collector.record_decisions,
                post_action_hook=collector.record_action,
                post_round_hook=collector.finish_batch,
            )

    print_report(collector.samples, (args.label_a, args.label_b))


if __name__ == "__main__":
    main()
