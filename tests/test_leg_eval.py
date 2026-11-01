"""Tests for the post-fetch eval drill (training.leg_eval, training.gt_check)."""
import os

from tichu_env.cards import (
    DOG_INDEX, DRAGON_INDEX, PHOENIX_INDEX, card_index, Rank, Suit,
)
from training.gt_check import STRENGTH_BUCKETS, hand_strength
from training.leg_eval import find_run_checkpoints


def test_hand_strength_specials():
    assert hand_strength([DRAGON_INDEX]) == 4
    assert hand_strength([PHOENIX_INDEX]) == 4
    assert hand_strength([DOG_INDEX]) == -1


def test_hand_strength_honors():
    ace = card_index(Suit.JADE, Rank.ACE)
    king = card_index(Suit.STAR, Rank.KING)
    queen = card_index(Suit.SWORD, Rank.QUEEN)
    two = card_index(Suit.PAGODA, Rank.TWO)
    assert hand_strength([ace]) == 3
    assert hand_strength([king]) == 2
    assert hand_strength([queen]) == 1
    assert hand_strength([two]) == 0
    assert hand_strength([ace, king, DRAGON_INDEX]) == 9


def test_strength_buckets_cover_full_range():
    # every reachable strength (min -1 = lone Dog, max 8*4 impossible but
    # bounded by 99) falls into exactly one bucket
    for s in range(-1, 30):
        hits = [(lo, hi) for lo, hi in STRENGTH_BUCKETS if lo <= s < hi]
        assert len(hits) == 1, f"strength {s} in {len(hits)} buckets"


def test_find_run_checkpoints_sorted(tmp_path):
    for name in ["2026-07-11_103232_standard", "2026-06-28_standard_v23",
                 "2026-07-06_114928_standard"]:
        d = tmp_path / name / "tichubot_outputs" / "checkpoints" / "history_x"
        d.mkdir(parents=True)
        (d / "latest.pt").touch()
    # a run without a checkpoint must be ignored
    (tmp_path / "2026-07-12_broken" / "tichubot_outputs").mkdir(parents=True)

    ckpts = find_run_checkpoints(str(tmp_path))
    assert len(ckpts) == 3
    names = [c.split(os.sep) for c in ckpts]
    assert names[0][-5] == "2026-06-28_standard_v23"
    assert names[-1][-5] == "2026-07-11_103232_standard"
