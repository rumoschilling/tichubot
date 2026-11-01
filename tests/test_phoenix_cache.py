from __future__ import annotations

import random

import pytest

from tichu_env import combinations as comb
from tichu_env.cards import DOG_INDEX, DRAGON_INDEX, MAHJONG_INDEX, PHOENIX_INDEX
from tichu_env.combinations import (
    _binary_phoenix_interpretations_uncached,
    binary_phoenix_interpretations,
)


def _candidate_sets(rng: random.Random, n: int) -> list[list[int]]:
    """Phoenix sets of 5-14 cards; mostly near-straights and near-full-houses."""
    out = []
    for _ in range(n):
        size = rng.randint(4, 13)
        if rng.random() < 0.6:
            start = rng.randint(2, 14 - size + 1) if size <= 13 else 2
            ranks = list(range(start, start + size))
            if rng.random() < 0.5 and len(ranks) > 1:
                ranks.pop(rng.randrange(len(ranks)))
        else:
            ranks = [rng.randint(2, 14) for _ in range(size)]
        cards = set()
        for r in ranks:
            for _ in range(4):
                c = rng.randrange(4) * 13 + (r - 2)
                if c not in cards:
                    cards.add(c)
                    break
        if rng.random() < 0.15:
            cards.add(MAHJONG_INDEX)
        if rng.random() < 0.03:
            cards.add(rng.choice([DOG_INDEX, DRAGON_INDEX]))
        cards.add(PHOENIX_INDEX)
        out.append(rng.sample(sorted(cards), len(cards)))
    return out


def _suit_variants(cards: list[int], rng: random.Random, k: int) -> list[list[int]]:
    """Same ranks, other suits (when the suit is free), shuffled order."""
    variants = []
    for _ in range(k):
        used: set[int] = set()
        new = []
        for c in cards:
            if c >= 52:
                new.append(c)
                continue
            options = [s * 13 + c % 13 for s in range(4) if s * 13 + c % 13 not in used]
            pick = rng.choice(options)
            used.add(pick)
            new.append(pick)
        rng.shuffle(new)
        variants.append(new)
    return variants


def _uncached_or_error(cards):
    try:
        return _binary_phoenix_interpretations_uncached(cards)
    except RuntimeError as exc:
        return exc


def test_cached_matches_uncached_across_suits_and_orders(monkeypatch):
    monkeypatch.setattr(comb, "_BINARY_PHOENIX_CACHE", {})
    rng = random.Random(20261018)
    checked = ambiguous = 0
    for cards in _candidate_sets(rng, 3000):
        for variant in [cards, *_suit_variants(cards, rng, 3)]:
            expected = _uncached_or_error(variant)
            if isinstance(expected, RuntimeError):
                with pytest.raises(RuntimeError):
                    binary_phoenix_interpretations(variant)
                continue
            got = binary_phoenix_interpretations(variant)
            assert got == expected, variant
            if got is not None:
                assert all(c.cards == tuple(sorted(variant)) for c in got)
                ambiguous += 1
            checked += 1
    assert checked > 10000
    assert ambiguous > 500  # the cache's hit path returns real meanings, not only None
    assert len(comb._BINARY_PHOENIX_CACHE) < checked


def test_short_or_phoenix_free_sets_bypass_cache(monkeypatch):
    monkeypatch.setattr(comb, "_BINARY_PHOENIX_CACHE", {})
    assert binary_phoenix_interpretations([0, 1, 2, 3, 4]) is None
    assert binary_phoenix_interpretations([0, 1, 2, PHOENIX_INDEX]) is None
    assert comb._BINARY_PHOENIX_CACHE == {}


def test_duplicate_cards_are_not_cached(monkeypatch):
    monkeypatch.setattr(comb, "_BINARY_PHOENIX_CACHE", {})
    assert binary_phoenix_interpretations([0, 0, 1, 2, PHOENIX_INDEX]) is None
    assert comb._BINARY_PHOENIX_CACHE == {}
