from __future__ import annotations

import pytest

from training.checkpoint_curve_eval import _validate_checkpoint_grid


def test_validate_checkpoint_grid_returns_exact_spacing():
    manifests = [
        {"games_played": 250_880},
        {"games_played": 500_736},
        {"games_played": 750_592},
    ]

    assert _validate_checkpoint_grid(manifests) == (
        249_856,
        [249_856, 249_856],
    )


def test_validate_checkpoint_grid_allows_documented_resume_offset():
    manifests = [
        {"games_played": 1_500_160},
        {"games_played": 1_750_016},
        {"games_played": 2_000_896},
        {"games_played": 2_250_752},
    ]

    nominal, deltas = _validate_checkpoint_grid(
        manifests,
        tolerance_games=1024,
    )

    assert nominal == 249_856
    assert deltas == [249_856, 250_880, 249_856]


@pytest.mark.parametrize(
    "games",
    [
        [250_880],
        [500_736, 250_880],
        [250_880, 500_736, 751_616],
        [250_880, None],
    ],
)
def test_validate_checkpoint_grid_rejects_invalid_grids(games):
    with pytest.raises(ValueError):
        _validate_checkpoint_grid(
            [{"games_played": games_played} for games_played in games]
        )
