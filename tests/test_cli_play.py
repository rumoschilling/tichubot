import random

import play as cli_play
from tichu_env.game_state import GameState


def test_random_cli_ai_handles_tuple_actions(capsys):
    gs = GameState()
    gs.reset(seed=1)
    player = gs.current_player
    before = len(gs.hands[player])

    cli_play.ai_play(gs, random.Random(1))

    assert len(gs.hands[player]) < before
    assert f"P{player}:" in capsys.readouterr().out


def test_random_cli_human_action_display_handles_tuple_actions(monkeypatch, capsys):
    actions = [([0], False), ([1], True)]
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    choice = cli_play.choose_action(actions, GameState(), 0)

    assert choice == 1
    assert "TICHU" in capsys.readouterr().out
