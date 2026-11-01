import json

import pytest

from training import advisor, checkpoints, play, promote_best, web


def test_interactive_tools_share_the_best_checkpoint():
    expected = checkpoints.BEST_CHECKPOINT

    assert checkpoints.DEFAULT_PLAY_CHECKPOINT == expected
    assert checkpoints.MODEL_ALIASES["best"] == expected
    assert checkpoints.MODEL_ALIASES["play"] == expected
    assert checkpoints.MODEL_ALIASES["standard"] == checkpoints.STANDARD_CHECKPOINT
    assert play.DEFAULT_CHECKPOINT == expected
    assert advisor.DEFAULT_CHECKPOINT == expected
    assert web.DEFAULT_CHECKPOINT == expected
    assert web.build_arg_parser().get_default("checkpoint") == expected


def test_best_record_carries_promotion_evidence():
    record = checkpoints.read_best_record()

    assert record["path"] == checkpoints.BEST_CHECKPOINT
    assert record["evidence"]["z"] >= promote_best.MIN_Z
    assert record["evidence"]["previous_best"] == record["history"][-1]["path"]


def test_paired_z_uses_block_standard_error():
    stats = promote_best.paired_z([1.0, 3.0, 5.0, 7.0])

    assert stats["mean"] == pytest.approx(4.0)
    assert stats["se"] == pytest.approx(1.2909944, rel=1e-6)
    assert stats["z"] == pytest.approx(3.0983867, rel=1e-6)


def _write_record(tmp_path, path="old.pt"):
    record_path = tmp_path / "best_checkpoint.json"
    record_path.write_text(json.dumps({"path": path, "history": []}))
    return record_path


def test_gate_keeps_best_below_two_standard_errors(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path)
    monkeypatch.setattr(promote_best, "resolve_checkpoint", lambda ref: tmp_path / ref)
    # mean 1.0, se 0.577 -> z 1.73
    comparison = {"block_means": [0.0, 2.0, -1.0, 3.0, 1.0, 1.0], "deal_pairs": 768}

    outcome = promote_best.decide("new.pt", comparison, str(tmp_path / "eval.json"), record_path=record_path)

    assert not outcome["promoted"]
    assert json.loads(record_path.read_text())["path"] == "old.pt"


def test_gate_promotes_at_two_standard_errors(tmp_path, monkeypatch):
    record_path = _write_record(tmp_path)
    monkeypatch.setattr(promote_best, "resolve_checkpoint", lambda ref: tmp_path / ref)
    monkeypatch.setattr(
        promote_best, "checkpoint_info", lambda ref: type("Info", (), {"games_played": 123})()
    )
    comparison = {"block_means": [4.0, 5.0, 6.0, 5.0], "deal_pairs": 512}

    outcome = promote_best.decide("new.pt", comparison, str(tmp_path / "eval.json"), record_path=record_path)

    record = json.loads(record_path.read_text())
    assert outcome["promoted"]
    assert record["path"].endswith("new.pt")
    assert record["games_played"] == 123
    assert record["evidence"]["previous_best"] == "old.pt"
    assert record["history"][-1]["path"] == "old.pt"
