"""Windows entry point for the self-contained visual Tichu application."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import traceback

import torch


APP_NAME = "TichuBot"


def _resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[2]


def _checkpoint_path() -> Path:
    return _resource_root() / "model" / "tichubot.pt"


def _message_box(text: str, title: str, flags: int) -> int:
    return int(ctypes.windll.user32.MessageBoxW(None, text, title, flags))


def _smoke_test(checkpoint: Path) -> None:
    from training.play import load_play_bundle
    from training.web import LocalGame

    device = torch.device("cpu")
    policy, pass_policy, gt_policy, kind, info = load_play_bundle(
        str(checkpoint), None, device
    )
    assert kind == "history"
    assert isinstance(info["games_played"], int)
    assert pass_policy is not None
    assert gt_policy is not None
    game = LocalGame(
        mode="play",
        seat=0,
        policy=policy,
        pass_policy=pass_policy,
        gt_policy=gt_policy,
        kind=kind,
        device=device,
    )
    game.new_round(seed=31)
    assert game.state()["mode"] == "play"


def _run() -> None:
    checkpoint = _checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Bundled model is missing: {checkpoint}")

    if os.environ.get("TICHUBOT_SMOKE_TEST") == "1":
        _smoke_test(checkpoint)
        return

    from training import web

    sys.argv = [
        sys.argv[0],
        "--mode",
        "play",
        "--menu",
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cpu",
        "--port",
        os.environ.get("TICHUBOT_PORT", "0"),
    ]
    if os.environ.get("TICHUBOT_NO_BROWSER") == "1":
        sys.argv.append("--no-browser")
    web.main()


def _error_log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    directory = base / APP_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "error.log"


def main() -> int:
    try:
        _run()
        return 0
    except BaseException:
        details = traceback.format_exc()
        try:
            log_path = _error_log_path()
            log_path.write_text(details, encoding="utf-8")
            message = f"TichuBot could not start. Details were written to:\n{log_path}"
        except OSError:
            message = f"TichuBot could not start:\n\n{details}"
        if os.name == "nt" and os.environ.get("TICHUBOT_SMOKE_TEST") != "1":
            _message_box(message, f"{APP_NAME} error", 0x00000010)
        else:
            sys.stderr.write(details)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
