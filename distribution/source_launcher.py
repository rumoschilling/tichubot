"""Shared entry point for the source-based desktop distributions."""

from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import re
import sys
import traceback


APP_NAME = "TichuBot"
MINIMUM_PYTHON = (3, 10)
MINIMUM_DEPENDENCIES = {"numpy": (1, 24), "torch": (2, 0)}
ROOT = Path(__file__).resolve().parent

# Usually Python adds a script's directory automatically. Some embeddable
# Windows runtimes disable that behavior, so make the packaged modules
# importable explicitly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _version_pair(version: str) -> tuple[int, int]:
    numbers = re.match(r"^(\d+)\.(\d+)", version)
    if numbers is None:
        raise ValueError(f"Unrecognized package version: {version}")
    return int(numbers.group(1)), int(numbers.group(2))


def _check_dependencies() -> None:
    if sys.version_info < MINIMUM_PYTHON:
        installed = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(
            f"Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer is required; "
            f"this is Python {installed}."
        )

    problems: list[str] = []
    versions: list[str] = []
    for package, minimum in MINIMUM_DEPENDENCIES.items():
        try:
            version = metadata.version(package)
            if _version_pair(version) < minimum:
                problems.append(
                    f"{package} {version} is too old (need {minimum[0]}.{minimum[1]}+)"
                )
            versions.append(f"{package} {version}")
        except metadata.PackageNotFoundError:
            problems.append(f"{package} is not installed")

    if problems:
        raise RuntimeError("Dependency check failed: " + "; ".join(problems))

    # Importing catches broken or wrong-architecture wheels that package
    # metadata alone cannot detect.
    import numpy  # noqa: F401
    import torch  # noqa: F401

    checkpoint = ROOT / "model" / "tichubot.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"The bundled model is missing: {checkpoint}")
    print(
        f"Dependency check passed: Python {sys.version_info.major}."
        f"{sys.version_info.minor}; {', '.join(versions)}"
    )


def _smoke_test() -> None:
    import torch

    from training.play import load_play_bundle
    from training.web import LocalGame

    checkpoint = ROOT / "model" / "tichubot.pt"
    device = torch.device("cpu")
    policy, pass_policy, gt_policy, kind, info = load_play_bundle(
        str(checkpoint), None, device
    )
    if kind != "history":
        raise RuntimeError(f"Expected a history model, found {kind!r}")
    if not isinstance(info["games_played"], int):
        raise RuntimeError("The bundled checkpoint has no games_played count")
    if pass_policy is None or gt_policy is None:
        raise RuntimeError("The bundled checkpoint is missing its pass or GT model")
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
    if game.state()["mode"] != "play":
        raise RuntimeError("The browser game did not initialize")
    print("Bundled model smoke test passed.")


def _run() -> None:
    from training import web

    checkpoint = ROOT / "model" / "tichubot.pt"
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
        "0",
    ]
    web.main()


def _error_log_directory() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    configured_root = os.environ.get("XDG_STATE_HOME")
    state_root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path.home() / ".local" / "state"
    )
    return state_root / APP_NAME


def _write_error_log(details: str) -> Path | None:
    try:
        directory = _error_log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "error.log"
        path.write_text(details, encoding="utf-8")
        return path
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-dependencies", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.check_dependencies:
        try:
            _check_dependencies()
            return 0
        except Exception as error:
            sys.stderr.write(f"{error}\n")
            return 1

    try:
        _check_dependencies()
        if args.smoke_test:
            _smoke_test()
        else:
            _run()
        return 0
    except BaseException:
        details = traceback.format_exc()
        log_path = _write_error_log(details)
        sys.stderr.write(details)
        if log_path is not None:
            sys.stderr.write(f"\nError details were written to {log_path}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
