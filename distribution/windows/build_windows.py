"""Build a self-contained Windows executable locally through Docker/Wine."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

import torch


ROOT = Path(__file__).resolve().parents[2]
BUILDER_DIR = Path(__file__).resolve().parent
STAGING_DIR = ROOT / "build" / "windows"
OUTPUT_DIR = ROOT / "dist" / "windows"
IMAGE = "tichubot-windows-builder:latest"
TORCH_WHEEL_NAME = "torch-2.11.0+cpu-cp311-cp311-win_amd64.whl"
TORCH_WHEEL_URL = (
    "https://download.pytorch.org/whl/cpu/"
    "torch-2.11.0%2Bcpu-cp311-cp311-win_amd64.whl"
)
NUMPY_WHEEL_NAME = "numpy-1.26.4-cp311-cp311-win_amd64.whl"
NUMPY_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/3f/6b/"
    "5610004206cf7f8e7ad91c5a85a8c71b2f2f8051a0c0c4d5916b76d6cbb2/"
    f"{NUMPY_WHEEL_NAME}"
)

# Checkpoints pickle TrainConfig by module name. A directly executed script's
# import path starts in distribution/windows rather than at the repository.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _checkpoint_candidates() -> list[Path]:
    patterns = (
        "kaggle_runs/**/checkpoints/history_standard_gpu/latest.pt",
        "checkpoints/**/latest.pt",
        "kagglecheckpoints/**/*.pt",
    )
    return sorted(
        {path for pattern in patterns for path in ROOT.glob(pattern)},
        key=lambda path: str(path),
    )


def _checkpoint_games(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("games_played", 0))


def _latest_checkpoint() -> tuple[Path, int]:
    ranked = [
        (_checkpoint_games(path), path)
        for path in _checkpoint_candidates()
    ]
    if not ranked:
        raise FileNotFoundError("No local checkpoint was found.")
    games, path = max(ranked, key=lambda item: (item[0], item[1].stat().st_mtime))
    return path, games


def _stage_checkpoint(source: Path) -> tuple[Path, int]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = ("games_played", "model", "config", "pass_model", "gt_model")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint {source} is missing packaged inference data: {missing}"
        )
    inference_checkpoint = {key: checkpoint[key] for key in required}
    model_dir = STAGING_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "tichubot.pt"
    torch.save(inference_checkpoint, target)
    return target, int(checkpoint["games_played"])


def _docker(*args: str) -> None:
    subprocess.run(["docker", *args], check=True, cwd=ROOT)


def _valid_wheel(path: Path, *, minimum_size: int, required_member: str) -> bool:
    if not path.is_file() or path.stat().st_size < minimum_size:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            archive.getinfo(required_member)
            return archive.testzip() is None
    except (OSError, KeyError, zipfile.BadZipFile):
        return False


def _ensure_wheel(
    label: str,
    filename: str,
    url: str,
    *,
    minimum_size: int,
    required_member: str,
) -> Path:
    """Keep Windows wheels resumable outside transactional Docker layers."""
    target = BUILDER_DIR / ".cache" / filename
    partial = target.with_suffix(f"{target.suffix}.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    is_valid = lambda path: _valid_wheel(
        path,
        minimum_size=minimum_size,
        required_member=required_member,
    )
    if is_valid(target):
        print(f"Using cached Windows {label} wheel: {target}", flush=True)
        return target
    if is_valid(partial):
        partial.replace(target)
        return target
    target.unlink(missing_ok=True)

    failures = 0
    while failures < 20:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if offset and response.status != 206:
                    offset = 0
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if is_valid(partial):
                partial.replace(target)
                print(f"Downloaded and validated: {target}", flush=True)
                return target
            failures += 1
        except (OSError, TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            failures += 1
            print(
                f"{label} download paused at {offset / 2**20:.1f} MiB "
                f"({exc}); reconnecting {failures}/20...",
                flush=True,
            )
        time.sleep(min(failures, 5))
    raise RuntimeError(f"Could not download {url}")


def _ensure_windows_wheels() -> None:
    _ensure_wheel(
        "PyTorch",
        TORCH_WHEEL_NAME,
        TORCH_WHEEL_URL,
        minimum_size=100 * 2**20,
        required_member="torch/__init__.py",
    )
    _ensure_wheel(
        "NumPy",
        NUMPY_WHEEL_NAME,
        NUMPY_WHEEL_URL,
        minimum_size=10 * 2**20,
        required_member="numpy/__init__.py",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(checkpoint_path: Path | None, *, skip_image_build: bool) -> Path:
    _ensure_windows_wheels()
    if checkpoint_path is None:
        checkpoint_path, discovered_games = _latest_checkpoint()
        print(
            f"Selected latest local checkpoint: {checkpoint_path.relative_to(ROOT)} "
            f"({discovered_games:,} games)",
            flush=True,
        )
    else:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

    staged, games = _stage_checkpoint(checkpoint_path)
    print(
        f"Staged inference-only model: {staged.relative_to(ROOT)} "
        f"({staged.stat().st_size / 2**20:.1f} MiB, {games:,} games)",
        flush=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exe = OUTPUT_DIR / "TichuBot.exe"
    exe.unlink(missing_ok=True)

    if not skip_image_build:
        _docker("build", "--progress=plain", "-t", IMAGE, str(BUILDER_DIR))
    _docker(
        "run",
        "--rm",
        "--entrypoint",
        "/workspace/distribution/windows/entrypoint.sh",
        "-e",
        f"HOST_UID={os.getuid()}",
        "-e",
        f"HOST_GID={os.getgid()}",
        "-v",
        f"{ROOT}:/workspace:ro",
        "-v",
        f"{STAGING_DIR}:/staging:ro",
        "-v",
        f"{OUTPUT_DIR}:/out",
        IMAGE,
        "build",
    )
    _docker(
        "run",
        "--rm",
        "--entrypoint",
        "/workspace/distribution/windows/entrypoint.sh",
        "-v",
        f"{ROOT}:/workspace:ro",
        "-v",
        f"{OUTPUT_DIR}:/out:ro",
        IMAGE,
        "smoke",
    )
    if not exe.is_file():
        raise RuntimeError(f"Builder did not create {exe}")

    checksum = _sha256(exe)
    checksum_path = OUTPUT_DIR / "TichuBot.exe.sha256"
    checksum_path.write_text(f"{checksum}  TichuBot.exe\n", encoding="ascii")
    print(f"Built and smoke-tested: {exe}", flush=True)
    print(f"Size: {exe.stat().st_size / 2**20:.1f} MiB", flush=True)
    print(f"SHA-256: {checksum}", flush=True)
    return exe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to embed (default: highest games_played found locally).",
    )
    parser.add_argument(
        "--skip-image-build",
        action="store_true",
        help="Reuse the existing local Docker builder image.",
    )
    args = parser.parse_args()
    build(args.checkpoint, skip_image_build=args.skip_image_build)


if __name__ == "__main__":
    main()
