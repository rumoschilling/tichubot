"""Build every lightweight desktop distribution from one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from distribution.linux.build_linux import build as build_linux
from distribution.macos.build_macos import build as build_macos
from distribution.source_bundle import ROOT, resolve_checkpoint
from distribution.windows_source.build_windows_source import build as build_windows_source


Builder = Callable[[Path | None], Path]
BUILDERS: tuple[tuple[str, Builder], ...] = (
    ("Linux", build_linux),
    ("macOS", build_macos),
    ("Windows source", build_windows_source),
)
FULL_WINDOWS_ARTIFACTS = (
    ROOT / "dist" / "windows" / "TichuBot.exe",
    ROOT / "dist" / "windows" / "TichuBot-Windows.zip",
)


def _file_state(path: Path) -> tuple[bool, int, int]:
    if not path.exists():
        return False, 0, 0
    stat = path.stat()
    return True, stat.st_size, stat.st_mtime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_all(checkpoint_path: Path | None = None) -> list[Path]:
    """Build all source ZIPs while proving the full Windows build is untouched."""

    checkpoint, games = resolve_checkpoint(checkpoint_path)
    print(f"Release checkpoint: {checkpoint} ({games:,} games)", flush=True)
    full_windows_before = {
        path: _file_state(path) for path in FULL_WINDOWS_ARTIFACTS
    }

    artifacts = [builder(checkpoint) for _, builder in BUILDERS]

    full_windows_after = {
        path: _file_state(path) for path in FULL_WINDOWS_ARTIFACTS
    }
    if full_windows_after != full_windows_before:
        raise RuntimeError("A full Windows artifact changed during the lightweight build.")

    print("\nLightweight release artifacts:", flush=True)
    for (label, _), artifact in zip(BUILDERS, artifacts):
        print(f"  {label}: {artifact}", flush=True)
        print(f"    SHA-256: {_sha256(artifact)}", flush=True)
    excluded = ", ".join(str(path) for path in FULL_WINDOWS_ARTIFACTS)
    print(f"Excluded and left untouched: {excluded}", flush=True)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Linux, macOS, and Windows-source TichuBot ZIPs."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint for every archive (default: highest games_played found locally).",
    )
    args = parser.parse_args()
    build_all(args.checkpoint)


if __name__ == "__main__":
    main()
