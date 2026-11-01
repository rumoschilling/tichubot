"""Shared builder for source-based Windows, macOS, and Linux ZIPs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


# This is the transitive source set used by training.web. The omitted training
# modules are where Gymnasium, PettingZoo, Stable Baselines, and SuperSuit enter
# the import graph; the browser runtime needs only NumPy and PyTorch.
RUNTIME_FILES = (
    "tichu_env/__init__.py",
    "tichu_env/cards.py",
    "tichu_env/combinations.py",
    "tichu_env/constants.py",
    "tichu_env/display.py",
    "tichu_env/encoding.py",
    "tichu_env/game_state.py",
    "training/__init__.py",
    "training/advisor.py",
    "training/best_checkpoint.json",
    "training/checkpoints.py",
    "training/config.py",
    "training/model.py",
    "training/model_embed.py",
    "training/model_gt.py",
    "training/model_history.py",
    "training/model_passing.py",
    "training/observations.py",
    "training/play.py",
    "training/rollout_embed.py",
    "training/web.py",
)


@dataclass(frozen=True)
class SourceBundleSpec:
    key: str
    builder_dir: Path
    package_name: str
    archive_name: str
    output_subdir: str
    start_script: str
    requirements_file: str
    executable_start_script: bool = False
    windows_line_endings: bool = False


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


def _best_checkpoint() -> tuple[Path, int] | None:
    """BEST_CHECKPOINT (training/best_checkpoint.json), if its file is present."""
    from training.checkpoints import read_best_record

    path = ROOT / read_best_record()["path"]
    return (path, _checkpoint_games(path)) if path.is_file() else None


def resolve_checkpoint(checkpoint_path: Path | None) -> tuple[Path, int]:
    """Resolve one explicit or automatically selected release checkpoint.

    The automatic choice is BEST_CHECKPOINT, the gate-promoted strongest model.
    Only if that file is missing locally does it fall back to the checkpoint
    with the most training games, which is not necessarily the strongest.
    """

    if checkpoint_path is None:
        return _best_checkpoint() or _latest_checkpoint()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path, _checkpoint_games(checkpoint_path)


def _stage_checkpoint(source: Path, package_dir: Path) -> tuple[Path, int]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = ("games_played", "model", "config", "pass_model", "gt_model")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint {source} is missing packaged inference data: {missing}"
        )
    inference_checkpoint = {key: checkpoint[key] for key in required}
    model_dir = package_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "tichubot.pt"
    torch.save(inference_checkpoint, target)
    return target, int(checkpoint["games_played"])


def _copy_start_script(spec: SourceBundleSpec, package_dir: Path) -> None:
    source = spec.builder_dir / spec.start_script
    target = package_dir / spec.start_script
    if spec.windows_line_endings:
        contents = source.read_text(encoding="utf-8")
        normalized = contents.replace("\r\n", "\n").replace("\n", "\r\n")
        target.write_bytes(normalized.encode("utf-8"))
    else:
        shutil.copy2(source, target)
    if spec.executable_start_script:
        target.chmod(
            target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )


def _copy_runtime(spec: SourceBundleSpec, package_dir: Path) -> None:
    for relative in RUNTIME_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required runtime source is missing: {source}")
        target = package_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    shutil.copytree(
        ROOT / "training" / "web_static",
        package_dir / "training" / "web_static",
    )
    shutil.copy2(ROOT / "distribution" / "source_launcher.py", package_dir / "run_tichubot.py")
    shutil.copy2(spec.builder_dir / "README.md", package_dir / "README.md")
    shutil.copy2(
        spec.builder_dir / spec.requirements_file,
        package_dir / spec.requirements_file,
    )
    _copy_start_script(spec, package_dir)


def _smoke_test(package_dir: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "run_tichubot.py", "--smoke-test"],
        cwd=package_dir,
        env=environment,
        check=True,
    )


def _zip_package(package_dir: Path, archive: Path, package_name: str) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(package_dir.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                output.write(path, Path(package_name) / path.relative_to(package_dir))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_bundle(
    spec: SourceBundleSpec,
    checkpoint_path: Path | None,
) -> Path:
    automatic = checkpoint_path is None
    checkpoint_path, discovered_games = resolve_checkpoint(checkpoint_path)
    if automatic:
        print(
            f"Selected release checkpoint (BEST_CHECKPOINT unless missing): {checkpoint_path.relative_to(ROOT)} "
            f"({discovered_games:,} games)",
            flush=True,
        )

    build_root = ROOT / "build" / spec.key
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tichubot-", dir=build_root) as temporary:
        package_dir = Path(temporary) / spec.package_name
        package_dir.mkdir()
        _copy_runtime(spec, package_dir)
        staged, games = _stage_checkpoint(checkpoint_path, package_dir)
        print(
            f"Staged inference-only model: {staged.stat().st_size / 2**20:.1f} MiB, "
            f"{games:,} games",
            flush=True,
        )
        _smoke_test(package_dir)
        archive = ROOT / "dist" / spec.output_subdir / spec.archive_name
        _zip_package(package_dir, archive, spec.package_name)

    checksum = _sha256(archive)
    checksum_path = archive.with_suffix(f"{archive.suffix}.sha256")
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(f"Built and smoke-tested: {archive}", flush=True)
    print(f"Size: {archive.stat().st_size / 2**20:.1f} MiB", flush=True)
    print(f"SHA-256: {checksum}", flush=True)
    return archive


def builder_main(spec: SourceBundleSpec) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to embed (default: highest games_played found locally).",
    )
    args = parser.parse_args()
    build_source_bundle(spec, args.checkpoint)
