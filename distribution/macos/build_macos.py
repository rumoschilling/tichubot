"""Build the source-based macOS TichuBot ZIP."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distribution.source_bundle import (
    SourceBundleSpec,
    build_source_bundle,
    builder_main,
)


SPEC = SourceBundleSpec(
    key="macos",
    builder_dir=Path(__file__).resolve().parent,
    package_name="TichuBot-Mac",
    archive_name="TichuBot-Mac.zip",
    output_subdir="macos",
    start_script="Start TichuBot.command",
    requirements_file="requirements-macos.txt",
    executable_start_script=True,
)


def build(checkpoint_path: Path | None = None) -> Path:
    return build_source_bundle(SPEC, checkpoint_path)


if __name__ == "__main__":
    builder_main(SPEC)
