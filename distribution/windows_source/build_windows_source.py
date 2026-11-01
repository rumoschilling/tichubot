"""Build the source-based Windows TichuBot ZIP."""

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
    key="windows-source",
    builder_dir=Path(__file__).resolve().parent,
    package_name="TichuBot-Windows-Source",
    archive_name="TichuBot-Windows-Source.zip",
    output_subdir="windows",
    start_script="Start TichuBot.bat",
    requirements_file="requirements-windows.txt",
    windows_line_endings=True,
)


def build(checkpoint_path: Path | None = None) -> Path:
    return build_source_bundle(SPEC, checkpoint_path)


if __name__ == "__main__":
    builder_main(SPEC)
