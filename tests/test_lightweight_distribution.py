from pathlib import Path
import subprocess
import sys

from distribution import build_lightweight
from distribution.linux.build_linux import SPEC as LINUX_SPEC
from distribution.source_bundle import _copy_runtime


def test_lightweight_builder_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "distribution/build_lightweight.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Linux, macOS, and Windows-source" in result.stdout


def test_staged_runtime_imports_browser_application(tmp_path):
    package_dir = tmp_path / "TichuBot-Linux"
    package_dir.mkdir()
    _copy_runtime(LINUX_SPEC, package_dir)

    assert (package_dir / "training" / "observations.py").is_file()
    result = subprocess.run(
        [sys.executable, "-c", "import training.web"],
        cwd=package_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_lightweight_builder_reuses_checkpoint_and_preserves_full_windows(
    monkeypatch, tmp_path, capsys
):
    checkpoint = tmp_path / "release.pt"
    checkpoint.write_bytes(b"checkpoint")
    full_exe = tmp_path / "TichuBot.exe"
    full_zip = tmp_path / "TichuBot-Windows.zip"
    full_exe.write_bytes(b"full executable")
    full_zip.write_bytes(b"full archive")
    calls: list[tuple[str, Path]] = []

    def builder(label: str):
        def run(path: Path | None) -> Path:
            assert path is not None
            calls.append((label, path))
            artifact = tmp_path / f"{label}.zip"
            artifact.write_bytes(label.encode("ascii"))
            return artifact

        return run

    monkeypatch.setattr(
        build_lightweight,
        "resolve_checkpoint",
        lambda path: (checkpoint, 123_456),
    )
    monkeypatch.setattr(
        build_lightweight,
        "BUILDERS",
        (("Linux", builder("linux")), ("macOS", builder("mac")),
         ("Windows source", builder("windows-source"))),
    )
    monkeypatch.setattr(
        build_lightweight,
        "FULL_WINDOWS_ARTIFACTS",
        (full_exe, full_zip),
    )

    artifacts = build_lightweight.build_all()

    assert [path for _, path in calls] == [checkpoint, checkpoint, checkpoint]
    assert artifacts == [
        tmp_path / "linux.zip",
        tmp_path / "mac.zip",
        tmp_path / "windows-source.zip",
    ]
    assert full_exe.read_bytes() == b"full executable"
    assert full_zip.read_bytes() == b"full archive"
    output = capsys.readouterr().out
    assert "123,456 games" in output
    assert "Excluded and left untouched" in output
