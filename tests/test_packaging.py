from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_offline_wheel_contains_runtime_packages_and_browser_assets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(ROOT / "README.md", source / "README.md")
    shutil.copytree(ROOT / "training", source / "training")
    shutil.copytree(ROOT / "tichu_env", source / "tichu_env")

    dist = tmp_path / "dist"
    script = (
        "import setuptools.build_meta as backend, sys; "
        "print(backend.build_wheel(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(dist)],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    assert "training/web.py" in names
    assert "training/observations.py" in names
    assert "training/web_static/index.html" in names
    assert "training/web_static/app.js" in names
    assert "training/web_static/assets/tichu-original-deck.svg" in names
    assert "tichu_env/game_state.py" in names
    assert not any(name.startswith("tests/") for name in names)
    assert not any(name.startswith("distribution/") for name in names)
    assert not any(name.startswith("kaggle/") for name in names)
    assert not any(name.startswith("checkpoints/") for name in names)
    assert not any(name.startswith("kaggle_runs/") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)

    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel.extractall(installed)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import training.web as web; "
            "assert Path(web.__file__).is_relative_to(Path.cwd() / 'installed'); "
            "assert (web.WEB_ROOT / 'index.html').is_file()",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr

