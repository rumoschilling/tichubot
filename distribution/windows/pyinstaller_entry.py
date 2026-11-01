"""Run PyInstaller without importing Torch during Windows DLL discovery.

The custom Torch hook already collects the wheel's DLLs. Importing every
collected Torch package a second time makes Intel OpenMP query Wine's
unimplemented NUMA API and aborts the build.
"""

from __future__ import annotations

import PyInstaller.__main__
from PyInstaller.building import build_main


_find_binary_dependencies = build_main.find_binary_dependencies


def _find_without_torch_imports(binaries, import_packages, suppression_patterns):
    torch_package_roots = ("torch", "functorch", "torchgen")
    import_packages = [
        package
        for package in import_packages
        if not any(
            package == root or package.startswith(f"{root}.")
            for root in torch_package_roots
        )
    ]
    return _find_binary_dependencies(
        binaries,
        import_packages,
        suppression_patterns,
    )


build_main.find_binary_dependencies = _find_without_torch_imports
PyInstaller.__main__.run()
