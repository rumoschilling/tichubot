"""PyInstaller hook for CPU inference without importing Torch under Wine."""

from PyInstaller import compat
from PyInstaller.utils.hooks import (
    PY_DYLIB_PATTERNS,
    collect_data_files,
    collect_dynamic_libs,
)


warn_on_missing_hiddenimports = False
hiddenimports = []
module_collection_mode = "pyz+py"
datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
    ],
)
binaries = collect_dynamic_libs(
    "torch",
    search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"],
)

# The CPU wheel used here does not declare the separate MKL distribution.
# Its required Windows DLLs live under torch/lib and are collected above.
assert compat.is_win
