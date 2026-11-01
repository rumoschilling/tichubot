#!/usr/bin/env bash

cd -- "$(dirname -- "$0")" || exit 1

fail() {
    printf '\nTichuBot could not start: %s\n' "$1" >&2
    exit 1
}

if [ "$(uname -s)" != "Linux" ]; then
    fail "this launcher is for Linux."
fi

if ! command -v python3 >/dev/null 2>&1; then
    fail "Python 3.10 or newer is required. Install Python 3 with your package manager and try again."
fi

if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    fail "Python 3.10 or newer is required. The python3 on PATH is too old."
fi

if [ ! -x ".tichubot-venv/bin/python" ]; then
    printf 'Preparing TichuBot for its first run...\n'
    python3 -m venv .tichubot-venv || fail \
        "Python could not create a virtual environment. On Debian or Ubuntu, install the python3-venv package."
fi

PYTHON=".tichubot-venv/bin/python"

if ! "$PYTHON" run_tichubot.py --check-dependencies; then
    printf '\nInstalling the required CPU packages. Internet access is needed for this first-time step.\n'
    "$PYTHON" -m pip install --disable-pip-version-check \
        --requirement requirements-linux.txt \
        || fail "NumPy could not be installed. Check the messages above and your internet connection."
    "$PYTHON" -m pip install --disable-pip-version-check \
        --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
        || fail "CPU-only PyTorch could not be installed. Check whether your Python version and CPU architecture are supported."
fi

"$PYTHON" run_tichubot.py --check-dependencies \
    || fail "the dependency check still fails after installation."

printf '\nStarting TichuBot...\n'
exec "$PYTHON" run_tichubot.py
