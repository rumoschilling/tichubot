#!/bin/bash

# Finder runs .command files from the user's home directory. Always resolve
# paths relative to this extracted TichuBot folder instead.
cd -- "$(dirname -- "$0")" || exit 1

pause_after_error() {
    printf '\nPress Return to close this window. '
    read -r _
}

fail() {
    printf '\nTichuBot could not start: %s\n' "$1" >&2
    pause_after_error
    exit 1
}

if [ "$(uname -s)" != "Darwin" ]; then
    fail "this launcher is for macOS."
fi

if ! command -v python3 >/dev/null 2>&1; then
    fail "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/macos/ and try again."
fi

if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    fail "Python 3.10 or newer is required. The python3 on PATH is too old."
fi

if [ ! -x ".tichubot-venv/bin/python" ]; then
    printf 'Preparing TichuBot for its first run...\n'
    python3 -m venv .tichubot-venv || fail "Python could not create its virtual environment."
fi

PYTHON=".tichubot-venv/bin/python"

if ! "$PYTHON" run_tichubot.py --check-dependencies; then
    printf '\nInstalling the required Python packages. Internet access is needed for this first-time step.\n'
    "$PYTHON" -m pip install --disable-pip-version-check --requirement requirements-macos.txt \
        || fail "the Python dependencies could not be installed. Check the messages above and your internet connection."
fi

"$PYTHON" run_tichubot.py --check-dependencies \
    || fail "the dependency check still fails after installation."

printf '\nStarting TichuBot...\n'
"$PYTHON" run_tichubot.py
status=$?
if [ "$status" -ne 0 ]; then
    pause_after_error
fi
exit "$status"
