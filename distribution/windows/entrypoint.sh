#!/bin/sh
set -eu

PYTHON='C:\Python311\python.exe'
WORKSPACE='Z:\workspace'
STAGING='Z:\staging'
OUTPUT='Z:\out'

build_exe() {
    rm -rf /tmp/pyinstaller
    mkdir -p /tmp/pyinstaller /out
    xvfb-run -a wine "$PYTHON" \
        "$WORKSPACE\\distribution\\windows\\pyinstaller_entry.py" \
        --noconfirm \
        --clean \
        --onefile \
        --windowed \
        --noupx \
        --name TichuBot \
        --paths "$WORKSPACE" \
        --additional-hooks-dir "$WORKSPACE\\distribution\\windows\\hooks" \
        --add-data "$WORKSPACE\\training\\web_static;training\\web_static" \
        --add-data "$STAGING\\model\\tichubot.pt;model" \
        --hidden-import training.config \
        --hidden-import training.model_history \
        --hidden-import training.model_gt \
        --exclude-module gymnasium \
        --exclude-module pettingzoo \
        --exclude-module stable_baselines3 \
        --exclude-module supersuit \
        --distpath "$OUTPUT" \
        --workpath 'C:\pyinstaller\work' \
        --specpath 'C:\pyinstaller' \
        "$WORKSPACE\\distribution\\windows\\launcher.py"
    chown "${HOST_UID:-0}:${HOST_GID:-0}" /out/TichuBot.exe
}

smoke_test() {
    KMP_TOPOLOGY_METHOD=flat TICHUBOT_SMOKE_TEST=1 \
        xvfb-run -a wine 'Z:\out\TichuBot.exe'
}

serve_exe() {
    KMP_TOPOLOGY_METHOD=flat TICHUBOT_NO_BROWSER=1 \
        TICHUBOT_PORT="${TICHUBOT_PORT:-8765}" \
        xvfb-run -a wine 'Z:\out\TichuBot.exe'
}

case "${1:-build}" in
    build) build_exe ;;
    smoke) smoke_test ;;
    serve) serve_exe ;;
    *) echo "Usage: $0 [build|smoke|serve]" >&2; exit 2 ;;
esac
