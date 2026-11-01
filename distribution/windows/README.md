# Windows executable

Build the visual Play and Advisor application as one self-contained Windows
executable. The recipient does not need Python. The build contains CPU-only
PyTorch, the card assets, and one inference-only checkpoint.

Requirements on the build machine:

- Docker
- the project's local Python environment, with PyTorch, for selecting and
  stripping the checkpoint

Build with the highest `games_played` checkpoint found locally:

```bash
python \
  distribution/windows/build_windows.py
```

Or select a checkpoint explicitly:

```bash
python \
  distribution/windows/build_windows.py --checkpoint path/to/latest.pt
```

The first invocation builds the Docker/Wine toolchain and downloads Windows
CPU PyTorch. Later builds can reuse it:

```bash
python \
  distribution/windows/build_windows.py --skip-image-build
```

Output:

```text
dist/windows/TichuBot.exe
dist/windows/TichuBot.exe.sha256
```

The application opens one local UI in the default browser. Its first screen
selects visual **Play** or the real-table **Advisor**, and the **Modes** button
switches between them. Use **Exit** to stop the bundled localhost server.

Test the actual Windows executable on Linux through the builder's Wine runtime:

```bash
docker run --rm --network host \
  -v "$PWD/dist/windows:/out:ro" \
  tichubot-windows-builder:latest serve
```

Then open `http://127.0.0.1:8765`. This runs `TichuBot.exe`, not the host
Python code. The browser remains a native Linux application.
