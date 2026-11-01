# Building the macOS source ZIP

Unlike the Windows release, this package does not use PyInstaller and does not
contain a Python runtime. It copies only the browser application's runtime
source, strips training state from one model checkpoint, and adds a macOS
launcher that prepares dependencies on the recipient's machine.

Requirements on the build machine:

- the project's local Python environment with PyTorch
- a checkpoint containing the play, passing, and Grand Tichu models

Build with the highest `games_played` checkpoint found locally:

```bash
python \
  distribution/macos/build_macos.py
```

Or select one explicitly:

```bash
python \
  distribution/macos/build_macos.py --checkpoint path/to/latest.pt
```

The builder loads and smoke-tests the staged source and model before writing:

```text
dist/macos/TichuBot-Mac.zip
dist/macos/TichuBot-Mac.zip.sha256
```

Extract the ZIP and run `Start TichuBot.command` on a Mac for the final test.
The ZIP is architecture-neutral source; the launcher's private virtual
environment installs packages suitable for the recipient's own Python and Mac.
