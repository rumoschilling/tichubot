# Building the Linux source ZIP

This distribution packages the ordinary Python runtime source and an
inference-only checkpoint. It does not use PyInstaller or bundle a Python
runtime. Dependencies are prepared on the recipient's Linux system.

Build with the highest `games_played` checkpoint found locally:

```bash
python \
  distribution/linux/build_linux.py
```

Or select one explicitly:

```bash
python \
  distribution/linux/build_linux.py --checkpoint path/to/latest.pt
```

The builder loads and smoke-tests the staged source and model before writing:

```text
dist/linux/TichuBot-Linux.zip
dist/linux/TichuBot-Linux.zip.sha256
```

The ZIP is architecture-neutral source. Its launcher installs a CPU-only
PyTorch wheel suitable for the recipient's Python and Linux architecture.
