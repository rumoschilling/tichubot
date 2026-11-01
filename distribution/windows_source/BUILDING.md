# Building the Windows source ZIP

This is the dependency-installing alternative to the self-contained Windows
executable. It packages ordinary Python source and an inference-only model;
the recipient's Windows installation supplies Python and downloads the two
runtime packages on first launch.

Build with the highest `games_played` checkpoint found locally:

```bash
python \
  distribution/windows_source/build_windows_source.py
```

Or select one explicitly:

```bash
python \
  distribution/windows_source/build_windows_source.py \
  --checkpoint path/to/latest.pt
```

The builder loads and smoke-tests the staged source and model before writing:

```text
dist/windows/TichuBot-Windows-Source.zip
dist/windows/TichuBot-Windows-Source.zip.sha256
```

The original standalone executable builder remains at
`distribution/windows/build_windows.py` and is not changed by this workflow.
