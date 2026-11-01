# TichuBot desktop source packages

The Windows, macOS, and Linux source distributions contain the visual **Play**
and **Advisor** application, an inference-only model checkpoint, and only the
source files needed at runtime. They are ordinary Python folders rather than
compiled applications.

Both packages require:

- Python 3.10 or newer
- internet access during the first launch
- a local web browser

The first launch creates a private `.tichubot-venv` directory inside the
extracted folder and installs NumPy and PyTorch there. It does not modify the
project's Python environment or install packages system-wide. Later launches
reuse that environment and normally work without internet access.

All three archives are assembled by the shared `source_bundle.py` builder and
run the shared `source_launcher.py`. The platform directories contain only
their small configuration wrapper, start script, dependency file, and docs.

## Windows

Use `dist/windows/TichuBot-Windows-Source.zip`.

1. Extract the ZIP.
2. Open the extracted `TichuBot-Windows-Source` folder.
3. Double-click **Start TichuBot.bat**.

If Python is missing, install Python 3 from
<https://www.python.org/downloads/windows/> and enable **Add python.exe to
PATH** during installation. The launcher installs CPU-only PyTorch, so CUDA
and an NVIDIA driver are not required.

The complete recipient guide is in
[`windows_source/README.md`](windows_source/README.md), and build instructions
are in [`windows_source/BUILDING.md`](windows_source/BUILDING.md).

The existing `TichuBot.exe` is a separate, fully self-contained Windows option
for recipients who should not install Python or download packages.

## macOS

Use `dist/macos/TichuBot-Mac.zip`.

1. Extract the ZIP.
2. Open the extracted `TichuBot-Mac` folder.
3. Double-click **Start TichuBot.command**.

If macOS blocks the first launch, right-click **Start TichuBot.command** and
choose **Open**. If its executable permission was lost, run this from Terminal:

```bash
chmod +x "Start TichuBot.command"
./"Start TichuBot.command"
```

Install Python from <https://www.python.org/downloads/macos/> if the launcher
reports that `python3` is missing.

The complete recipient guide is in [`macos/README.md`](macos/README.md), and
build instructions are in [`macos/BUILDING.md`](macos/BUILDING.md).

## Linux

Use `dist/linux/TichuBot-Linux.zip`.

Extract the ZIP, open a terminal in the extracted folder, and run:

```bash
./start-tichubot.sh
```

If necessary, restore its executable permission first:

```bash
chmod +x start-tichubot.sh
```

On Debian or Ubuntu, install `python3-venv` if Python cannot create the private
environment:

```bash
sudo apt install python3-venv
```

The complete recipient guide is in [`linux/README.md`](linux/README.md), and
build instructions are in [`linux/BUILDING.md`](linux/BUILDING.md).

## Using TichuBot

The launcher starts a localhost-only server and opens the mode menu in the
default browser. Select **Play** to play against the model or **Advisor** to
enter a real table and receive suggestions. Use the browser's **Exit** button
or press Control-C in the launcher terminal to stop the server.

No game data is sent to an external server. The application listens only on
the local computer.

## Resetting dependencies

Delete the `.tichubot-venv` directory inside the extracted package, then run
the launcher again. It will create a clean environment and reinstall the
dependencies.

## Rebuilding

Build all lightweight platform archives from one automatically selected
checkpoint (BEST_CHECKPOINT from `training/best_checkpoint.json`; only if
that file is missing locally, the checkpoint with the most games):

```bash
python3 distribution/build_lightweight.py
```

Use `--checkpoint PATH` to select a specific model. The command resolves the
checkpoint once, passes that exact file to every builder, and explicitly
leaves the full `TichuBot.exe` and `TichuBot-Windows.zip` artifacts untouched.
Before writing each source ZIP and SHA-256 file, it strips training-only
checkpoint state and smoke-tests model loading plus browser-game
initialization.
