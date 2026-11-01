# TichuBot for Windows (Python package)

This folder contains the visual Play and Advisor interface, the trained
inference model, and only the Python source files needed to run them. Unlike
the standalone `TichuBot.exe` distribution, this package installs its Python
dependencies on the recipient's computer during the first launch.

## Start

1. Extract the ZIP before running anything in it.
2. Open the extracted `TichuBot-Windows-Source` folder.
3. Double-click **Start TichuBot.bat**.

The first launch checks for Python 3.10 or newer, creates a private environment
inside this folder, and installs NumPy plus CPU-only PyTorch if necessary. It
therefore needs internet access and can take a few minutes. Later launches
reuse the installed packages.

TichuBot opens its Play/Advisor menu in the default browser. Its server listens
only on the local computer. Use the **Exit** button in the browser or press
Control-C in the Command Prompt window to stop it.

If a browser does not open automatically, use the `http://127.0.0.1:.../`
address printed in the window.

## If Python is missing

Install a current 64-bit Python 3 release from:

<https://www.python.org/downloads/windows/>

During installation, enable **Add python.exe to PATH** or install the standard
Python Launcher. Then double-click **Start TichuBot.bat** again.

## Windows security prompt

Windows may warn about a batch file downloaded from the internet. Inspect the
plain-text launcher if desired, then use **More info** and **Run anyway** if
Windows presents that option and you trust where the ZIP came from.

## Troubleshooting

To redo dependency setup, delete the hidden `.tichubot-venv` folder inside the
extracted package and launch again.

Startup errors are also written to:

```text
%LOCALAPPDATA%\TichuBot\error.log
```

The model always runs on the CPU; no NVIDIA driver or CUDA installation is
required.
