# TichuBot for macOS

This folder is a source-based TichuBot distribution. It contains the visual
Play and Advisor interface, the trained inference model, and only the Python
source files needed to run them. Python packages are installed locally on the
recipient's Mac instead of being bundled into a platform-specific executable.

## Start

1. Extract the ZIP before running anything in it.
2. Double-click **Start TichuBot.command**.
3. If macOS asks for confirmation, right-click the file and choose **Open**.

The first launch checks for Python 3.10 or newer, creates a private environment
inside this folder, and installs NumPy and PyTorch if they are missing. That
first launch therefore needs internet access and can take a few minutes. Later
launches reuse the installed packages.

TichuBot opens its Play/Advisor menu in the default browser. The server listens
only on this Mac. Close it with the **Exit** button in the browser or press
Control-C in the Terminal window.

## If Python is missing

Install a current Python 3 release from:

<https://www.python.org/downloads/macos/>

Then double-click the launcher again.

## Troubleshooting

If Finder will not execute the launcher, open Terminal, drag this folder into
the Terminal window after typing `cd `, press Return, and run:

```bash
chmod +x "Start TichuBot.command"
./"Start TichuBot.command"
```

To redo dependency setup, delete the hidden `.tichubot-venv` folder and launch
again. Startup errors are also written to:

```text
~/Library/Logs/TichuBot/error.log
```

The model always runs on the CPU, so this package works independently of
Apple's optional GPU acceleration support.
