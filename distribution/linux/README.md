# TichuBot for Linux

This folder contains the visual Play and Advisor interface, the trained
inference model, and only the Python source files needed to run them. It does
not contain the training environment or a bundled Python runtime.

## Start

Extract the ZIP, open a terminal in the extracted `TichuBot-Linux` folder, and
run:

```bash
./start-tichubot.sh
```

If the ZIP tool did not retain executable permissions, run this once first:

```bash
chmod +x start-tichubot.sh
```

The first launch checks for Python 3.10 or newer, creates a private environment
inside this folder, and installs NumPy plus CPU-only PyTorch if necessary. It
therefore needs internet access and may take a few minutes. Later launches
reuse the installed packages.

TichuBot opens its Play/Advisor menu in the default browser. Its server listens
only on the local computer. Use the **Exit** button in the browser or press
Control-C in the terminal to stop it.

If a browser does not open automatically, use the `http://127.0.0.1:.../`
address printed in the terminal.

## Troubleshooting

On Debian or Ubuntu, virtual-environment support may need to be installed
separately:

```bash
sudo apt install python3-venv
```

Use the equivalent package from your distribution when necessary. To redo the
dependency setup, remove the hidden `.tichubot-venv` folder and run the
launcher again.

Startup errors are also written to:

```text
~/.local/state/TichuBot/error.log
```

If `XDG_STATE_HOME` is configured, that directory is used instead of
`~/.local/state`.
