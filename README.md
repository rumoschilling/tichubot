# TichuBot

A self-play reinforcement-learning AI for the four-player card game
**Tichu**, with a browser table to play against it and an advisor mode for
real games. Everything runs locally; nothing is sent anywhere.

- **Play**: you take one seat; the bot plays your partner and both opponents.
- **Advisor**: enter the cards at a real table and see what the bot would do.
- **Train**: the full training code, including the recipe behind the
  released model (PPO self-play with Magnetic Mirror Descent).

## Quick start: play against the bot

You need Python 3.10 or newer.

```bash
git clone https://github.com/rumoschilling/tichubot.git
cd tichubot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Download the trained model, `tichubot-best.pt`, from the
[latest release](https://github.com/rumoschilling/tichubot/releases/latest)
and save it as `checkpoints/tichubot-best.pt`:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/tichubot-best.pt \
  https://github.com/rumoschilling/tichubot/releases/latest/download/tichubot-best.pt
```

Then start the table:

```bash
python -m training.web --menu           # choose Play or Advisor in the browser
python -m training.web --mode play      # straight into a game
python -m training.web --mode advisor   # advise a real table
```

It opens `http://127.0.0.1:8765/`. The server only listens on your own
machine. `--random-ai` runs the table without a model, and `--no-browser`
skips opening the browser. A terminal version is `python -m training.play`.
CPU is enough to play; no GPU needed.

### Advisor card input

The advisor starts with your first eight cards, so it can advise on Grand
Tichu before the remaining six and the passing are entered. Card fields
accept names and short macros:

```text
m d w p              Mahjong, Dragon, Dog, Phoenix
2 rgb                2r 2g 2b
2 2 3 3 r g s b      2r 2g 3s 3b
2-7 r g b s g s      2r 3g 4b 5s 6g 7s
J bomb               Jr Jg Jb Js
```

### Packaged versions for non-programmers

`distribution/` builds self-contained source ZIPs for Windows, macOS and
Linux that install their own dependencies on first launch; see
[`distribution/README.md`](distribution/README.md).

## Train your own model

```bash
scripts/train_mmd.sh                          # writes to runs/mmd
OUT=runs/test GAMES=200000 scripts/train_mmd.sh   # a short try
```

This is the recipe behind the released model: a compact history model
(an MLP over the current situation plus a small transformer over the
round's plays), trained from scratch by PPO self-play, with Magnetic Mirror
Descent replacing a hand-tuned entropy schedule. Separate small networks
learn passing, Grand Tichu calls, the Dragon gift and the Mahjong wish.

It needs a CUDA GPU with about 9 GB free (`DEVICE=cpu` works, slowly).
The released model was trained for 36.6 million games, about a week on one
RTX 3080; it is still improving at that point. The script resumes from its
last checkpoint after a crash. All options: `python -m training.train_history --help`.

Compare two checkpoints by playing them against each other:

```bash
python -m training.eval runs/mmd/checkpoints/latest.pt checkpoints/tichubot-best.pt --games 2000
```

## Repository layout

| Path | Contents |
| --- | --- |
| `tichu_env/` | Game engine: rules, legal moves, scoring, observation encoding |
| `training/` | Models, PPO, rollouts, evaluation, terminal and browser play |
| `training/web_static/` | Browser UI (plain HTML/CSS/JS, no framework) |
| `scripts/` | Training launcher |
| `distribution/` | Builders for the desktop source packages |
| `tests/` | Test suite: `pip install -e '.[dev]' && pytest -q` |

## Background reading

- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md): the full development log,
  every experiment and result, mistakes and corrections included.
- [`TRAINING_EFFICIENCY_WRITEUP.md`](TRAINING_EFFICIENCY_WRITEUP.md): how
  rollout and PPO throughput were made fast enough.
- [`LEARNING_ARCHITECTURE_AUDIT.md`](LEARNING_ARCHITECTURE_AUDIT.md): what
  the network can and cannot represent.
- [`PURE_PPO_ARCHITECTURE_STUDY.md`](PURE_PPO_ARCHITECTURE_STUDY.md): design
  study for a causal event-memory model.
- [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md): a rules and code audit, with
  every finding fixed.

The card faces and card back were drawn for this project. Tichu is a game
by Urs Hostettler; this project is not affiliated with its publishers.

## License

[CC BY-NC 4.0](LICENSE): you may use, share and adapt the code and the
model for any **noncommercial** purpose, with credit to the author. Selling
it or using it commercially is not allowed without permission.
