# Learning Architecture Audit

Date: 2026-08-12

Status, checked 2026-11-02:

- **Nonlinear action keys: done** (2026-08-12). The query-key dot product was
  deliberately retained.
- **Phoenix action semantics: done** (August 2026). Ambiguous Phoenix plays
  are two legal actions told apart by a `phoenix_high` bit (58-wide action
  vector); see the expressivity section below.
- **Optimizer batching: done** (2026-08-12). `--optimizer-batch-size` steps
  Adam on fixed-size batches instead of once per rollout; the compact runs
  since 2026-08-13, including the one behind the released model, use 2,048
  transitions per step with a 512-transition memory microbatch. The value was chosen on measured runtime. The controlled
  learning comparison proposed below was never run as a clean A/B.
- **History-event omissions: still open.** The history tokens still lack the
  resolved Dragon recipient, fulfilled wish, resolved Phoenix role and
  first-eight/last-six deal provenance.

## Bottom line

The strongest overlooked bottleneck is the optimizer schedule used by the
history trainer. `batch_size=1024` is only a memory microbatch: gradients are
accumulated over the complete roughly 70,000-75,000-transition rollout and the
optimizer steps only once per epoch. With two epochs and 1,024 rounds per
rollout, the history policy normally receives only two Adam steps per 1,024
new self-play rounds.

This is radically less optimization per collected round than the embed
trainer, which steps Adam after every 1,024-transition minibatch. It can make a
larger history model and architecture changes appear sample-inefficient or
neutral even when the rollout data contains useful learning signal.

This is a strong diagnosis, not yet proof that more frequent updates improve
playing strength. It needs the controlled comparison below.

## Direct checkpoint evidence

Adam's saved per-parameter step counters were inspected in the current
checkpoints:

| Checkpoint | Self-play rounds | Main Adam steps |
| --- | ---: | ---: |
| `checkpoints/embed/latest.pt` | 700,032 | 98,666 |
| `kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt` | 1,748,992 | 3,736 |
| `kaggle_runs/2026-08-03_082524_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt` | 3,258,368 | 7,004 |
| `kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt` | 14,790,656 | 30,118 |

The mature standard history policy therefore saw about 21.1 times as many
rounds as the 700k embed champion, but only about 30.5% as many Adam updates.
The static-attention history model had received only 3.8% as many Adam updates
as the embed champion when it was evaluated.

The history update implementation confirms the cause:

- `training/ppo_history.py:298-299` zeros gradients once at the start of each
  epoch.
- `training/ppo_history.py:306-392` processes every microbatch and accumulates
  its proportionally scaled gradient.
- `training/ppo_history.py:419-422` clips and steps once after the entire
  rollout has been consumed.

By contrast, `training/ppo_embed.py:97-100` zeros, backpropagates, clips, and
steps for every minibatch.

The production history rollout contains roughly 73,000 transitions per 1,024
rounds. With two epochs, the history trainer uses two Adam steps where ordinary
1,024-transition minibatch PPO would use roughly 140-144 steps. This is about a
70-fold difference in optimizer steps per self-play round.

The latest logged history updates also move only slightly:

- Mature standard final row: approximate KL `0.00026`, clip fraction `0.00225`.
- Static-attention compact final row: approximate KL `0.00135`, clip fraction
  `0.01371`.

Because no parameters change during the first accumulated epoch, its PPO ratio
is exactly one and its KL/clipping contribution is approximately zero. Only the
second epoch measures the effect of one preceding optimizer step. Even allowing
for that averaging, the 20% PPO clip is rarely active.

Full-rollout accumulation does produce an unbiased average gradient and avoids
the ordering effect of stepping through sorted homogeneous buckets. However,
one Adam update on an averaged gradient is not equivalent to dozens of
minibatch Adam updates. The nonlinear Adam state evolution, data reuse, policy
movement, and PPO clipping behavior are all different. In the present code,
`batch_size` controls memory and padding, not the effective optimization batch.

## Controlled next experiment

Change only the optimizer batching semantics while keeping the existing model,
observations, rollout size, PPO coefficients, passing/GT policies, and
synchronous collection fixed.

Recommended comparison from the same source checkpoint:

1. Control: current full-rollout accumulation, one Adam step per epoch.
2. Variant: randomized stratified macro-batches assembled from the existing
   history/legal-length buckets, initially targeting an effective optimizer
   batch of 4,096-8,192 transitions. Step after each macro-batch while retaining
   the padding savings.

Record:

- exact Adam steps and transitions per step;
- KL and clip fraction per epoch, not only their combined average;
- gradient norms for static trunk, history encoder, actor readout, and critic;
- balanced-seat score against the starting checkpoint and fixed external
  checkpoints;
- equal-round and equal-wall-clock results;
- exact checkpoint paths, commands, evaluation rounds, and uncertainty.

Checkpoint frequently during the first short leg. More frequent updates could
move too far; use measured KL and strength rather than assuming the embed
trainer's update rate is automatically optimal for the history model.

## Original model expressivity ceiling (now addressed in the scorer)

At audit time, the play-action decoder was strictly additive over cards. In
both
`training/model_embed.py` and `training/model_history.py` it computes:

```text
query = query_head(state)
key(action) = Linear(action_card_mask)
logit(state, action) = query dot key(action)
                     = sum(card in action) utility(state, card)
```

Consequences:

- a pair's logit is exactly the sum of its two card utilities;
- straights, steps, full houses, and bombs have no nonlinear interaction term;
- the model cannot value keeping cards together independently of their
  individual utilities;
- increasing the Transformer or embedding width does not remove this algebraic
  restriction; and
- the Tichu-call bit contributes the same state-dependent offset to every
  possible first play, so the call decision factorizes from the selected play.

Both policies now use a two-hidden-layer nonlinear residual action encoder.
Its output projection is zero-initialized, so pre-upgrade checkpoints begin
with exactly unchanged logits. The new key removes the additive scoring ceiling
while keeping the original query-key dot product.

The learned action space now also represents the only two physically ambiguous
Phoenix cases. Embed/history policies receive a 58-wide action vector: 56 card
bits, the Tichu-call bit, and one `phoenix_high` bit. An ambiguous straight or
full house is exposed as two semantic legal actions over the same physical
cards, ordered lower then higher. Each interpretation is checked separately
when following, and the selected interpretation is passed through
`GameState.play()`. Human and fixed-table action ordering remains unchanged.
Legacy 57-wide checkpoint weights and Adam moments are zero-padded, so the two
interpretations start tied and become distinguishable through further
training.

## History compression finding

Compression itself is not inherently a failure. A policy ultimately needs a
finite decision statistic. The current design compresses at the wrong time and
from incomplete input:

- play history is pooled into one vector before candidate actions are scored;
- static attention conditions that pooling on the current state, but the same
  pooled history is still shared by every legal action;
- the history query therefore cannot retrieve evidence specific to a candidate
  pair, straight, bomb, or other play; and
- the recorded 63-wide tokens omit observed information including the resolved
  Dragon trick recipient, fulfilled Mahjong wish, resolved Phoenix role, and
  private first-eight versus later-six deal provenance.

The Phoenix action bit fixes choosing an interpretation at the moment of play;
it does not add that resolved interpretation to later history tokens after the
trick has cleared.

Dragon resolution itself is implemented, including a trained specialist head
that chooses the recipient. The omission is in subsequent observations: the
recipient is stored in `trick_history` and `won_cards`, but the standard
196-wide static observation and 63-wide play tokens do not expose that resolved
gift after the pending state clears. This creates a hard observation alias for
later decisions. The separate counted-trick-points observation partially
addresses ownership by exposing each seat's current captured-point total.

A later memory architecture should retain a complete typed event stream and
let the actor's candidate queries and critic query attend to the relevant
prefix. That postpones task-specific compression until the question is known.
It should be tested after optimizer batching and nonlinear action valuation are
isolated, rather than changing all three variables in one long run.

## Secondary concern

Actor and critic remain substantially shared. The compact split architecture
has separate pooling queries and fusion layers, but still shares the static
trunk and Transformer token encoder. Existing critic probes suggest that a
dedicated critic can learn some bookkeeping more easily, but optimizer
intensity and stationary-policy differences must be controlled before
attributing the whole gain to representation sharing.

## Validation performed during this audit

```text
295 tests passed in 17.96 seconds
git diff --check passed
```

No training run was changed or stopped. The later action-key implementation
modified model, configuration, checkpoint compatibility, and training setup
source files; optimizer batching itself was not changed.
