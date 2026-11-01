# Research log

This is the working log kept while TichuBot was developed, published as it
was written: experiments, dead ends, bugs found later, and corrections are
all left in, newest sections at the end. It was written for the people (and
coding agents) working on the project, so it is terse and full of internal
references. Paths such as `ring_runs/`, `kaggle_runs/`, `local_evaluations/`
and checkpoint files refer to data that is not part of this repository.
"training-server" is the GPU machine most later runs used, and
`<kaggle-user>` the Kaggle account of the earlier runs.

---

# Handoff Memory

Updated: 2026-08-12 (Europe/Berlin)

## Nonlinear action key encoder (implemented 2026-08-12)

Both `TichuPolicyEmbed` and `TichuPolicyHistory` now retain the legacy linear
action key as a residual base, add one trainable nonlinear key path, and keep
the original query-key dot product:

```text
action key = Linear(action) + MLP(58 -> 128 -> 128 -> embed_dim)
logit = dot(query, action key)
```

The new output projection is zero-initialized. Pre-upgrade checkpoints
therefore reproduce their old logits exactly when first loaded, while that
projection receives action-dependent gradients immediately. The width defaults
to `action_hidden_dim=128`, with CLI override `--action-hidden`. This removes
the strict per-card additivity of the old key while deliberately retaining the
query-key factorization.

The learned play action is now 58-wide: 56 physical-card bits, the Tichu-call
bit, and one `phoenix_high` bit. `GameState.get_policy_legal_actions()` exposes
ambiguous Phoenix straights and full houses twice over the same cards, lower
then higher, and filters the two meanings separately against the current
combination. The selected bit is resolved to `phoenix_as` by `GameState.play()`.
Other Phoenix combinations have a forced interpretation and keep the bit zero.
The existing two-field `get_legal_actions()` API and its ordering remain in use
for humans and fixed-table policies.

Checkpoint loading supplies behavior-neutral values for absent encoder keys,
including under `strict=True`. History/embed initialization copies the new
encoder when the source contains it. Resume migration appends the encoder's 6
parameter tensors to the existing play optimizer group without moving legacy
Adam moments when the legacy group layout is otherwise compatible. It also
zero-pads legacy action weights and two-dimensional Adam moments from 57 to 58
inputs. Thus an upgraded checkpoint initially assigns exactly equal logits to
the lower/higher Phoenix variants, but can learn the distinction on resume.

Compatibility was checked against the mature standard checkpoint:

```text
checkpoint: kaggle_runs/2026-08-01_110222_standard/
            tichubot_outputs/checkpoints/history_standard_gpu/latest.pt
games_played: 14,790,656
load mode: strict model load plus history training optimizer restore
result: old key outputs remained zero-residual; legacy Adam state restored;
        action weights and Adam moments were padded to 58 inputs with a zero
        Phoenix column; 32,192 new encoder parameters had fresh/lazy state
```

No training or strength evaluation was run. Validation commands/results:

```bash
python -m py_compile \
  tichu_env/combinations.py tichu_env/game_state.py \
  training/model_embed.py training/model_history.py training/checkpoints.py \
  training/rollout_embed.py training/rollout_history.py training/play.py \
  training/arena.py training/eval.py training/advisor.py training/web.py \
  training/train_embed.py training/train_history.py training/config.py

python -m pytest -q
```

```text
303 passed in 19.62s
git diff --check passed
```

## Critic capacity and target experiments (2026-08-10)

All probes froze the counted-static actor, passing model, and GT model from
`checkpoints/embed_counted_points_local_20260806/latest.pt` (2,707,968 play
rounds). The probe critic reads the complete 200-wide counted-static input
directly; no actor parameter receives gradients. Error below is in game points.

The live PPO implementation has no GAE or TD bootstrap. Every transition in a
round receives the realized team-relative final return `G`, the actor uses the
normalized Monte Carlo advantage `G - V_old(s)` in the clipped PPO objective,
and the critic minimizes `(V(s) - G)^2`. The combined main loss is policy loss
plus `value_coef * value_loss` minus `entropy_coef * entropy`.

### Wider all-state Monte Carlo versus TD(lambda=0.8)

Both all-state probes used a separate two-layer 512-wide Tanh critic (366,081
parameters), two epochs per 1,024 newly collected rounds, and trained through
1,000,000 rounds. The final resume commands were:

```bash
python -u -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 1000000 --validation-games 4096 --games-per-update 1024 \
  --eval-every 20480 --checkpoint-every 20480 --batch-size 4096 \
  --epochs 2 --train-cohort all --hidden 512 --layers 2 \
  --critics tanh --workers 4 \
  --resume checkpoints/critic_capacity_512x2/latest.pt \
  --output-dir checkpoints/critic_capacity_512x2

python -u -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 1000000 --validation-games 4096 --games-per-update 1024 \
  --eval-every 20480 --checkpoint-every 20480 --batch-size 4096 \
  --epochs 2 --train-cohort all --hidden 512 --layers 2 \
  --critics tanh --workers 4 --td-lambda 0.8 \
  --resume checkpoints/critic_capacity_512x2_lambda08/latest.pt \
  --output-dir checkpoints/critic_capacity_512x2_lambda08
```

The two runs used separate fixed validation deals, so compare each trained
critic with the original critic on its own holdout:

```text
cohort              MC original -> MC       lambda original -> lambda
all states              116.0 -> 114.2             115.7 -> 117.8
terminal                 26.3 ->  22.0              26.2 ->  18.0
t-5                      51.3 ->  48.0              50.7 ->  43.5
t-10                     63.7 ->  60.8              65.5 ->  64.1
t-20                     98.8 ->  97.6              97.8 ->  98.7
forced terminal          23.7 ->  18.9              23.6 ->  14.0
forced double            23.5 ->  19.0              23.6 ->  17.2
```

Monte Carlo continued improving materially between 500k and 1M and produced a
small broad gain. Lambda propagated a much stronger signal through roughly the
last five decisions, barely helped t-10, and did not improve t-20 or aggregate
MAE. Its late-state metrics were noisy: at 991,520 rounds it briefly reached
15.3 terminal and 11.7 forced-terminal MAE before ending at 18.0 and 14.0.
This is an equal-training-time target tradeoff, not evidence that lambda has
globally solved value learning.

Artifacts:

```text
checkpoints/critic_capacity_512x2/latest.pt
checkpoints/critic_capacity_512x2_lambda08/latest.pt
```

The earlier 2x512 Monte Carlo result at 500k did not beat the 2x256 all-state
Tanh critic: overall MAE was 117.9 versus 115.9 and forced-terminal MAE was
23.5 versus 23.7. Width by itself was therefore not the bottleneck.

### Fixed-position specialist

Do not use a cohort selected as `steps_to_end == 10` to infer performance at a
normal information state. That selector depends on the round's future length.
A partial 154,624-round artifact exists at
`checkpoints/critic_step10_probe/latest.pt`, but the experiment was stopped and
discarded for this reason.

The validation rollouts average 74.21 main-play decisions, placing their t-10
state at zero-based position 64.21. The replacement cohort therefore selects
the 65th decision from round start (`start_65`), using an explicit 1-based
`steps_from_start` rollout field. Rounds ending before that point contribute no
training or validation state. Command:

```bash
python -u -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 500000 --validation-games 8192 --games-per-update 1024 \
  --eval-every 51200 --checkpoint-every 51200 --batch-size 1024 \
  --epochs 10 --train-cohort start_65 --hidden 256 --layers 2 \
  --critics tanh --workers 4 \
  --output-dir checkpoints/critic_start65_probe
```

Final fixed holdout (`6,667` eligible states out of `8,192` rounds):

```text
                    original       start-65 specialist
bias                    -2.5              -0.1
MAE                     74.7              85.3
RMSE                   105.1             116.9
```

The specialist learned the cohort mean but did not match the original critic's
individual predictions. Unlike forced terminal states, start-65 states retain
large hidden-hand and future-policy variance. The probe also saw only about
410k unique eligible states, while the original actor/critic representation
was trained on every transition over 2.7M rounds. This result rejects the
narrow claim that 500k rounds of focused 2x256 training immediately expose a
better valid mid/late-round solution; it does not prove a more capable history
critic cannot do better.

Artifact: `checkpoints/critic_start65_probe/latest.pt`.

### History architecture/input audit

The compact trained history model still projects each 63-wide play/pass token
to width 96, applies two shared four-head Transformer layers (FFN 192), then
uses separate constant actor and critic attention-pooling queries. Each pooled
96-vector is concatenated with the 128-wide static MLP feature, projected back
to 128, and added residually. Static state therefore conditions the model only
after history has been compressed. The proposed architecture in
`PURE_PPO_ARCHITECTURE_STUDY.md` instead puts current decision context into
actor-candidate and critic queries that cross-attend the complete causal event
prefix.

The report's observation omissions are not implemented yet. The environment
resolves the events correctly, but durable history tokens still omit the
Mahjong wish choice after fulfillment, Dragon trick recipient, resolved
Phoenix role, and first-eight versus later-six private deal provenance. The
typed observer-specific event stream remains the next architecture task.

### Static-conditioned history attention (prepared 2026-08-11)

The new compact-scratch architecture enables `--static-attention`. Actor and
critic each retain their own learned constant pooling query, add an independent
linear projection of the current 128-wide static feature, and use that result
to cross-attend the shared encoded event sequence. Static state now determines
which history events are pooled before fusion; the Transformer token encoder
itself remains unchanged. The actor and critic projections are separate, like
their pooling attention and fusion paths.

This is opt-in (`TrainConfig.static_attention=False`) so existing checkpoints
reconstruct with their original constant queries. The compact Kaggle profile
enables it and is intended to start a new lineage from scratch. Prepare with:

```bash
python kaggle/prepare_compact_scratch_bundle.py
```

Kaggle accepted the fresh static-attention lineage as compact-scratch kernel
version 5 at 2026-08-11 23:10:47 CEST:

```text
kernel:  <kaggle-user>/tichu-compact-scratch
run:     kaggle_runs/2026-08-11_231044_compact-scratch
device:  NvidiaTeslaT4
limit:   11.7 hours
resume:  none (fresh initialization)
```

The command includes `--split-actor-critic-history --static-attention` and does
not enable heteroscedastic value training. Fetch after completion with:

```bash
bash kaggle/run.sh --fetch compact-scratch
```

### Heteroscedastic critic probe (started 2026-08-11)

This is deliberately separate from the static-attention Kaggle lineage. It
freezes the counted-static actor and compares an ordinary two-layer Tanh critic
against a critic with the identical trunk and value-mean initialization plus a
state-dependent variance head. The ordinary critic trains with MSE; the
heteroscedastic critic trains with Gaussian NLL. Both see the same Monte Carlo
targets and minibatches. PPO and the source actor are unchanged.

```bash
python -u -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 500000 --validation-games 8192 --games-per-update 1024 \
  --eval-every 51200 --checkpoint-every 51200 --batch-size 1024 \
  --epochs 10 --train-cohort all --hidden 256 --layers 2 \
  --critics tanh tanh_hetero --workers 4 \
  --output-dir checkpoints/critic_heteroscedastic_mc_20260811
```

Console log: `logs/critic_heteroscedastic_mc_20260811.console.log`.
Compare mean bias/MAE/RMSE as usual. For `tanh_hetero`, also inspect predicted
sigma in points, one-sigma coverage (a calibrated Gaussian is about 68%),
correlation between predicted sigma and absolute error, and Gaussian NLL. Do
not interpret intermediate checkpoints while the run is healthy.

Interim state when the 2026-08-11 session ended: the persistent user service
`tichu-critic-hetero-20260811.service` was active at 377,856/500,000 rounds.
The latest completed fixed-holdout validation was at 358,400 rounds:

```text
cohort           original MAE   tanh MSE MAE   hetero MAE/RMSE
all                    117.2          125.6       117.0 / 158.6
terminal                26.8           29.4         9.9 / 17.9
step_5                  50.7           51.4        41.3 / 60.0
step_10                 64.7           68.5        58.8 / 87.8
step_20                 99.0          105.3       99.5 / 136.2
start_65                75.9           80.0       71.0 / 106.5
forced_terminal         23.8           26.5         6.1 / 8.5
forced_double           24.5           33.7         4.7 / 7.4
```

The heteroscedastic mean was consistently better than the identically
initialized MSE critic and had narrowly reached the mature original critic's
overall MAE, although overall RMSE remained worse than original (158.6 versus
154.4). This is still interim; use the final 500k report tomorrow.

The first static-attention Kaggle leg intentionally does **not** use this loss.
A later resume can add it without changing initial actor logits or the existing
value mean, provided the migration is implemented carefully:

1. Append the variance head after all existing modules so optimizer parameter
   ordering for old weights stays unchanged.
2. Initialize it to variance 1 and use twice the conventional Gaussian NLL,
   `error^2 * exp(-logvar) + logvar`, so the initial mean gradient exactly
   matches this project's current MSE gradient.
3. Add the new parameter ids to the correct saved Adam group instead of
   dropping all optimizer moments on the group-size mismatch.
4. Keep `R - V_mean` unchanged and never uncertainty-weight the actor loss.
5. Track ordinary holdout mean MAE/RMSE and calibration, because NLL alone can
   hide systematic mean error by inflating variance.

The remaining pathological risks are finite-capacity variance inflation
(treating learnable representation error as irreducible noise), variance
collapse on near-deterministic states, and PPO nonstationarity. Bound variance
numerically when integrating it. The current fixed-holdout mean improvements
argue that the probe is learning useful structure rather than merely hiding
error.

## 2026-08-12 completed evaluations

### Static-attention compact scratch v5

Fetched completed output to:

```text
kaggle_runs/2026-08-11_231044_compact-scratch/
final checkpoint games: 1,748,992
```

Standard 8,000-round drill:

```bash
python -m training.leg_eval \
  --run kaggle_runs/2026-08-11_231044_compact-scratch \
  --prev kaggle_runs/2026-08-03_082524_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --games 8000
```

Results:

```text
vs old compact at 3,258,368 rounds: -14.4 points/round, 43.5% vs 52.7% wins
vs embed-700k baseline:             +27.4 points/round, 57.3% vs 39.2% wins
GT mean call probability:             7.0%, strength correlation +0.554
GT on vs off, same weights:           +8.3 points/round
GT declarations:                      13.8%, made 65.2%
```

Because the old compact opponent had almost twice the play-training budget,
the fair architecture check used the old 1,728,512-round checkpoint:

```bash
python -m training.eval \
  kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  kaggle_runs/2026-08-01_223035_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --games 8000 --batch 64 --device cpu
```

Static attention scored `+2.3` points/round with 49.4% versus 46.6% wins and
4.0% draws. At the established roughly +/-3.5 points/round sampling error,
this is neutral: static-conditioned pooling did not demonstrate a playing-
strength improvement at equal round budget, but neither did it regress.

#### Static-attention usage and critic accuracy

The neutral arena result does **not** mean the new branch is dead. An
on-policy empirical-mean ablation was used instead of zeroing the query (zero
is far outside the query distribution). Across 36,498 static-attention
self-play decisions, the actor's projected query had mean norm `3.803` and
mean per-state norm `5.787`, versus only `0.212` for the constant query. The
control added the empirical mean projected query to the constant query, zeroed
only `history_static_query.weight`, and left the critic query and every other
weight unchanged. In 8,000 rounds the intact checkpoint beat that control by
`+41.9` points/round (62.8% versus 33.0% wins, 4.1% draws): the trained actor
depends strongly on decision-conditioned pooling after co-adaptation.

The repeatable diagnostic uses the same empirical-mean control for the critic
and reports attention statistics:

```bash
python \
  -m training.static_attention_diagnostics \
  kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --games 512 --max-attention-states 4096 --batch-size 256 --device cpu
```

It collected 36,705 decisions. Dynamic-query empirical-mean norms were
`3.875` for the actor and `4.753` for the critic. Attention statistics over
4,096 sampled non-empty histories were:

```text
branch   entropy  normalized entropy  max weight  KL(dynamic || mean)
actor      2.399              0.728       0.291                 0.856
critic     2.855              0.869       0.174                 0.124
```

Thus the actor query substantially changes and sharpens retrieval. The critic
query is more diffuse and closer to its average, but is still useful. On the
exact same 512 completed rounds, replacing only the dynamic critic query with
its empirical mean changed overall MAE from `78.9` to `83.2` points and RMSE
from `98.7` to `102.7`; at T-1, MAE changed from `44.6` to `53.8`.

The fair independent-checkpoint critic comparison, with each checkpoint in
self-play and its own bundled Grand Tichu head enabled, was:

```bash
python -m training.critic_value_eval \
  kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  kaggle_runs/2026-08-01_223035_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --label-a static-attention --label-b old-compact-equal \
  --games 4000 --batch 64 --device cpu --self-play --gt
```

```text
steps       static MAE/RMSE     old compact MAE/RMSE
1                 42.2 / 52.4            40.0 / 50.2
2                 51.4 / 63.8            51.6 / 62.7
3                 55.0 / 68.7            55.0 / 68.5
4-5               55.6 / 70.3            56.1 / 70.2
6-10              65.2 / 84.3            66.6 / 84.3
11-20            80.6 / 107.3           83.1 / 109.3
21-40           102.5 / 136.2          103.3 / 136.7
41-80           118.0 / 155.0          119.6 / 158.0
81+              99.1 / 135.5          113.8 / 150.8
overall          99.9 / 134.5          101.3 / 136.4
```

Overall critic accuracy improved slightly and mostly earlier in rounds, while
terminal accuracy regressed slightly. A no-GT 4,000-round control had almost
identical overall MAE (`77.3` static versus `77.6` old), with static again
worse at T-1 (`40.7` versus `39.2`). The different absolute scale is expected:
enabled declarations make realized scores more variable.

Conclusion: static-conditioned attention is active and useful inside the
trained checkpoint, but it did not add much *independent* playing strength at
equal training budget. The old model can learn a comparable policy through
constant history pooling followed by static/history fusion, while both models
still compress the complete prefix into one 96-wide vector. The query is also
shared across all legal actions, so it cannot retrieve action-specific history
evidence, and it cannot recover events omitted from the history tokens. This
looks like co-adaptation around a mostly redundant read mechanism rather than
a frozen branch or attention saturation.

### Heteroscedastic critic final and fresh validation

The local probe completed 500,000 rounds successfully. A new 8,192-round
holdout, collected after training rather than reused from the training-time
fixed validation set, reproduced the result:

```bash
python -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 500000 --validation-games 8192 --games-per-update 1024 \
  --eval-every 51200 --checkpoint-every 51200 --batch-size 1024 \
  --epochs 10 --train-cohort all --hidden 256 --layers 2 \
  --critics tanh tanh_hetero --workers 4 \
  --resume checkpoints/critic_heteroscedastic_mc_20260811/latest.pt \
  --output-dir checkpoints/critic_heteroscedastic_mc_20260812_fresh_eval
```

Fresh-holdout value-mean errors in points:

```text
cohort           mature original MAE   matched MSE MAE   hetero MAE/RMSE
all                            116.6             126.5       116.2 / 158.9
terminal                        26.1              31.2         7.9 / 16.1
step_5                          50.9              51.0        41.1 / 60.3
step_10                         64.0              69.4        56.8 / 83.9
step_20                         98.7             107.8       97.5 / 133.7
start_65                        75.5              81.3       69.0 / 104.7
forced_terminal                 23.6              28.1         4.6 / 6.6
forced_double                   24.0              32.4         3.0 / 4.5
```

Overall heteroscedastic mean MAE narrowly beat the mature source critic, but
RMSE remained worse (`158.9` versus `154.2`), indicating rarer large errors.
It decisively beat the identically initialized MSE critic everywhere measured.
Uncertainty correlated `+0.510` with absolute error overall and `+0.628` at
t-10. Overall one-sigma coverage was only 51.8%, so the variance head is
under-dispersed even though its ranking signal is useful. Artifact:
`checkpoints/critic_heteroscedastic_mc_20260812_fresh_eval/latest.pt`.

### Heteroscedastic critic extension to 1M (started 2026-08-12)

To test whether the equal-budget NLL advantage persists or whether MSE catches
up, both critics resumed from the exact 500k checkpoint with their optimizer
states and continue on identical new rollouts to 1M. The source actor remains
frozen, so this measures convergence/sample efficiency but not the additional
nonstationarity of an actor changing during PPO.

```bash
python -u -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 1000000 --validation-games 8192 --games-per-update 1024 \
  --eval-every 51200 --checkpoint-every 51200 --batch-size 1024 \
  --epochs 10 --train-cohort all --hidden 256 --layers 2 \
  --critics tanh tanh_hetero --workers 4 \
  --resume checkpoints/critic_heteroscedastic_mc_20260811/latest.pt \
  --output-dir checkpoints/critic_heteroscedastic_mc_1m_20260812
```

Persistent service: `tichu-critic-hetero-1m-20260812.service`.
Console: `logs/critic_heteroscedastic_mc_1m_20260812.console.log`.
Do not interpret intermediate checkpoints while the run is healthy.

Probe support added in `training/critic_probe.py` and
`training/rollout_embed.py` includes selectable critic variants, fitted
TD(lambda) targets with team-perspective sign changes, horizon cohorts, and the
leak-free `steps_from_start` label. Focused regression coverage is in
`tests/test_critic_probe.py`.

Validation after completing these probes:

```text
284 passed in 9.17s
git diff --check passed
```

## Counted-points embed experiment and history ablation

### Overnight embed run

The opt-in `--counted-trick-points` embed observation appends four captured
trick-point totals to the unchanged 196 static features. They are divided by
100, not clamped, and ordered in the observer's usual relative rotation:
self, next clockwise, partner, previous clockwise. Existing history and embed
checkpoints remain 196-dimensional.

The local CPU experiment started fresh play/value weights, attached the mature
passing and frozen GT models from the current standard checkpoint, and trained
for ten hours:

```text
checkpoint: checkpoints/embed_counted_points_local_20260806/latest.pt
games:      2,707,968
log:        logs/embed_counted_points_local_20260806.csv
console:    logs/embed_counted_points_local_20260806.console.log
```

Seat-swapped evaluation against `play` (5,000 rounds per orientation, bundled
passing and GT enabled) found the counted embed model about 54.7 points/round
weaker, with a pooled 37.7% round win rate versus standard's 60.5%.

Strength against the fixed standard opponent by the counted run's lineage
(1,000 rounds per orientation at each checkpoint):

```text
games    counted points/round    counted win rate
0.5M             -99.15              28.95%
1.0M             -79.19              32.70%
1.5M             -72.34              34.80%
2.0M             -58.75              36.80%
2.5M             -58.62              36.25%
2.708M           -56.29              37.40%
```

Interpretation: meaningful improvement continued to roughly 2.0M games, then
plateaued within evaluation noise.

### Critic accuracy by decisions remaining

Reusable diagnostic: `training/critic_value_eval.py`. It records each acting
policy's pre-action value, attaches the realized team-relative round score,
and reports bias, MAE, RMSE, and correlation by remaining main-play decisions.
`steps=1` is immediately before the round-ending main play.

For training-signal analysis, use `--self-play`: each checkpoint must play
against itself. Head-to-head critic calibration is confounded by the strength
difference between models and is not the relevant PPO-training distribution.

```bash
python \
  -m training.critic_value_eval \
  checkpoints/embed_counted_points_local_20260806/latest.pt play \
  --label-a static --label-b history --games 4096 --batch 256 --gt --self-play
```

This produced 303k counted-static and 294k history-policy decisions, with
4,096 self-play rounds per model:

```text
steps       static MAE    history MAE    history improvement
1              27.4          27.4               0.0
2              36.7          35.6               1.1
3              43.2          41.7               1.5
4-5            45.9          43.9               2.0
6-10           61.4          56.2               5.2
11-20          86.4          75.5              10.9
21-40         120.1         101.2              18.9
41-80         143.2         122.1              21.1
81+           146.0         119.1              26.9
overall        116.9          99.3              17.6
```

Overall RMSE was 153.9 points for counted static versus 133.7 for history;
correlation was 0.654 versus 0.680. Bias was nearly zero as it should be in
self-play (`-1.4` static, `+1.7` history). This strongly supports the user's
hypothesis: history does not improve the nearly determined terminal estimate,
but gives the critic a much better baseline earlier in the round, where most
PPO transitions occur. That can materially improve advantage signal-to-noise
and thereby train even the shared static pathway more effectively.

A follow-up instrumented `evaluate()` immediately after every selected main
action, rather than assuming that `steps=1` meant a truly terminal action. On
4,096 fresh self-play rounds per checkpoint, every `steps=1` action set
`round_over` immediately (`4,096/4,096` for both models). There was no hidden
Dragon handoff or automatic continuation behind the terminal measurement.

Adding the one-legal-action restriction removed sampled-action uncertainty but
did not remove the floor:

```text
model        n     bias    MAE +/- SE    RMSE   corr
static      2441    -4.0    24.1 +/- 0.4  31.8  0.984
history     2607    -4.6    24.9 +/- 0.4  32.1  0.980
```

One real observation omission was found: the play-token history records the
Dragon card but not which opponent received its trick. Almost every forced
terminal state had an earlier Dragon give (2,582/2,607 for history), so normal
round scores are not always exactly reconstructible by the history checkpoint.
The counted-static checkpoint is unaffected because it receives each player's
current captured-point total explicitly.

Forced double wins are the useful observable control: captured piles and the
Dragon recipient do not enter double-win scoring, the declaration vector is
explicit, the action ends the round immediately, and there is only one legal
action. Error remained substantial:

```text
model       n    bias    MAE +/- SE    RMSE   corr
static     685    +2.7    25.2 +/- 0.9  34.4  0.896
history    668    -9.0    18.6 +/- 0.6  24.1  0.959
```

This rules out unresolved future play and hidden captured-pile ownership as a
complete explanation for the terminal floor. The remaining error is a genuine
representation/value-optimization failure: neither critic has learned exact
terminal bookkeeping even where the observed state determines the result.

### Frozen-actor raw critic activation experiment

`training/critic_probe.py` tests whether Tanh itself caused the terminal value
floor. It loads the 2.708M counted-static checkpoint, freezes the actor plus
passing/GT policies, and trains four raw-observation critics side by side on
the same realized self-play returns. The actor is only run under `no_grad` for
action selection. Its SHA-256 is checked after every update; it remained
exactly `40a155d9554a...` throughout this run.

Architectures (all targets and outputs use the standard `/400` normalization):

```text
linear:         obs(200) ---------------------------------> value
tanh:           obs(200) -> 256 -> Tanh -> 256 -> Tanh -> value
silu:           obs(200) -> 256 -> SiLU -> 256 -> SiLU -> value
silu_residual:  obs(200) -> 256 -> SiLU -> 256 -> SiLU -> value
                     `---- obs(200) -> Linear -------------'
```

Training used all transitions with ordinary return MSE, Adam `3e-4`, two
epochs, batch size 4,096, four CPU rollout workers, and a stationary mature
self-play actor. A fixed 4,096-round holdout contained 303,285 transitions,
4,096 terminal states, 2,462 forced terminal states, and 665 forced double
wins. Reproducible command:

```bash
python -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 500000 --validation-games 4096 --games-per-update 1024 \
  --eval-every 20480 --checkpoint-every 20480 --batch-size 4096 \
  --epochs 2 --workers 4 \
  --output-dir checkpoints/critic_activation_probe
```

Final MAE in points after 500,000 critic-training rounds:

```text
cohort             original   linear    tanh    silu   silu+residual
all                   115.3    134.4   115.9   116.7      116.9
terminal               27.4     72.2    27.2    32.4       36.3
forced terminal        24.5     71.1    23.7    28.3       33.9
forced double          25.2     69.8    25.4    40.4       60.5
```

Tanh is decisively not the bottleneck. It was the best fresh critic, matched
the original critic around 345k rounds, and then slightly beat it on terminal
and forced-terminal MAE. SiLU learned more slowly, while the direct raw linear
residual did not help and eventually hurt terminal scoring. The linear model's
failure confirms that score prediction needs conditional state logic, not just
an unconditioned sum of point inputs.

The main positive result is separation: a dedicated raw Tanh critic reached
the mature shared critic with far fewer new rounds and without any actor
gradient. This implicates shared actor/critic representation and joint
optimization much more strongly than activation choice. Sample-efficiency is
not perfectly apples-to-apples because the probe trains under a stationary,
mature actor distribution, while the original critic followed a changing
actor during PPO. Saved local artifact:
`checkpoints/critic_activation_probe/latest.pt` (`raw_critic_probe_v1`).

### Forced-terminal-only critic experiment

To distinguish loss dilution from an intrinsic inability to do terminal
bookkeeping, the same four frozen-actor critics were trained only on states
where the next action ended the round and there was exactly one legal action.
The rollout still generates complete games, but every other transition is
excluded from critic gradients. The actor SHA-256 again remained
`40a155d9554a...`.

```bash
python -m training.critic_probe \
  checkpoints/embed_counted_points_local_20260806/latest.pt \
  --games 500000 --validation-games 8192 --games-per-update 1024 \
  --eval-every 51200 --checkpoint-every 51200 --batch-size 1024 \
  --epochs 10 --train-cohort forced_terminal --workers 4 \
  --output-dir checkpoints/critic_terminal_probe
```

Fresh 8,192-round holdout after 500,000 training rounds:

```text
cohort             original   linear    tanh    silu   silu+residual
terminal               26.2     64.3    13.9    21.2       20.1
forced terminal        23.6     64.2     5.3     12.7       10.9
forced double          23.6     39.2     4.7      8.9        8.0
```

The result is decisive: the two-layer raw Tanh branch can learn terminal
bookkeeping. Its forced-terminal MAE fell from 78.8 at 10k rounds to 24.3 at
151k, 11.1 at 356k, and 5.3 at 500k, while the mature shared critic remained
at 23.6. Tanh learned materially faster than either SiLU variant. The linear
model failed because normal finishes and double wins require conditional
logic, not because score components are absent.

The forced action is not hidden information in this cohort. A round-ending
play empties the observer's hand, so its card vector equals the hand mask
already in the observation. If Tichu were still callable, each base action
would appear both with and without the declaration and `legal_lens` would not
be one.

As an observation-sufficiency control, `forced_double_score_oracle()` computes
the exact forced-double margin from the declaration bits. The observer's
partner is necessarily first out, so the relative margin is:

```text
200 - self declaration + left declaration + partner declaration
    + right declaration
```

where Tichu is worth 100 and Grand Tichu 200. It achieved exactly 0.0 bias,
MAE, and RMSE on all 1,377 forced-double holdout states. Therefore the
remaining learned error is optimization/generalization error, not missing
state. Ordinary all-transition value MSE dilutes this rare exact-bookkeeping
signal, and sharing the representation with the actor compounds that problem.
The terminal-only critic is a diagnostic, not a usable full-round value head:
as expected, it deliberately becomes badly calibrated away from terminal
states. Saved artifact: `checkpoints/critic_terminal_probe/latest.pt`.

The discarded head-to-head run produced misleading overall biases of `+44.4`
for static and `-51.5` for history because both critics retained symmetric
self-play priors while history was winning the asymmetric matchup. Do not use
that run to judge intrinsic critic accuracy.

### Clean no-history ablation

Use the checked command below. Do not manually zero the complete history
fusion layer:

```bash
python -m training.ablate_history \
  play /tmp/standard_no_history.pt
```

Implementation: `training/ablate_history.py`; regression coverage:
`tests/test_ablate_history.py`.

The clean ablation preserves every fusion layer's learned static-input
columns and bias, and zeros only its history-input columns. For the current
standard model that means preserving columns `0:256` and zeroing columns
`256:384` in `history_fuse.0.weight`. The command discovers these dimensions,
also handles `critic_history_fuse` when actor/critic history is split, records
metadata in the output checkpoint, and verifies with two different histories
that both logits and critic values change by exactly zero.

Important failure mode: zeroing all of `history_fuse.0.weight` and its bias is
not a history-only ablation. It also deletes a learned static residual. That
incorrect procedure made standard appear roughly 36 points/round weaker than
the counted model and must be discarded.

Correct seat-swapped result (5,000 rounds per orientation): standard with
history input disabled remained about 12.7 points/round stronger than the
counted embed model; pooled round win rates were essentially tied (49.1% for
no-history standard, 48.7% for counted embed). Full standard was about 54.7
points/round stronger, so history contributed roughly 42 points/round in this
matchup.

Validation after adding the automated ablation:

```text
272 passed in 8.75s
git diff --check passed
```

## Latest completed work

The action-vector cache, dense rollout-history storage, and stale adapt
prefetch issues from the run-efficiency audit have been fixed and validated.
The standard Kaggle history bundle now runs synchronously. Read
`TRAINING_EFFICIENCY_WRITEUP.md` for the full problem analysis, representation
design, scheduling semantics, benchmarks, tradeoffs, changed files, deferred
audit findings, and validation results.

Current validation:

```text
226 passed in 7.74s
git diff --check passed
compileall passed
```

The worktree still contains uncommitted changes from several completed tasks.
Do not discard them.

## Earlier handoff snapshot

## Project and environment

- TichuBot is a PyTorch PPO/self-play project for the four-player card game
  Tichu. Use `python`; the system Python
  does not provide the project runtime dependencies.
- The terminal game and advisor are in `training/play.py` and
  `training/advisor.py`. Keep their game logic and CLI workflow intact. The
  requested product direction is a visual browser layer over those programs,
  not a replacement game engine or a rewrite of core logic.

## Kaggle state

- Fetched completed standard leg:
  `kaggle_runs/2026-07-25_203727_standard/`
  - final checkpoint:
    `tichubot_outputs/checkpoints/history_standard_gpu/latest.pt`
  - checkpoint game count: 12,653,568.
- Full post-fetch drill completed successfully with
  `python -m training.leg_eval --run kaggle_runs/2026-07-25_203727_standard`:
  - versus prior leg (11,281,408 games): `+5.3` points/round, 50.4% wins;
  - versus embed-700k baseline: `+72.5` points/round, 65.9% wins;
  - GT head: 7.8% mean call, strength/probability correlation `+0.52`;
  - same-weights GT-on vs GT-off: `-2.4` points/round; GT declared 15.3% of
    rounds and made 54.9% of them. Interpret the last result cautiously.
- Started the next standard T4 leg by resuming the above checkpoint. Kaggle
  accepted kernel version 39 at 2026-07-26 09:10:40 CEST.
  - saved fetch state: `kaggle/.runs/_last.json`
  - expected output directory:
    `kaggle_runs/2026-07-26_091035_standard/`
  - when complete: `bash kaggle/run.sh --fetch standard`
  - status API is known to be stale; use the Kaggle UI to confirm completion.

## Uncommitted worktree state

Do not discard these files. `training/model.py` was already modified before
this handoff work began. It only removes redundant parentheses in one
log-probability expression and appears user-authored/unrelated.

### Replay-pass standardization (substantial, unfinished review)

Modified: `training/play.py`

New untracked test: `tests/test_play_replay.py`

The implementation agent added:

- `PassChoice` type alias and made `ai_pass()` return the directed pass tuple;
- `run_replay_passing_phase()`, which reuses recorded directed passes for
  every non-replayed player during AI replay, while the replayed seat samples
  the current pass policy;
- live-round recording of every player's pass and initial Tichu declarations;
- `run_ai_replay()` parameters to receive that recorded state and a
  `standardize_opponent_passes` switch;
- CLI option `--replay-opponent-passes {recorded,ai}`, default `recorded`.

Intent: during replay, a different decision by the replayed player should be
visible without non-human pass randomness changing the deal. Normal gameplay
rules and non-replay passing stay unchanged.

The test asserts that opponent recorded passes are repeated deterministically
and that all directed passes remain legal. It has not been independently run
or reviewed by the primary agent after the interruption. Recommended next
commands:

```bash
python -m pytest tests/test_play_replay.py
python -m pytest tests/test_game_state.py
```

Review concern: confirm replay semantics are correct when the replayed seat's
new pass changes cards received by all players. The current behavior keeps
other players' *outgoing* directed choices fixed, as requested; it cannot
make all post-pass hands identical once the replayed seat selects new cards.

### Visual UI (only a start; no server/integration exists)

Untracked asset: `training/web_static/assets/english-pattern-deck-cc0.svg`
(about 7.8 MB).

No `training.web` module, HTML, JavaScript, HTTP server, tests, or launch
command was completed. The visual agent's intended design was a separate
`training.web` layer that reuses `GameState` and play/advisor helpers while
preserving the CLI. Before keeping or shipping the SVG asset, verify its
actual upstream source/license even though its filename says `cc0`.

Suggested continuation:

1. Choose a minimal local browser framework or standard-library HTTP server;
   avoid a game engine and avoid a full rewrite of `training.play`.
2. Factor only the small state/action boundary needed to drive `GameState`
   from HTTP/WebSocket handlers.
3. Render the human hand with the local card asset, legal-action selection,
   table state, turn state, and advisor recommendation.
4. Start a local dev server, validate desktop/mobile interaction, and add
   focused tests without weakening the CLI path.

## Current git status expected at handoff

- modified: `training/model.py` (pre-existing)
- modified: `training/play.py` (replay-pass implementation)
- untracked: `tests/test_play_replay.py`
- untracked: `training/web_static/assets/english-pattern-deck-cc0.svg`
- untracked: this `MEMORY.md`

## 2026-08-12 — History optimizer-batch Kaggle sweep (prepared)

Direct inspection confirmed that history PPO previously treated
`batch_size=1024` only as a forward/backward memory microbatch and accumulated
the complete roughly 70k-transition adapt rollout before each Adam step. The
new implementation separates the two controls:

- `--batch-size`: padding/VRAM memory microbatch;
- `--optimizer-batch-size`: transitions averaged per main-policy Adam step;
- omitted optimizer batch preserves the legacy full-rollout behavior.

The finite optimizer-batch path scales partial microbatch losses to the exact
optimizer-batch mean, clips once per optimizer batch, and randomizes completed
padding-efficient microbatches so sorted history/action buckets are not
consumed monotonically. Dragon/wish heads retain one update per epoch because
their gradients and parameters are disjoint from the main play loss.

The dedicated private Kaggle bundle is `optimizer-sweep`, kernel
`<kaggle-user>/tichu-optimizer-batch-sweep`. It requires this exact source:

```text
kaggle_runs/2026-08-11_231044_compact-scratch/
  tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
source games: 1,748,992
```

Before training, the kernel replays one frozen 512-game rollout through fresh
source-checkpoint clones with memory microbatches 256, 512, 1024, 2048, and
4096 (two measured repeats after warmup). It records PPO seconds and true peak
CUDA allocated/reserved memory, then selects the fastest successful size no
larger than 1024. The cap is necessary because the smallest optimizer batch is
1024; using a larger forward batch only for larger optimizer candidates would
confound the learning comparison.

Three independent variants then start from the exact same source checkpoint.
The sole training variable is effective optimizer batch: 1024, 2048, or 4096.
Each trains 262,144 new self-play rounds (256 updates of 1024 games, two PPO
epochs), ending at 2,011,136 games. Architecture, LR, entropy schedule,
rollout size, memory microbatch, synchronous collection, and source
passing/GT policies are shared. Passing and GT updates are frozen during the
sweep. The current nonlinear action encoder, Phoenix semantics, shared-history
rollout representation, pass-provenance revival, and other 2026-08-12 fixes
are included in the archived source.

Evaluation is balanced across both team/seat assignments with 4096 rounds per
orientation. Every variant is compared against the source and every pair of
variants is compared directly. Exact outputs are written under
`tichubot_outputs/optimizer_batch_sweep/`, including:

- `memory_microbatch_benchmark.json`;
- `experiment_manifest.json`;
- per-variant CSV logs and final checkpoints;
- `evaluation_summary.json`.

Prepared push command:

```bash
bash kaggle/run.sh optimizer-sweep --resume \
  kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
```

Pre-push validation:

```text
305 passed in 20.86s
git diff --check passed
generated kernel run.py compiled successfully
local benchmark smoke: one real rollout, two memory sizes, completed
```

Final Kaggle version, timing, memory benchmark, checkpoint Adam counters, and
balanced evaluation metrics are pending and must be added after completion.

Launch accepted at 2026-08-12 13:45:24 CEST:

```text
kernel: <kaggle-user>/tichu-optimizer-batch-sweep
kernel version: 1
accelerator: NvidiaTeslaT4
private input dataset: <kaggle-user>/tichu-resume-optimizer-sweep
saved fetch state: kaggle/.runs/optimizer-sweep.json
expected fetch directory: kaggle_runs/2026-08-12_134514_optimizer-sweep
UI: https://www.kaggle.com/code/<kaggle-user>/tichu-optimizer-batch-sweep
```

Fetch after the UI reports completion:

```bash
bash kaggle/run.sh --fetch optimizer-sweep
```

## 2026-08-12 — Optimizer sweep scope correction and statistical benchmark

The learning/evaluation sweep described immediately above was intentionally
canceled: 262,144 training games per variant and 4096 evaluation games per
orientation were not sufficient to support the requested learning-quality
claim, and learning quality was not the question being asked. Kernel version 1
was manually canceled. A one-rollout coarse timing kernel was then pushed and
also manually canceled because timing repeats on one frozen rollout estimate
device noise but not rollout-to-rollout variance.

The active replacement does **no persistent training and no policy
evaluation**. Every timed PPO pass uses a fresh disposable clone of the same
source checkpoint, one epoch, and the full frozen rollout as its effective
optimizer batch. Only the forward/backward memory microbatch changes. It
records PPO time, transitions/second, and true peak CUDA allocated/reserved
memory.

Experimental design:

```text
source checkpoint: kaggle_runs/2026-08-11_231044_compact-scratch/
  tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
source games: 1,748,992
GPU: NvidiaTeslaT4
rollout games per independent block: 512
candidate memory batches:
  256, 512, 768, 1024, 1280, 1536, 1792, 2048,
  2560, 3072, 4096, 6144, 8192
pilot: 8 independently seeded rollouts, 2 randomized-order timing repeats each
confirmation: 24 fresh independently seeded rollouts, 3 randomized-order
  timing repeats each
seed: 20260812 (confirmation seeds begin at 21260812)
maximum timed PPO replays: 1,144, plus one untimed warmup
```

The pilot selects one candidate by geometric mean PPO time. The confirmation
stage tests that independently selected candidate against every candidate that
completed all pilot blocks. The analysis uses per-rollout median time, paired
log-time differences, 200,000-sample one-sided sign-flip tests, Holm correction
at family-wise alpha 0.05, and 50,000-sample bootstrap speedup intervals. This
avoids reusing selection rollouts for confirmatory inference. A statistically
unique winner is reported only if it beats every other feasible candidate after
Holm correction; an inconclusive/tied result is valid and must not be promoted
to a unique optimum.

Exact launch command:

```bash
bash kaggle/run.sh optimizer-sweep --resume \
  kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
```

Validation before launch:

```text
307 passed in 9.77s
generated run.py compiled successfully
git diff --check passed
local end-to-end statistical smoke completed on CPU
```

Active launch:

```text
kernel: <kaggle-user>/tichu-memory-microbatch-benchmark
kernel version: 3
accelerator: NvidiaTeslaT4
private input dataset: <kaggle-user>/tichu-resume-optimizer-sweep
pushed: 2026-08-12T14:02:49+02:00
saved fetch state: kaggle/.runs/optimizer-sweep.json
expected fetch directory: kaggle_runs/2026-08-12_140244_optimizer-sweep
status immediately after push: RUNNING
UI: https://www.kaggle.com/code/<kaggle-user>/tichu-memory-microbatch-benchmark
```

Fetch after completion:

```bash
bash kaggle/run.sh --fetch optimizer-sweep
```

Primary outputs are
`tichubot_outputs/memory_batch_benchmark/experiment_manifest.json` and
`statistical_memory_microbatch_benchmark.json`. The result JSON is written
atomically after every completed rollout, so partial measurements survive an
interrupted run.

### Final result

Kernel version 3 completed successfully and was fetched to
`kaggle_runs/2026-08-12_140244_optimizer-sweep/`. Kernel execution took about
4976 seconds (82.9 minutes). All 8 pilot and 24 fresh confirmation rollouts
completed; each contained about 36.0k--37.0k play transitions.

The pilot selected memory microbatch **512**. Confirmation established it as
the unique fastest feasible candidate at family-wise alpha 0.05. Key
confirmation results (geometric-mean transitions/second, maximum PyTorch CUDA
allocated/reserved memory across the successful measurements) were:

```text
batch   transitions/s   max allocated   max reserved
  256         10,042        715 MiB        1,032 MiB
  512         10,928      1,498 MiB        2,210 MiB
  768         10,741      2,424 MiB        3,652 MiB
 1024         10,443      3,347 MiB        5,084 MiB
 1280         10,117      3,347 MiB        5,576 MiB
 1536          9,944      5,355 MiB        7,856 MiB
 1792          9,922      3,348 MiB        6,592 MiB
 2048          9,261      7,608 MiB       10,856 MiB
 2560          8,778      3,844 MiB        8,482 MiB
 3072          8,401     11,643 MiB       14,360 MiB
 4096     incomplete (OOM in 6/72 timing attempts; 22/24 full blocks)
 6144     excluded after OOM on every pilot rollout
 8192     incomplete (OOM in 9/72 timing attempts; 21/24 full blocks)
```

Paired confirmatory comparisons for 512:

```text
vs batch   geometric speedup   bootstrap 95% interval   Holm-adjusted p
     256          1.0883             [1.0815, 1.0955]       4.50e-5
     768          1.0174             [1.0111, 1.0236]       4.50e-5
    1024          1.0465             [1.0406, 1.0525]       4.50e-5
    1280          1.0802             [1.0710, 1.0895]       4.50e-5
    1536          1.0990             [1.0908, 1.1071]       4.50e-5
    1792          1.1014             [1.0914, 1.1120]       4.50e-5
    2048          1.1800             [1.1692, 1.1910]       4.50e-5
    2560          1.2450             [1.2341, 1.2560]       4.50e-5
    3072          1.3008             [1.2854, 1.3165]       4.50e-5
```

512 beat 1024 on all 24 paired confirmation rollouts and beat 2048 on all
24. It beat the close 768 candidate on 21 of 24. Therefore the production
recommendation for this exact compact history model and Tesla T4 is
`--batch-size 512`. This is the padding/VRAM memory microbatch only; it does
not imply an optimizer batch of 512. Keep optimizer batch configured
separately according to the learning experiment, or omit it to preserve the
legacy full-rollout update.

Post-run qualification (2026-08-14): this experiment establishes the isolated
single-process PPO runtime optimum, not end-to-end production VRAM safety. It
collected each frozen rollout with `n_workers=1`, deleted the rollout models,
called `torch.cuda.empty_cache()`, and only then timed disposable PPO clones.
Production instead keeps four CUDA rollout worker processes alive while the
main process performs PPO. Consequently the benchmark did not measure the
aggregate caching-allocator footprint that later caused the compact scratch
run to OOM. Keep 512 as the measured PPO speed optimum, but do not treat it as
a production safety guarantee until the worker-lifetime issue is fixed and
tested with the full process topology.

## 2026-08-13 — Adam optimizer-batch runtime curve (prepared)

A separate runtime-only benchmark is prepared as bundle `optimizer-runtime`,
kernel `<kaggle-user>/tichu-adam-batch-runtime`. It fixes the statistically
validated memory microbatch at 512 and varies only the effective Adam batch:

```text
512, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
12288, 16384, 24576, 32768, 49152, 65536, full rollout
```

The design uses 24 independently seeded production-sized 1024-game rollouts,
two randomized-order timing repeats per candidate per rollout, one disposable
PPO epoch per measurement, and no persistent training. It reports optimizer
step counts, runtime/throughput, memory, paired bootstrap intervals versus the
one-step full-rollout asymptote, paired adjacent-size gains, and Holm-corrected
sign-flip tests. The predeclared diminishing-returns knee is the smallest
finite Adam batch whose paired bootstrap 95% upper runtime-ratio bound is no
more than 1.02 versus full-rollout accumulation.

Prepared launch command:

```bash
bash kaggle/run.sh optimizer-runtime --resume \
  kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
```

Pre-launch validation:

```text
308 passed in 10.31s
generated run.py compiled successfully
git diff --check passed
local end-to-end optimizer-runtime smoke completed on CPU
```

The first launch attempt was not executed because external checkpoint upload
requires explicit payload-and-destination approval. Pending approval, the
exact external transfer is the 5.62 MiB `latest.pt` above to private Kaggle
dataset `<kaggle-user>/tichu-resume-optimizer-runtime`, followed by private T4
kernel `<kaggle-user>/tichu-adam-batch-runtime`.

Approval was received and the checkpoint upload completed. Kaggle kernel
version 1 was pushed while the newly created dataset slug was still being
registered; Kaggle dropped the input attachment, and version 1 exited after
about 15 seconds with `Expected exactly one source checkpoint ... found 0`.
This is not an experimental run and has no measurements. `kaggle/run.sh` now
waits for a resume dataset to report ready before pushing a kernel.

After the dataset reported ready, the same generated bundle was pushed again
without re-uploading or changing the experiment. The active run is:

```text
kernel: <kaggle-user>/tichu-adam-batch-runtime
kernel version: 2
accelerator: NvidiaTeslaT4
private input dataset: <kaggle-user>/tichu-resume-optimizer-runtime
checkpoint: latest.pt, 5.62 MiB, source games 1,748,992
saved fetch state: kaggle/.runs/optimizer-runtime.json
expected fetch directory: kaggle_runs/2026-08-12_205331_optimizer-runtime
status after corrected push: RUNNING
UI: https://www.kaggle.com/code/<kaggle-user>/tichu-adam-batch-runtime
```

Fetch after completion:

```bash
bash kaggle/run.sh --fetch optimizer-runtime
```

### Final Adam-runtime result

Corrected kernel version 2 completed and was fetched to
`kaggle_runs/2026-08-12_205331_optimizer-runtime/`. It completed all 24
independent 1024-game rollouts and all 720 timed measurements in about 5975
seconds (99.6 minutes). The memory microbatch stayed fixed at 512 throughout.

```text
Adam batch       PPO sec    transitions/s   median Adam steps/epoch
       512        7.4919          9,747                143
      1024        7.2358         10,092                 72
      1536        7.1427         10,223                 48
      2048        7.1009         10,284                 36
      3072        7.0828         10,310                 24
      4096        7.0616         10,341                 18
      6144        7.0136         10,412                 12
      8192        6.9953         10,439                  9
     12288        7.0016         10,429                  6
     16384        7.0066         10,422                  5
     24576        6.9797         10,462                  3
     32768        6.9722         10,473                  3
     49152        6.9671         10,481                  2
     65536        6.9901         10,447                  2
full rollout      6.5483         11,151                  1
```

Adjacent paired gains show the finite-batch elbow:

```text
512 -> 1024:   3.54% speedup, 95% CI [3.17%, 3.88%]
1024 -> 1536:  1.30% speedup, 95% CI [0.97%, 1.62%]
1536 -> 2048:  0.59% speedup, 95% CI [0.11%, 1.06%]
2048 -> 3072:  0.25% speedup, 95% CI includes no gain
3072 -> 4096:  0.30% speedup, 95% CI includes no gain
```

Therefore **2048 is the practical runtime knee among finite Adam batches**:
it is the smallest tested value after which the next increase has no reliable
runtime gain, and increasing from 2048 all the way to the best observed finite
plateau at 49152 improves throughput only another 1.92%. A more aggressive
small-batch choice of 1536 gives up only 0.59% runtime versus 2048.

The predeclared strict endpoint rule returned no finite knee: even 49152 was
6.40% slower than the legacy full-rollout path, and every finite setting was
significantly slower than full rollout after Holm correction. This endpoint
must not be interpreted as a smooth Adam-step asymptote. In the implementation,
`optimizer_batch_size=None` both takes one Adam step and retains monotonically
ordered padding buckets, while any finite value enables shuffled bucketed
microbatches. Thus the full-rollout discontinuity includes both optimizer-step
and execution-order effects. It does represent the real runtime of the two
production modes, but it does not isolate the marginal cost of the last Adam
step.

Runtime evidence alone therefore supports `--optimizer-batch-size 2048` as a
reasonable learning experiment default, not as a claim that 2048 learns best.
The learning-quality choice still requires powered training/evaluation.

Primary result:
`kaggle_runs/2026-08-12_205331_optimizer-runtime/tichubot_outputs/adam_batch_runtime/statistical_adam_batch_runtime.json`.

## 2026-08-13 — Batch optima applied to next compact scratch run

The next fresh `compact-scratch` Kaggle bundle now passes the two measured T4
settings explicitly:

```text
--batch-size 512
--optimizer-batch-size 2048
```

The generated command retains `--games-per-update 1024`, two PPO epochs from
the trainer default, four rollout workers, and synchronous collection. The
settings are deliberately scoped to the compact history profile because the
benchmarks used that exact architecture and a Kaggle Tesla T4. Generic
`TrainConfig`, standard-history, embed, monster, and resume-lineage defaults
were not silently changed.

Prepared fresh-run command (not launched):

```bash
bash kaggle/run.sh compact-scratch
```

Validation after regenerating `kaggle/compact_scratch_bundle/run.py`:

```text
generated command contains --batch-size 512 --optimizer-batch-size 2048
generated run.py compiles
310 passed in 9.44s
git diff --check passed
```

## 2026-08-13 — Powered architecture runtime-attribution experiment pushed

A private Kaggle Tesla T4 experiment was pushed to locate why the compact
history architecture's 41.5% parameter reduction produces a much smaller
end-to-end throughput improvement than parameter count alone suggests.

```text
kernel:  <kaggle-user>/tichu-architecture-runtime-attribution
version: 1
status at push verification: RUNNING
pushed:  2026-08-13T11:08:42+02:00
run:     kaggle_runs/2026-08-13_110840_architecture-runtime
UI:      https://www.kaggle.com/code/<kaggle-user>/tichu-architecture-runtime-attribution
```

The predeclared controlled design is:

```text
hardware:                       Kaggle Nvidia Tesla T4
warm-up blocks (excluded):      4
target measured paired blocks: 48
minimum valid paired blocks:   32
games/architecture/block:      1,024
rollout workers:               4, synchronous CUDA collection
memory batch, both models:     512
Adam batch, both models:       2,048
PPO epochs, both models:       2
wall-clock cap:                55 minutes
paired bootstrap samples:      50,000
paired sign-flip samples:      200,000
family-wise alpha:             0.05 (Holm correction)
equivalence margin:            +/-2%
persistent training:           false
```

Each block uses a new seed and alternates standard/compact execution order.
Both architectures use neutral behavior heads, so they traverse the same
environment trajectory while still executing their complete trunk, history,
action encoder, value, and specialist computation. The run asserts exact
trajectory equality before accepting a block. PPO replays the same frozen
rollout through both architectures, alternates PPO order, and restores the
same populated model and Adam state before each timed update. Thus architecture
is the controlled variable; rollout trajectory, memory batch, effective Adam
batch, epochs, and optimizer-state maturity are held fixed.

Primary paired endpoints are modeled cycle wall time, rollout wall time, and
PPO wall time. Non-overlapping rollout stages and CUDA-event PPO stages are a
Holm-corrected diagnostic family. Four excluded warm-up blocks precede all
statistical measurements. The experiment stops at 55 minutes only after at
least 32 measured pairs; otherwise it targets all 48 pairs. It performs no
checkpoint training or evaluation benchmark.

Local validation before push:

```text
single-process end-to-end smoke passed
multi-process state/cache smoke passed
generated Kaggle run.py compiled
314 passed in 9.62s
git diff --check passed
```

Fetch after completion with:

```bash
bash kaggle/run.sh --fetch architecture-runtime
```

## 2026-08-13 — Architecture runtime attribution result (33 paired blocks)

Kaggle version 1 completed successfully at the predeclared wall-clock limit.
It finished 33 measured paired blocks after four excluded warm-ups, exceeding
the minimum of 32. Total statistical-run time was 55.76 minutes. The outputs
were fetched to:

```text
kaggle_runs/2026-08-13_110840_architecture-runtime/
```

Primary result:

```text
tichubot_outputs/architecture_runtime/statistical_architecture_runtime.json
```

The controlled current-code policies had 770,673 standard parameters and
455,697 compact parameters: compact had 314,976 fewer, or 40.87% fewer.

All three predeclared primary paired endpoints were significant after Holm
correction (each adjusted p = 2.999985e-05):

```text
endpoint       standard median   compact median   geometric std/compact ratio (95% CI)
modeled cycle       45.386 s         44.681 s      1.01484 [1.01085, 1.01875]
rollout             17.614 s         18.107 s      0.97536 [0.97020, 0.98061]
PPO                 27.576 s         26.334 s      1.04193 [1.03668, 1.04713]
```

Thus compact is only 1.48% faster end to end (95% CI 1.09%-1.87%), despite
having 40.87% fewer parameters. Compact PPO is 4.19% faster, but compact
rollout is 2.53% slower (inverse of the reported standard/compact time ratio),
canceling most of the PPO gain. The modeled-cycle result is both statistically
different and inside the predeclared +/-2% practical-equivalence margin.

Holm-corrected non-overlapping rollout diagnostics locate the regression:

```text
stage (per 100k transitions)  standard median  compact median  result
policy forward                    17.636 s        18.405 s     compact slower
immediate decisions                5.871 s         6.268 s     compact slower
environment/post-play              5.883 s         6.356 s     compact slower
```

The latter two timers include specialist-network work rather than being pure
environment cost. Their nested descriptive timers show the compact specialist
paths taking materially longer:

```text
pre-play specialist mean/worker median:  0.486 s standard, 0.589 s compact
post-play specialist mean/worker median: 0.618 s standard, 0.750 s compact
```

The compact architecture's split actor/critic history and static-conditioned
pooling add readout work during every action and specialist decision, so fewer
parameters do not imply cheaper rollout inference. Fixed Python environment,
legal-action, encoding, IPC, and buffer costs were otherwise not distinguished.

PPO attribution shows where compact does win:

```text
stage          standard median  compact median  geometric ratio/result
forward/loss        8.280 s          8.121 s     compact 1.67% faster
backward           18.074 s         17.022 s     compact 5.95% faster
Adam step           0.294 s          0.342 s     compact 16.93% slower
```

The Adam reversal is consistent with tensor-count overhead: the standard
policy has 70 parameter tensors while compact has 87 because the split and
normalization structure creates more small tensors, despite fewer total
elements. This is a mechanism-consistent interpretation, not a separately
randomized causal ablation.

Operationally, model shrinking alone is no longer the next throughput target.
The measured bottleneck is the extra split/static-conditioned rollout
readouts—especially specialist decisions—followed by small-tensor Adam launch
overhead. Any optimization should preserve the architecture and first reuse or
fuse actor/critic history features in rollout inference, then repeat this same
paired benchmark.

## 2026-08-13 — Fresh Phoenix-aware compact scratch run pushed

A new compact lineage was pushed from scratch after incorporating the improved
Phoenix action representation, nonlinear action encoder, and measured T4 batch
settings. No resume checkpoint was attached.

```text
kernel:  <kaggle-user>/tichu-compact-scratch
pushed:  2026-08-13T13:24:45+02:00
run:     kaggle_runs/2026-08-13_132440_compact-scratch
final Kaggle status: ERROR (training checkpoint preserved; GT-probe OOM)
UI:      https://www.kaggle.com/code/<kaggle-user>/tichu-compact-scratch
```

The generated training command includes:

```text
Phoenix-aware action width: 58 (56 card bits + call bit + Phoenix-high bit)
nonlinear action encoder:    enabled
games/update:                1,024
rollout workers:             4, synchronous CUDA
memory batch:                512
effective Adam batch:        2,048 (shuffled bucket microbatches)
PPO epochs:                  2
resume checkpoint:           none
```

Pre-push validation:

```text
embedded source archive contains the 58d Phoenix-aware representation
generated run.py compiles
29 focused Phoenix/action/batch/rollout tests passed
git diff --check passed
```

Fetch after completion with:

```bash
bash kaggle/run.sh --fetch compact-scratch
```

Fetched on 2026-08-14. Kaggle ran for about 7 h 11 min and completed 782
ordinary PPO updates before failing during post-checkpoint probe practice:

```text
last saved checkpoint: ckpt_00800768.pt (also latest.pt)
ordinary games:        800,768
transitions:           58,241,355
median throughput:     36.5 games/s
harmonic throughput:   36.40 games/s
ordinary cycle time:   6.112 h (3.113 h PPO, 2.999 h rollout inferred)
final row:              return -0.0240, win 46.8%, entropy 0.37799,
                        KL 0.00455, clip fraction 0.04166
```

The 200,704-, 400,384-, and 600,064-game GT-probe blocks each completed all
40,000 game-equivalents (20 independent logged chunks of 1,600 paired diffs).
The fourth block began after saving the 800,768-game checkpoint and completed
22,000/40,000 game-equivalents before the failure. Aggregating the logged
chunk means (chunk-level t intervals, useful as run diagnostics rather than a
checkpoint arena comparison):

```text
checkpoint  probe games  mean control-differenced score  mean declare p
200,704       40,000      -101.59 pts  [ -104.22, -98.97]   0.0065
400,384       40,000       -97.39 pts  [ -100.15, -94.63]   0.0238
600,064       40,000       -95.69 pts  [  -99.58, -91.81]   0.0479
800,768       22,000      -103.61 pts  [ -109.36, -97.86]   0.0531 (partial)
```

Exact failure: `torch.OutOfMemoryError` in
`NonlinearActionEncoder.residual[0]` during `ppo_update_history` for the fourth
GT-probe practice block. The failed forward tried to allocate 334 MiB with
only 310.75 MiB free on the 14.56 GiB T4. At that instant the four persistent
CUDA rollout workers held 3.49, 3.32, 2.56, and 3.19 GiB, while the main
process held 1.58 GiB. The optimizer's logical 2,048-transition batch was not
materialized at once: the failing forward used the configured 512-row memory
microbatch and padded it to that bucket's maximum legal-action count. Thus the
evidence points to multi-process CUDA cache/peak accumulation plus a large
padded probe microbatch, not to 2,048-row gradient accumulation itself.

The older successful compact runs are not counterexamples to this mechanism:
the 2026-08-01, 2026-08-03, and 2026-08-11 bundles used a single 57->32 linear
action projection and ran memory microbatch 1,024 without OOM (8 GT-probe
blocks each). The new run uses the 58d Phoenix input plus a residual MLP with
two 128-wide hidden activations before returning to width 32. For the failed
512-row forward, the requested 334 MiB first-layer output implies a padded
legal-action width of about 1,336 (`512 * 1336 * 128 * 4 bytes`). At the same
legal width the old 1,024-row linear output was only about 167 MiB. Autograd
must additionally retain the new MLP's intermediate activations.

This makes the immediate trigger unlucky (an extreme legal-action bucket),
but the missing headroom systematic. The production worker pool has no CUDA
cache cleanup/recycling path; four long-lived workers independently retain
their historical allocation peaks. By contrast the batch benchmark explicitly
used one rollout worker and emptied the CUDA cache before timed PPO. The
correct next target is worker memory lifetime/recycling under the real
four-worker topology, not an unsupported claim that 512 itself is too large.

Reproducible artifacts:

```text
kaggle_runs/2026-08-13_132440_compact-scratch/tichu-compact-scratch.log
kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/logs/train_history_compact_scratch.csv
kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00800768.pt
```

## 2026-08-14 — Conditional CUDA worker-cache guard

The rollout worker lifecycle now checks CUDA memory after every worker job in
both normal and GT-probe collection. This does not change rollout, PPO, batch,
or optimizer semantics.

Fast path:

```text
read memory_allocated, memory_reserved, and global device free memory
if device free >= min(4 GiB, 30% of device): do not call empty_cache
```

Pressure path:

```text
if global free is below the target and this worker has >=64 MiB unused cache:
    call torch.cuda.empty_cache() in that worker
```

The guard distinguishes live tensor memory (`allocated`) from allocator cache
(`reserved - allocated`). Each persistent worker records its lowest observed
post-rollout live allocation. Growth beyond both 64 MiB and 25% of that floor
raises a `live-growth-alert` in the rollout profile; `empty_cache` is not
misrepresented as a cure for live references. The aggregate profile also logs
worker allocated/reserved/cached MiB, minimum device free before/after,
released MiB, cleanup count, alert count, and total check time. This makes the
overhead measurable on the next production run.

The 4 GiB T4 target comes from the observed main-process PPO peak just above
3 GiB plus safety margin. Main PPO already releases its own unused cache after
every update, so this guard addresses the four independent persistent worker
allocators. A healthy rollout performs counter reads only; a cleanup can cost
some allocation time on the next rollout and is deliberately pressure-gated.

Validation:

```text
23 focused rollout/GT/CUDA-guard tests passed
319 full-suite tests passed in 9.89 s
py_compile passed
git diff --check passed
```

GPU overhead and the pressure threshold still require validation with the real
four-worker Kaggle topology before claiming the OOM is resolved.

## 2026-08-14 — Local paired 800k-vs-1.7M compact strength check

The 800,768-game nonlinear-action compact checkpoint was evaluated locally on
CPU against both independent mature compact runs.  Each comparison used 4,096
fixed deal seeds and replayed every deal with the candidate on seats 0/2 and
then 1/3 (8,192 rounds per opponent).  Passing, Grand Tichu, dragon, and wish
heads were enabled.  The primary endpoint was the deal-paired candidate score
advantage per round.  Inference used 32 contiguous 128-deal blocks, 50,000
block-bootstrap samples, 200,000 block sign flips, and family-wise correction
across both opponents (Bonferroni simultaneous intervals and Holm p-values).
The sample size was fixed before the run; there was no early stopping.

Exact command:

```bash
python -m training.paired_checkpoint_eval \
  new800=kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00800768.pt \
  old_aug01=kaggle_runs/2026-08-01_223035_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  old_aug11=kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --output local_evaluations/paired_compact_800k_vs_17m.json \
  --device cpu --deals 4096 --batch 256 --block-size 128 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 --seed 20260814
```

Results (positive means the 800k candidate was stronger):

```text
opponent                         advantage    ordinary 95% CI     simultaneous 95% CI    W / L / D        Holm p
Aug-01, 1,728,512 games          +0.695       [-3.158, +4.609]    [-3.661, +5.149]        48.80/48.57/2.62  0.7335
Aug-11, 1,748,992 games          -1.545       [-4.480, +1.436]    [-4.899, +1.857]        48.41/48.66/2.93  0.6526
```

Neither difference was statistically significant at family-wise alpha 0.05.
The defensible conclusion is that this evaluation cannot distinguish the
800,768-game checkpoint from either 1.7M checkpoint in playing strength; it
does not establish exact equivalence.  The full raw seeds, scores, block
means, checkpoint configs, and statistics are preserved in
`local_evaluations/paired_compact_800k_vs_17m.json`.  Total local runtime was
280.5 seconds.

## 2026-08-14 — Compact continuation from the validated 800k checkpoint

Kaggle kernel version 7 was pushed at 2026-08-14 08:41:59 CEST and reported
`KernelWorkerStatus.RUNNING` on the immediate post-push status check.

```text
kernel:          <kaggle-user>/tichu-compact-scratch
local run id:    kaggle_runs/2026-08-14_084154_compact-scratch
accelerator:     NvidiaTeslaT4
resume games:    800,768
resume source:   kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00800768.pt
resume dataset:  <kaggle-user>/tichu-resume-compact-scratch
max wall time:   11.7 h (compact profile default)
reinit GT:       no
UI:              https://www.kaggle.com/code/<kaggle-user>/tichu-compact-scratch
```

Exact push command:

```bash
bash kaggle/run.sh compact-scratch --resume \
  kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00800768.pt
```

The bundle keeps the measured compact settings (`batch_size=512`,
`optimizer_batch_size=2048`, four synchronous workers), the nonlinear
58-dimensional Phoenix/action encoder, static attention, and the conditional
CUDA worker-cache guard.  The resume contract is baked into the kernel, so it
aborts instead of silently starting from scratch if the input checkpoint is
missing.

## 2026-08-14 — Completed 2M continuation: download and paired strength eval

Kernel version 7 completed normally at the 11.7-hour wall-clock cap. Its
outputs were downloaded with:

```bash
bash kaggle/run.sh --fetch compact-scratch
```

Downloaded artifact summary:

```text
run:                  kaggle_runs/2026-08-14_084154_compact-scratch
files / disk:         203 files / 728 MiB
periodic checkpoints: 120
resume games:         800,768
final games:          2,001,920 (latest.pt interrupt checkpoint)
games added:          1,201,152 across 1,173 ordinary PPO updates
median throughput:    33.1 games/s
harmonic throughput:  33.59 games/s
median PPO time:      14.63 s/update
final entropy:        0.25201
final logged win:     46.4% (self-play diagnostic, not arena strength)
latest.pt sha256:     013d48893ddf6a4da830571fb097c39d8958387abc044f100d88abbb81f8b34f
```

The final periodic checkpoint was `ckpt_02000896.pt`; after the last ordinary
update, the wall-cap handler saved the newer 2,001,920-game state to
`latest.pt`. Use `latest.pt` for evaluation and future continuation.

There was no OOM or traceback. The conditional worker-cache guard fired three
times, releasing 2,246 MiB, 1,618 MiB, and 3,152 MiB respectively. No
live-growth alert was logged. This run demonstrates that the guarded
four-worker topology can survive a full production leg, though it cannot prove
the counterfactual that the unguarded run would necessarily have failed.

The final checkpoint was evaluated locally against its exact 800,768-game
ancestor and both independent mature compact references. Each comparison used
4,096 fixed deal seeds in both team orientations (8,192 rounds), 32 independent
128-deal blocks, all deployed specialist heads, 50,000 bootstrap samples,
200,000 sign flips, and family-wise correction across all three comparisons.
The sample size was fixed with no early stopping.

Exact command:

```bash
python -m training.paired_checkpoint_eval \
  new2m=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  old800=kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00800768.pt \
  old_aug01=kaggle_runs/2026-08-01_223035_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  old_aug11=kaggle_runs/2026-08-11_231044_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --output local_evaluations/paired_compact_2m_vs_800k_and_17m.json \
  --device cpu --deals 4096 --batch 256 --block-size 128 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 --seed 20260814
```

Results (positive means the 2M candidate was stronger; W/L/D is the paired-deal
effect sign):

```text
opponent                         advantage    ordinary 95% CI      simultaneous 95% CI    W / L / D          Holm p
own 800,768-game ancestor        +36.136      [+32.119, +40.001]   [+31.230, +40.914]      55.69/42.35/1.97   0.000015
Aug-01, 1,728,512 games          +32.247      [+28.401, +36.080]   [+27.526, +36.918]      56.87/40.60/2.53   0.000015
Aug-11, 1,748,992 games          +31.508      [+28.823, +34.072]   [+28.229, +34.591]      56.82/40.47/2.71   0.000015
```

All three improvements are statistically significant at family-wise alpha
0.05. Most importantly, the same-lineage comparison isolates continued
training: from 800,768 to 2,001,920 games the model gained about 36.1 points
per round. The complete raw scores, deal seeds, block means, manifests, and
statistics are in
`local_evaluations/paired_compact_2m_vs_800k_and_17m.json`. Local evaluation
runtime was 464.6 seconds.

## 2026-08-14 — Direct 2M compact vs 14.79M standard mature baseline

The earlier 2M evaluation used mature *compact* references. To answer the
separate mature-baseline question, the 2,001,920-game compact checkpoint was
compared directly with the 14,790,656-game standard checkpoint. This is a
playing-strength comparison, not a causal architecture ablation: architecture
and training budget both differ.

```bash
python -m training.paired_checkpoint_eval \
  compact2m=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  standard_mature=kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt \
  --output local_evaluations/paired_compact_2m_vs_standard_mature_14m.json \
  --device cpu --deals 4096 --batch 256 --block-size 128 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 --seed 20260814
```

The fixed sample used 4,096 identical deal seeds in both team orientations
(8,192 rounds), 32 independent 128-deal blocks, and every deployed specialist
head. Positive advantage denotes compact strength.

```text
compact score advantage:    -10.377 points/round
ordinary/simultaneous CI:   [-14.548, -6.318]
paired-deal W / L / D:      47.01% / 50.54% / 2.45%
two-sided sign-flip p:      0.000025
statistically significant:  yes
```

Thus the 2M compact model is decisively better than the mature 1.7M compact
lineages, but remains about 10.4 points per round behind the much larger
14.79M-game standard mature model. Full raw results are in
`local_evaluations/paired_compact_2m_vs_standard_mature_14m.json`.

## 2026-08-14 — Recent compact learning slope / plateau check

To determine whether the compact lineage had actually plateaued, the final
2,001,920-game checkpoint was compared directly against its own 1.8M, 1.6M,
and 1.4M checkpoints. Each comparison used 4,096 deal seeds in both team
orientations (8,192 rounds), all specialist heads, 32 independent blocks, and
family-wise correction across the three fixed-sample comparisons.

```bash
python -m training.paired_checkpoint_eval \
  compact2m=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  compact1_8m=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_01800192.pt \
  compact1_6m=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_01600512.pt \
  compact1_4m=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_01400832.pt \
  --output local_evaluations/paired_compact_2m_recent_learning_curve.json \
  --device cpu --deals 4096 --batch 256 --block-size 128 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 --seed 20260814
```

```text
opponent games   games since opponent   2M advantage   simultaneous 95% CI   Holm p
1,800,192             201,728              +4.325       [+0.526, +8.224]       0.0120
1,600,512             401,408             +12.529       [+8.244, +17.034]      0.000015
1,400,832             601,088             +12.609       [+8.615, +16.394]      0.000015
```

The lineage is not fully plateaued: even the final ~202k games produced a
small but family-wise-significant gain. As a rough rate comparison, the final
interval gained 4.29 points per 200k games, versus 6.02 points per 200k averaged
over the direct 800,768-to-2,001,920 comparison, about a 29% slowdown in
marginal improvement. Treat that percentage as descriptive rather than a
parametric learning-curve fit. Head-to-head checkpoint strength is not exactly
transitive, and the three cumulative effects used independent deal sets, so
the 1.4M and 1.6M point estimates should not be subtracted to manufacture an
unmeasured intermediate interval. The defensible summary is "diminishing
returns, but still measurably learning," not "plateaued."

Full raw results:
`local_evaluations/paired_compact_2m_recent_learning_curve.json`.

## 2026-08-15 — Second compact continuation pushed from 2.002M

After the Kaggle quota reset, kernel version 8 was pushed at 07:38:22 CEST and
reported `KernelWorkerStatus.RUNNING` on the immediate post-push check.

```text
kernel:          <kaggle-user>/tichu-compact-scratch
local run id:    kaggle_runs/2026-08-15_073817_compact-scratch
accelerator:     NvidiaTeslaT4
resume games:    2,001,920
resume source:   kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
resume sha256:   013d48893ddf6a4da830571fb097c39d8958387abc044f100d88abbb81f8b34f
resume dataset:  <kaggle-user>/tichu-resume-compact-scratch
max wall time:   11.7 h
reinit GT:       no
UI:              https://www.kaggle.com/code/<kaggle-user>/tichu-compact-scratch
```

Exact command:

```bash
bash kaggle/run.sh compact-scratch --resume \
  kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
```

The generated resume bundle retains memory microbatch 512, effective Adam
batch 2,048, four synchronous rollout workers, the nonlinear Phoenix-aware
action encoder, static attention, specialist heads, and the conditional CUDA
worker-cache guard. Its baked resume contract aborts if the attached
checkpoint cannot be found rather than silently starting fresh.

## 2026-08-16 — Version 8 compact continuation fetched at 3.262M

Kernel version 8 completed normally at the 11.7-hour wall-clock cap and was
downloaded with:

```bash
bash kaggle/run.sh --fetch compact-scratch
```

```text
run:                  kaggle_runs/2026-08-15_073817_compact-scratch
files / disk:         209 files / 764 MiB
periodic checkpoints: 126
resume games:         2,001,920
final games:          3,262,464 (latest.pt interrupt checkpoint)
games added:          1,260,544 across 1,231 ordinary PPO updates
median throughput:    35.5 games/s
harmonic throughput:  35.42 games/s
median PPO time:      14.94 s/update
final entropy:        0.22864
final logged win:     46.0% (self-play diagnostic, not arena strength)
latest.pt sha256:     a555a2d900e07659278f4604bf13e53e0f8926f822ac1aeaf0bbe4d75b5035bf
```

The last periodic file is `ckpt_03260416.pt`; the clean wall-cap handler then
saved the newer 3,262,464-game state to `latest.pt`. There was no OOM,
traceback, or live-growth alert. The conditional worker-cache guard performed
six pressure-triggered cleanups during the full production leg.

## 2026-08-16 — 3.262M compact strength and plateau evaluation

The fetched 3,262,464-game checkpoint was evaluated locally against its own
3,060,736-game recent checkpoint, its 2,001,920-game parent, and the mature
14,790,656-game standard model. Each comparison used 4,096 fixed deal seeds in
both team orientations (8,192 rounds), all specialist heads, 32 independent
128-deal blocks, 50,000 bootstrap samples, 200,000 sign flips, and family-wise
correction across the three fixed-sample comparisons.

```bash
python -m training.paired_checkpoint_eval \
  compact3_26m=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  compact3_06m=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03060736.pt \
  compact2m=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  standard_mature=kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt \
  --output local_evaluations/paired_compact_3_26m_eval.json \
  --device cpu --deals 4096 --batch 256 --block-size 128 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 --seed 20260816
```

```text
opponent                         advantage    ordinary 95% CI      simultaneous 95% CI    W / L / D          Holm p       significant
own 3,060,736-game checkpoint     +3.140      [-0.770, +7.205]     [-1.643, +8.125]       51.17/47.30/1.53   0.1379       no
own 2,001,920-game parent        +24.749      [+19.796, +29.890]   [+18.706, +31.040]     54.02/44.40/1.59   0.000015     yes
standard mature, 14,790,656       +6.987      [+3.295, +10.667]    [+2.488, +11.476]      51.06/47.08/1.86   0.001810     yes
```

This leg added a decisive 24.75 points per round over its exact 2M parent and
the compact lineage now significantly beats the standard mature baseline by
about 6.99 points per round. The final ~202k games have a positive point
estimate but no statistically detectable gain after family-wise correction,
which is the first strong evidence that the lineage is approaching a practical
plateau. This does not prove exact equivalence between 3.061M and 3.262M; the
simultaneous interval still permits roughly -1.64 to +8.12 points per round.
Full raw results are in `local_evaluations/paired_compact_3_26m_eval.json`.

## 2026-08-16 — Compact learning curve at 250k-game intervals

To resolve the shape of the compact lineage rather than infer it from one
short late comparison, every available approximately 250k-game checkpoint was
compared directly with its immediate predecessor. The grid spans 250,880 to
3,250,176 games. Eleven intervals contain 249,856 games; the run-boundary
interval from 1,750,016 to 2,000,896 contains 250,880 games (one extra
1,024-game PPO update), so the evaluator used a 1,024-game spacing tolerance.
No pre-250k checkpoint exists in this lineage.

Each of the 12 comparisons used the same 8,192 fixed deal seeds in both team
orientations (16,384 rounds per interval), all specialist heads, 32 independent
256-deal-pair blocks, 50,000 block-bootstrap samples, and 200,000 block
sign-flips. In total the evaluation played 196,608 rounds in 3,677.369 seconds
(61.29 minutes) on CPU. The simultaneous confidence intervals use a Bonferroni
family-wise alpha of 0.05 across all 12 intervals, and the p-values use Holm
correction across the same family.

```bash
python -m training.checkpoint_curve_eval \
  c025=kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00250880.pt \
  c050=kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00500736.pt \
  c075=kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00750592.pt \
  c100=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_01000448.pt \
  c125=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_01250304.pt \
  c150=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_01500160.pt \
  c175=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_01750016.pt \
  c200=kaggle_runs/2026-08-14_084154_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_02000896.pt \
  c225=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_02250752.pt \
  c250=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_02500608.pt \
  c275=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_02750464.pt \
  c300=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03000320.pt \
  c325=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03250176.pt \
  --output local_evaluations/compact_250k_interval_curve.json \
  --device cpu --deals 8192 --batch 256 --block-size 256 \
  --spacing-tolerance 1024 --bootstrap-samples 50000 \
  --sign-flip-samples 200000 --seed 20260816
```

The advantage below is the later checkpoint minus the earlier checkpoint, in
points per round:

```text
games interval                 advantage    ordinary 95% CI       simultaneous 95% CI    Holm p       significant
  250,880 ->   500,736          +38.185      [+35.838, +40.446]    [+34.736, +41.413]     0.000060     yes
  500,736 ->   750,592          +17.856      [+15.440, +20.336]    [+14.396, +21.540]     0.000060     yes
  750,592 -> 1,000,448           +8.266      [ +6.029, +10.541]    [ +5.096, +11.631]     0.000060     yes
1,000,448 -> 1,250,304           +9.515      [ +6.866, +12.059]    [ +5.616, +13.142]     0.000060     yes
1,250,304 -> 1,500,160          +10.131      [ +8.250, +12.032]    [ +7.333, +12.948]     0.000060     yes
1,500,160 -> 1,750,016           +1.854      [ -0.399,  +4.067]    [ -1.541,  +5.087]     0.248549     no
1,750,016 -> 2,000,896           +5.716      [ +3.535,  +7.877]    [ +2.471,  +8.864]     0.000150     yes
2,000,896 -> 2,250,752          +11.959      [ +9.885, +13.946]    [ +8.832, +14.800]     0.000060     yes
2,250,752 -> 2,500,608           -2.209      [ -4.496,  +0.053]    [ -5.577,  +1.122]     0.207779     no
2,500,608 -> 2,750,464          +10.997      [ +9.305, +12.780]    [ +8.588, +13.608]     0.000060     yes
2,750,464 -> 3,000,320           +1.991      [ -0.587,  +4.778]    [ -1.704,  +6.145]     0.248549     no
3,000,320 -> 3,250,176           +4.644      [ +2.134,  +7.142]    [ +0.918,  +8.276]     0.005200     yes
```

The curve learns very quickly through 750k, continues with steady roughly
8–10 point gains per 250k through 1.5M, and then becomes bursty rather than
smoothly plateauing. Nine of 12 intervals are significantly positive after
family-wise correction; three are statistically flat, and none is
significantly negative. In particular, 3.000M to 3.250M is still significantly
positive, so the prior inference from the lower-powered 3.061M-to-3.262M
comparison should be revised: late learning is slower and intermittent, but
there is not yet evidence of a sustained plateau at 3.25M.

Summing adjacent advantages gives a useful descriptive trajectory from 0.0 at
250,880 games to +118.905 at 3,250,176 games, but this is not an additive Elo
scale and must not be interpreted as a direct endpoint matchup because
self-play policies can be non-transitive. Full raw scores, shared deal seeds,
block means, checkpoint metadata, and adjusted statistics are in
`local_evaluations/compact_250k_interval_curve.json`.

## 2026-08-20 — Fixed-state PPO replay/policy-drift diagnostics

A reusable diagnostic was added as `training/replay_policy_drift.py`. A
behavior checkpoint generates fixed deployment self-play states, sampled
actions, histories, and complete legal-action sets. Later checkpoints then
rescore those exact states. This measures the actual PPO sampled-action
importance ratios and clipping behavior as well as exact full-distribution
KL, Jensen-Shannon divergence, total variation, largest single-action
probability movement, selected-action movement, and argmax changes. The raw
NPZ retains every legal-action vector and every checkpoint's probability for
every action in every state, while JSON includes the ten most-diverged states
per age with human-readable actions.

Both powered runs used eight independently seeded 256-game rollout blocks
(2,048 games), 32-game inference/bootstrap blocks, 50,000 block-bootstrap
draws, all deployment specialist heads during collection, and fixed states
across checkpoint ages. Simultaneous intervals are Bonferroni-corrected across
all checkpoint ages times all nine primary metrics, separately for the
all-state and discretionary-state tables. Approximately one third of main-play
decisions have only one legal action and therefore cannot drift; the
discretionary-state table (`legal_lens > 1`) is the primary interpretation.

### Current-policy two-update measurement

The final interrupt checkpoint is exactly two ordinary 1,024-game PPO updates
after the last periodic checkpoint: 3,260,416 to 3,262,464 games. The exact
command was:

```bash
python -m training.replay_policy_drift \
  base3260=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03260416.pt \
  latest3262=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --output local_evaluations/compact_current_2update_replay_policy_drift.json \
  --raw-output local_evaluations/compact_current_2update_replay_policy_drift_raw.npz \
  --device cpu --batch 256 --threads 6 \
  --rollout-blocks 8 --games-per-rollout 256 --games-per-inference-block 32 \
  --bootstrap-samples 50000 --alpha 0.05 --clip-eps 0.2 \
  --top-states 10 --seed 20260821
```

It produced 153,814 decisions, of which 100,466 (65.32%) were discretionary,
and completed in 125.721 seconds. On discretionary states after two updates:

```text
metric                                      estimate     simultaneous 95% CI
outside [0.8, 1.2] ratio range              10.359%      [10.078%, 10.630%]
PPO objective actually saturated             5.189%      [ 4.999%,  5.370%]
mean full-distribution total variation        3.530%      [ 3.474%,  3.585%]
argmax action changed                         3.950%      [ 3.782%,  4.122%]
chosen-action absolute probability change     2.991 pp    [ 2.942,   3.038] pp
behavior-to-latest KL                         0.01544      [ 0.01484, 0.01604]
```

The raw importance-ratio effective-sample-size fraction was 96.26%. Ratio
quantiles were p01=0.5233, p05=0.7847, median=1.0000, p95=1.1932,
p99=1.6201, p99.9=2.8862, with maximum 11.2835. The median discretionary
state moved only 0.845% in total variation, p95 moved 15.05%, and p99 moved
27.63%. Rare outliers nevertheless exist: the most-diverged two-update state
moved 95.12% of its action probability mass and flipped from an almost-certain
Dog lead to an almost-certain eight-card combination.

Including forced actions dilutes the aggregate to 6.766% outside the clip
range, 3.389% PPO-saturated, 2.306% mean total variation, and 2.580% argmax
changes. The strong two-update result supports a small recent replay window:
the current compact policy usually changes little across two updates, and raw
importance weights remain well behaved in aggregate.

### Longer late-training age sweep

The late-lineage sweep used fixed 3,000,320-game behavior states and exact
available checkpoint ages of 10, 20, 39, 78, 156, 235, and 256 ordinary PPO
updates:

```bash
python -m training.replay_policy_drift \
  base3m=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03000320.pt \
  age10=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03010560.pt \
  age20=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03020800.pt \
  age39=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03040256.pt \
  age78=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03080192.pt \
  age156=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03160064.pt \
  age235=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_03240960.pt \
  age256=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  --output local_evaluations/compact_late_replay_policy_drift.json \
  --raw-output local_evaluations/compact_late_replay_policy_drift_raw.npz \
  --device cpu --batch 256 --threads 6 \
  --rollout-blocks 8 --games-per-rollout 256 --games-per-inference-block 32 \
  --bootstrap-samples 50000 --alpha 0.05 --clip-eps 0.2 \
  --top-states 10 --seed 20260820
```

It produced 151,372 decisions, of which 102,146 (67.48%) were discretionary,
and completed in 377.333 seconds. Primary discretionary-state results:

```text
age       outside clip   PPO saturated   mean TV   TV p95   argmax flip   chosen |dp|   raw-ratio ESS
  10 upd       27.931%         14.108%     11.149%    47.05%       12.631%       9.490 pp       17.204%
  20 upd       29.548%         14.869%     12.087%    50.88%       13.754%      10.259 pp       27.798%
  39 upd       33.029%         16.703%     14.785%    62.04%       16.603%      12.532 pp       14.523%
  78 upd       34.151%         17.098%     15.800%    66.55%       17.749%      13.330 pp       19.620%
 156 upd       39.187%         19.670%     20.339%    80.19%       22.842%      17.186 pp       13.735%
 235 upd       40.977%         20.645%     22.241%    85.40%       24.726%      18.815 pp        8.516%
 256 upd       41.869%         21.059%     22.906%    85.79%       25.667%      19.400 pp        5.886%
```

At ten updates the simultaneous 95% intervals are [27.372%, 28.455%] for
outside-clip fraction and [10.929%, 11.342%] for mean total variation. At 256
updates they are [41.306%, 42.393%] and [22.596%, 23.231%], respectively.

The raw importance-ratio ESS is deliberately reported but is tail-sensitive
and non-monotone: at ten updates its p99 ratio is only 2.923, but one sampled
action has ratio 617.31, collapsing raw ESS to 17.2%. At 256 updates p99 is
4.535, p99.9 is 21.13, maximum is 995.73, and raw ESS is 5.89%. PPO clipping
makes those tails safer, which is why the actual same-direction saturated
fraction is much smaller than the generic outside-range fraction.

Interpretation: the policy is highly stable over the directly measured two
updates, so recent replay is not ruled out and is now a credible experiment.
By ten updates, however, roughly 28% of discretionary sampled actions are
outside PPO's nominal ratio range and rare importance weights already become
pathological. This supports testing a one- or two-update replay queue, not a
large replay buffer. This diagnostic measures replay viability/staleness; it
does not itself show that training with replay improves final strength or
wall-clock learning efficiency.

Behavior rescoring passed its reconstruction guard in both runs. Re-batching
changes floating-point reduction order: p99 absolute stored-versus-rescored
log-probability errors were 1.57e-5 and 1.41e-5; worst rare errors were 0.0143
and 0.0145, still over an order of magnitude inside PPO's log clipping
boundaries. Full summaries and state examples are in the two JSON files; the
full action distributions are in their corresponding raw NPZ files.

### Replicated production-update ages 2 through 10

The historical sweep above conditions on one realized training trajectory and
therefore cannot estimate update-path variance. A new experiment
(`training/replay_age_branch_experiment.py`) started 12 independently seeded
training continuations from the same mature 3,262,464-game checkpoint. Every
branch performed ten real production-sized 1,024-game PPO updates, with memory
batch 512, effective Adam batch 2,048, and two epochs. The policy after every
integer age from 2 through 10 was scored on the same held-out 1,024-game
rollout. Inference used 32 independent 32-game state blocks. The 50,000-draw
crossed bootstrap resampled both training branches and state blocks, and its
simultaneous intervals use family-wise alpha 0.05 over 9 ages x 9 metrics (81
cells). This supplies genuine training-path replication while remaining
conditional on one starting checkpoint.

The private T4 run was launched with:

```bash
bash kaggle/run.sh replay-age \
  --resume kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
```

The embedded experiment command was:

```bash
python -m training.replay_age_branch_experiment \
  --checkpoint /kaggle/input/datasets/<kaggle-user>/tichu-resume-replay-age/latest.pt \
  --output /kaggle/working/tichubot_outputs/replay_age/statistical_replay_age_curve.json \
  --raw-output /kaggle/working/tichubot_outputs/replay_age/statistical_replay_age_curve_raw.npz \
  --device cuda --branches 12 --updates 10 \
  --ages 2,3,4,5,6,7,8,9,10 \
  --games-per-update 1024 --batch 512 --optimizer-batch 2048 \
  --epochs 2 --workers 4 --threads 4 \
  --eval-games 1024 --eval-games-per-block 32 --eval-batch 256 \
  --bootstrap-samples 50000 --alpha 0.05 --clip-eps 0.2 \
  --seed 20260824
```

The run completed all 12 branches in 4,284.109 seconds (71.40 minutes). It
trained on 9.189 million main-policy decisions in total and evaluated 77,339
fixed decisions, 50,120 of them discretionary. Behavior rescoring passed:
p99 absolute stored-versus-rescored log-probability error was 1.42e-5 and the
worst row was 0.00692. Primary discretionary-state results are:

```text
age  outside clip       PPO saturated      mean TV             argmax changed      chosen |dp|       ratio ESS
 2    9.922% [9.171,10.622]  5.009% [4.576,5.444]  3.377% [3.174,3.577]  3.840% [3.538,4.123]  2.883 pp [2.706,3.057]  96.88%
 3   12.371% [11.412,13.327] 6.269% [5.753,6.738]  4.103% [3.816,4.363]  4.660% [4.306,5.010]  3.504 pp [3.259,3.716]  95.46%
 4   14.224% [13.147,15.356] 7.187% [6.627,7.732]  4.738% [4.387,5.085]  5.378% [4.951,5.816]  4.046 pp [3.748,4.326]  93.55%
 5   16.014% [14.780,17.198] 8.099% [7.460,8.725]  5.343% [4.889,5.774]  6.072% [5.545,6.573]  4.562 pp [4.174,4.923]  91.46%
 6   17.272% [15.966,18.441] 8.726% [8.021,9.314]  5.816% [5.348,6.265]  6.588% [6.022,7.088]  4.970 pp [4.578,5.337]  89.60%
 7   18.326% [17.168,19.482] 9.267% [8.654,9.842]  6.228% [5.790,6.693]  7.073% [6.545,7.594]  5.322 pp [4.946,5.712]  87.56%
 8   19.231% [18.090,20.382] 9.734% [9.131,10.316] 6.598% [6.133,7.080]  7.464% [6.921,8.016]  5.637 pp [5.232,6.053]  86.10%
 9   20.250% [19.069,21.487] 10.250% [9.635,10.854] 7.017% [6.510,7.565]  7.952% [7.375,8.579]  5.994 pp [5.563,6.468]  84.51%
10   21.073% [19.831,22.359] 10.686% [10.046,11.333] 7.388% [6.872,7.950] 8.374% [7.764,9.056]  6.309 pp [5.878,6.801]  81.05%
```

Brackets are the 81-cell simultaneous 95% intervals. Every adjacent measured
age increment was positive even under that simultaneous correction, so drift
has not plateaued by age ten. It is nevertheless strongly concave. From the
third update to the tenth, the per-update outside-clip increase falls from
2.450 percentage points [2.061, 2.921] to 0.823 [0.194, 1.245]; actual PPO
saturation falls from +1.260 pp [1.018, 1.482] to +0.435 [0.110, 0.649]; and
mean TV falls from +0.726 pp [0.604, 0.848] to +0.371 [0.138, 0.547]. The
production pass head updates after main-policy update eight on every branch;
the small age-nine marginal rebound is therefore production-faithful but
cannot be attributed solely to ordinary main-policy drift.

There is no sharp empirical cliff between ages 2 and 10. Age two loses only
5.01% of sampled-action PPO terms to same-direction clipping; age ten loses
10.69%, while the mean distribution moves 7.39% in total variation and the
argmax changes in 8.37% of discretionary states. A descriptive 10% saturation
threshold puts the last point-estimate-safe age at eight, or age seven if the
simultaneous upper interval must remain below 10%. This is a useful replay or
latent-dreaming horizon baseline, not evidence that replay improves playing
strength; that still requires a controlled training comparison.

Raw importance-weight tails remain relevant despite reasonable aggregate ESS.
At age ten mean branch ESS is 81.05%, but branches range from 55.25% to 87.75%
and one rare sampled action reaches ratio 176.85. The mean ratio remains close
to one at every age. PPO clipping contains these rare tails, explaining why
actual objective saturation remains around half the generic outside-range
rate.

Downloaded results, manifest, raw bootstrap draws, and the kernel log are in
`kaggle_runs/2026-08-20_131405_replay-age`. The private kernel is
`<kaggle-user>/tichu-replicated-replay-age-curve`.

## 2026-08-20 — Normal mature compact continuation (Kaggle version 9)

After completing the replay-age diagnostics, ordinary production training was
resumed from the mature compact checkpoint without any replay or dreaming
changes. The source is the 3,262,464-game `latest.pt` from the 2026-08-15 leg.

```bash
bash kaggle/run.sh compact-scratch --resume \
  kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
```

```text
kernel:             <kaggle-user>/tichu-compact-scratch
kernel version:     9
accelerator:        NvidiaTeslaT4
pushed:             2026-08-20T15:41:25+02:00
saved run:          kaggle_runs/2026-08-20_154119_compact-scratch
source games:       3,262,464
wall-clock limit:   11.7 hours (graceful checkpoint/stop)
games/update:       1,024
workers:            4, synchronous CUDA rollout
memory batch:       512
Adam batch:         2,048
PPO epochs:         2
GT head reinit:     no
status after push:  RUNNING
```

Fetch with `bash kaggle/run.sh --fetch compact-scratch`. Evaluate the new
checkpoint against its own 3,262,464-game parent with a powered paired arena
before deciding whether this lineage has plateaued; training telemetry alone
does not establish a strength plateau.

### Final training result

Version 9 completed normally at its 11.7-hour wall-clock cap and was fetched
to `kaggle_runs/2026-08-20_154119_compact-scratch` on 2026-08-21. There was no
traceback or OOM. The graceful interrupt checkpoint is:

```text
tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
final games:          4,538,368
new games this leg:   1,275,904 (1,246 ordinary 1,024-game updates)
new transitions:      94,776,181
periodic checkpoints: 127
downloaded size:      771 MiB
```

Across all 1,246 logged updates, median throughput was 35.9 games/s, median
PPO time was 14.494 seconds, median approximate KL was 0.00424, median clip
fraction was 3.326%, median policy entropy was 0.22687, median reported GPU
utilization was 87%, and median one-minute CPU load was 2.26. The final row at
4,538,368 games reported 33.7 games/s, PPO 15.790 seconds, KL 0.00468, clip
fraction 3.447%, entropy 0.23378, and return -0.0339. These are healthy
telemetry values, but they do not answer whether playing strength improved;
that requires the planned paired evaluation against the 3,262,464-game
parent.

## 2026-08-21 — Checkpoint prune and 4.538M strength evaluation

Dense periodic checkpoints had grown `kaggle_runs/` to 24 GiB. A conservative
resolved prune deleted 2,699 unreferenced `ckpt_*.pt` files from runs before
the 2026-08-20 leg and reclaimed 21.758 GiB, reducing the directory to 1.3
GiB. Every `latest.pt`/`final.pt`, all 24 periodic compact checkpoints named by
existing research/evaluation records, the mature standard endpoint, and all
127 periodic checkpoints in the new 4.538M run were retained. The deleted
local periodic copies are not locally recoverable.

The usual fixed-sample paired drill compared the 4,538,368-game endpoint with
the recent 4,300,800-game checkpoint, its exact 3,262,464-game parent, and the
14,790,656-game mature standard baseline:

```bash
python -m training.paired_checkpoint_eval \
  compact4_538m=kaggle_runs/2026-08-20_154119_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  compact4_300m=kaggle_runs/2026-08-20_154119_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_04300800.pt \
  compact3_262m=kaggle_runs/2026-08-15_073817_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  standard_mature=kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt \
  --output local_evaluations/paired_compact_4_538m_eval.json \
  --device cpu --deals 4096 --batch 256 --block-size 128 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 \
  --seed 20260821
```

Each comparison used 4,096 shared deals in both team orientations (8,192
rounds), all specialist heads, 32 independent 128-deal blocks, 50,000
bootstrap draws, 200,000 sign flips, and family-wise correction across the
three comparisons. The complete evaluation took 512.344 seconds.

```text
opponent                  advantage    ordinary 95% CI      simultaneous 95% CI    W / L / D          Holm p       significant
own 4,300,800 checkpoint   +3.107      [-2.699, +8.633]     [-3.989, +9.781]       49.90/49.00/1.10   0.294754     no
own 3,262,464 parent      +10.175      [+6.067,+14.056]     [+5.049,+14.911]       50.57/48.05/1.38   0.000090     yes
standard mature          +15.151      [+10.419,+20.042]    [+9.444,+21.145]       52.36/45.62/2.03   0.000015     yes
```

The 1.276M-game continuation produced a statistically clear +10.17 points per
round over its exact parent, so the lineage was not plateaued across the full
leg. The last 237,568 games have only a +3.11 point estimate and are not
distinguishable from ordinary variance after family-wise correction. This is
evidence of a recent practical plateau, but not proof of equivalence: the
simultaneous interval still permits approximately -3.99 to +9.78 points per
round. The compact endpoint's advantage over the mature standard baseline has
grown to +15.15 points per round. Full deal scores and block effects are in
`local_evaluations/paired_compact_4_538m_eval.json`.

### High-precision recent-slope follow-up

The 4,096-deal recent comparison above was used only as a variance pilot. To
resolve the final 237,568-game interval with a 95% confidence half-width near
one point, a fresh fixed sample compared only the 4,538,368 endpoint with its
own 4,300,800 checkpoint:

```bash
python -m training.paired_checkpoint_eval \
  compact4_538m=kaggle_runs/2026-08-20_154119_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  compact4_300m=kaggle_runs/2026-08-20_154119_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_04300800.pt \
  --output local_evaluations/paired_compact_4_538m_vs_4_300m_precision.json \
  --device cpu --deals 147456 --batch 256 --block-size 512 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 \
  --seed 20260822
```

The test used 147,456 fresh shared deals in both team orientations (294,912
rounds), all specialist heads, 288 independent 512-deal blocks, 50,000 block
bootstrap draws, 200,000 block sign flips, and no early stopping. It completed
in 5,329.901 seconds (88.83 minutes).

```text
interval                    advantage    bootstrap 95% CI     W / L / D          two-sided p
4,300,800 -> 4,538,368       +4.020      [+3.312, +4.723]     50.09/48.68/1.23   0.000005
```

The achieved interval has a 0.706-point average half-width, tighter than the
requested approximately one point. This supersedes the low-powered recent
plateau interpretation above: the final 237,568 games produced a real positive
gain. Together with the prior 250k-interval curve, the defensible conclusion is
that learning continues but is bursty and slower than early training. It is
not literally positive at every measured checkpoint interval, but there is no
evidence of a sustained plateau through 4.538M games. Full deal scores, block
effects, seeds, checkpoint metadata, and statistics are in
`local_evaluations/paired_compact_4_538m_vs_4_300m_precision.json`.

## 2026-08-21 — Compact continuation from 4.538M (Kaggle version 10)

The high-precision recent-slope evaluation established that the lineage was
still improving, so ordinary production training was continued without an
architecture, optimizer, replay, or schedule change:

```bash
bash kaggle/run.sh compact-scratch --resume \
  kaggle_runs/2026-08-20_154119_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt
```

```text
kernel:             <kaggle-user>/tichu-compact-scratch
kernel version:     10
accelerator:        NvidiaTeslaT4
pushed:             2026-08-21T22:11:23+02:00
saved run:          kaggle_runs/2026-08-21_221118_compact-scratch
source games:       4,538,368
wall-clock limit:   11.7 hours (graceful checkpoint/stop)
games/update:       1,024
workers:            4, synchronous CUDA rollout
memory batch:       512
Adam batch:         2,048
PPO epochs:         2
GT head reinit:     no
status after push:  RUNNING
```

Immediately before launch, `kaggle quota` reported 5.37 of 30.00 weekly GPU
hours remaining and `refreshAt = 2026-08-22T00:00:00` UTC, which is 02:00 CEST.
The kernel began about 3 hours 49 minutes before that refresh. Fetch after
completion with `bash kaggle/run.sh --fetch compact-scratch`.

### Final training result and standard evaluation

Version 10 completed normally at its 11.7-hour cap and was fetched to
`kaggle_runs/2026-08-21_221118_compact-scratch` on 2026-08-22. There was no
traceback, OOM, live-growth alert, or pressure-triggered worker-cache cleanup.

```text
final games:          5,796,864
new games this leg:   1,258,496 (1,229 ordinary 1,024-game updates)
new transitions:      94,849,765
periodic checkpoints: 126
downloaded size:      765 MiB (212 files)
```

Across all 1,229 updates, median throughput was 35.4 games/s, median PPO time
was 15.015 seconds, median approximate KL was 0.00397, median clip fraction was
3.110%, median policy entropy was 0.22171, median GPU utilization was 88%, and
median one-minute CPU load was 2.24. The final update reported 37.2 games/s,
PPO 13.907 seconds, KL 0.00385, clip fraction 3.061%, entropy 0.21683, and mean
return -0.0321.

The standard fixed-sample paired drill compared the endpoint against its
latest approximately 250k-game checkpoint, its exact 4,538,368-game parent,
and the mature standard baseline:

```bash
python -m training.paired_checkpoint_eval \
  compact5_797m=kaggle_runs/2026-08-21_221118_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  compact5_550m=kaggle_runs/2026-08-21_221118_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_05550080.pt \
  compact4_538m=kaggle_runs/2026-08-20_154119_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt \
  standard_mature=kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt \
  --output local_evaluations/paired_compact_5_797m_eval.json \
  --device cpu --deals 4096 --batch 256 --block-size 128 \
  --bootstrap-samples 50000 --sign-flip-samples 200000 \
  --seed 20260823
```

Each comparison used 4,096 fresh shared deals in both team orientations (8,192
rounds), all specialist heads, 32 independent 128-deal blocks, 50,000 block
bootstrap draws, 200,000 sign flips, and family-wise correction across the
three comparisons. The evaluation completed in 486.907 seconds.

```text
opponent                  advantage    ordinary 95% CI      simultaneous 95% CI    W / L / D          Holm p       significant
own 5,550,080 checkpoint   +5.145      [-0.056,+10.143]     [-1.248,+11.117]       49.82/48.52/1.66   0.059120     no
own 4,538,368 parent       +5.696      [+1.245,+10.210]     [+0.280,+11.217]       50.26/48.38/1.37   0.040070     yes
standard mature          +19.612      [+15.552,+23.694]    [+14.650,+24.585]      52.95/44.92/2.12   0.000015     yes
```

The full 1.258M-game leg produced a family-wise-significant gain over its exact
parent, so learning continued through 5.797M games. The final 246,784-game
slice has a substantial positive point estimate but narrowly misses the
corrected threshold; it is unresolved by this standard-size drill rather than
evidence of a plateau. The endpoint is also decisively stronger than the
mature standard model. Full scores, seeds, block effects, and statistics are
in `local_evaluations/paired_compact_5_797m_eval.json`.

## 2026-09-23 — Codebase audit remediation

Completed the 20 grouped findings T1–T4, D1–D4, I1–I5, R1–R7 in
`CODEBASE_AUDIT.md`. Three Herdr `gpt-5.6-sol` agents handled game rules,
interfaces, and distribution; training/worker/checkpoint reasoning stayed in
the main session. Existing unrelated working-tree changes were preserved;
no commits were made.

Key changes: immutable async policy snapshots; architecture/device-aware
worker reconstruction; safe GT optimizer migration; atomic invalid game and
advisor actions; complete normal-rank Phoenix substitutions; phase/wish rules;
Dragon history numbering and PettingZoo wishing; model-aware CLI inference;
known Mahjong passing provenance; runtime/wheel packaging; resume guards.
History observation scaling is unchanged; only its declared bound was widened.

Final integration command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **373 passed, 1 skipped in 13.85 seconds**; whitespace check clean.
The skipped worker test requires unavailable CUDA. The suite also builds and
imports an offline wheel and staged lightweight runtime in temporary directories.
No training, strength evaluation, checkpoint modification, remote Kaggle work,
native desktop build, or live browser validation was performed. These results
establish regression coverage, not a model-strength comparison. Legal-action
corrections mean historical evaluations used a different engine; rerun controlled
comparisons before attributing any strength change to these fixes.

## 2026-09-23 — Follow-up bug fixes

Continued the audit and fixed six additional defects (T5–T6, R8–R9, I6–I7 in
`CODEBASE_AUDIT.md`): fixed-policy Mahjong wishes and zero-game worker allocation;
invalid Dragon recipients; inactive-player action masks; advisor reconciliation
that violated known passing ownership or hand counts; negative HTTP body lengths.
The two rollout regressions failed before the implementation fixes. The advisor
and rules agents reproduced their respective failures and added behavioral tests.

Final validation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **380 passed, 1 skipped in 13.45 seconds**; diff check clean. CUDA remains
unavailable. No training, strength evaluation, checkpoint changes, remote work,
or live HTTP/browser validation was performed. Changes remain uncommitted and
existing unrelated work was preserved.

## 2026-09-23 — Passing-worker and pending-wish fixes

Fixed T7–T9 and I8 from `CODEBASE_AUDIT.md`. The standalone passing collector
now preserves the loaded static value function across worker processes instead
of reconstructing it from unrelated trainer settings, infers specialist shapes
from dispatched weights, omits zero-deal worker jobs, and encodes 196/200-feature
observations according to the value policy. The existing static value-delta
reward definition is unchanged. The advisor now rejects the next opponent play
until a pending Mahjong wish is recorded.

Six passing regression cases reproduced failures before the repair. The new
seven-test passing suite includes exact value comparisons for fixed/embed/history
models and a real two-worker spawn-pool round trip with custom specialist heads.

Final command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **388 passed, 1 skipped in 14.30 seconds**; diff check clean. The skipped
check requires unavailable CUDA. No training, model-strength experiment,
checkpoint modification, remote action, or live browser validation was performed.
Existing unrelated work is preserved and changes remain uncommitted.

## 2026-09-23 — Dragon scoring and interface-state fixes

Fixed R10, I9, and I10 from `CODEBASE_AUDIT.md`. A bomb that captures Dragon now
keeps the trick and points in both ordinary completion and round-end scoring;
a winning Dragon still gives the trick to the opposing team. This matches page
2 of the publisher rules linked in the audit. Advisor alternative plays now
wait for the pending Phoenix choice to be resolved or cancelled. A round-ending
Mahjong no longer opens a wish prompt in either human or advisor play.

Final command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **394 passed, 1 skipped in 14.51 seconds**; diff check clean. CUDA remains
unavailable. No training, model-strength comparisons, checkpoint changes,
remote actions, or live browser validation occurred. The scoring correction
changes game outcomes; historical strength measurements were not rerun.
Changes remain uncommitted and existing unrelated work is preserved.

## 2026-10-15 — Advisor fixes, BEST_CHECKPOINT gate, ring league prepared

### Advisor (commit c3bf2f5)

Replaying random real games through both advisors found three dead ends,
all fixed with regressions (`tests/test_advisor_input.py`, `tests/test_web.py`):

- `force_play` enforced the Mahjong wish against the *guessed* opponent
  hand, rejecting real moves and re-prompting the same player. Control
  without the fix: 39/600 replays stuck; with it: 0/600 (CLI and web).
- Web advisor could not record an opponent's Dragon gift: 218/300 replays
  dead-ended. Now `give_dragon` records any giver in advisor mode.
- CLI typed Phoenix straights/full houses always took the lower meaning.

### Best checkpoint (commit 46d82bd)

`training/best_checkpoint.json` / `training.checkpoints.BEST_CHECKPOINT`
is the default for advisor/play/web/llm_play. Change it only with

```bash
python -m training.promote_best CANDIDATE.pt
```

(fresh 4,096-deal paired eval vs current best; promote iff mean >= 2 SE).
Current best: compact-scratch 5,796,864
(`kaggle_runs/2026-08-21_221118_compact-scratch/.../latest.pt`), promoted
from `local_evaluations/paired_compact_5_797m_eval.json`: +19.61 +- 2.11
points/round over the standard model, z = 9.27.

### Ring league (commits 1511027, e5c58cf, 9dd5a4f) — RUNNING since 2026-10-16 21:07 CEST

User design: four agents from compact checkpoints 250,880 / 500,736 /
750,592 / 800,768 (`kaggle_runs/2026-08-13_132440_compact-scratch/...`);
one pair trains while the other is frozen; trainees play (a,b), (a,a),
(b,b) vs (c,d), (c,c), (d,d); phases cycle through all six pair splits;
a phase ends at parity (pooled paired advantage >= 0 and no matchup >= 2 SE
behind, checked every 40,960 ring games after a 40,960 minimum) or at a
500,000-game cap. Phase-end trainees are screened against BEST_CHECKPOINT
(256 deals) and gated by `promote_best` if ahead. Each agent keeps the
lineage recipe (PPO, pass harvest, GT probes) and entropy schedule
(0.03 -> 0.01 over 2M) on its own self-play-equivalent game count.

training-server setup: env `~/envs/tichubot` (torch 2.14.0+cu130, pettingzoo
1.24.3, gymnasium 1.2.3, numpy 2.4.3), code in `~/tichubot` via
`git archive HEAD` (9dd5a4f), start checkpoints + best copied. Server test
suite: 420 passed (2 Kaggle-launcher tests need the workstation path).
Measured with 8 workers on the RTX 3080: 107 ring games/s; 512-deal
9-matchup parity check 59 s.

Launch (when the server is free):

```bash
cd ~/tichubot && mkdir -p ~/ring_runs && setsid nohup env OMP_NUM_THREADS=1 \
  ~/envs/tichubot/bin/python -m training.train_ring --out ~/ring_runs/ring1 \
  --device cuda --workers 8 > ~/ring_runs/ring1.log 2>&1 < /dev/null &
# graceful stop: kill -INT <pid>; resume: same command plus --resume
```

Matched control (approved, not launched): plain self-play from the 800,768
checkpoint under the same code, on Kaggle after the weekly GPU quota
refresh (2026-10-17T00:00Z; only 2.86 h left on 2026-10-15):

```bash
bash kaggle/run.sh compact-scratch --resume \
  kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/ckpt_00800768.pt
```

Comparison plan: strongest ring agent vs the control checkpoint at the same
total learner-equivalent games, paired 4,096 deals; both also vs
BEST_CHECKPOINT.

Started 2026-10-16 21:07 CEST on training-server via `~/ring_runs/launch_ring1.sh`
(code 9dd5a4f; out `~/ring_runs/ring1`, log `~/ring_runs/ring1.log`,
state `ring1/ring_state.json`, per-update metrics `ring1/ring_log.csv`,
parity checks `ring1/checks/`). First updates: 100-112 ring games/s,
~74k transitions per trainee per update, KL 0.004-0.005, clip 4-5%,
entropy 0.45 (agent0) / 0.41 (agent1). Resume after a stop:
`RESUME=--resume setsid nohup ~/ring_runs/launch_ring1.sh >> ~/ring_runs/ring1.log 2>&1 < /dev/null &`.

Restart 2026-10-17: run 1 died at 21:09:58 on 2026-10-16 (third update) with
a CUDA OOM: 8 GPU rollout workers held ~6.1 GB of the 10 GB RTX 3080 while
PPO needed 3.2 GB; the 2-update smoke test had not shown it. Crashed run
archived as `~/ring_runs/ring1_oom_crash_20261016`. Relaunched fresh
2026-10-17 11:31 CEST with `--workers 4` (98 g/s vs 112 with 8; CPU workers
x11 measured 80 g/s), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
an auto-resume launcher loop (<= 20 attempts), agents saved at every state
save (954f9c9) and the phase-end promotion screen on CPU (f463df2). Peak GPU
over the first 10 updates + parity check: 8.1 GB of 10.2 GB. Graceful
restart at 53,248 ring games to deploy f463df2; resumed 11:42 CEST.

Second OOM 2026-10-17 12:56 CEST at 393,216 ring games: the GT-probe
practice PPO update (2,000-game chunk, ~150k transitions) ran out of GPU
memory while the 4 GPU workers kept ~4.7 GB of cache (cleanup only fired
below 30% free). Fixed in 31cde8e: `TICHU_WORKER_CUDA_ALWAYS_RELEASE=1`
(set in the launcher) makes workers release their cache after every
collection, and the trainer empties its own cache before rollouts, probe
collections and parity checks. A throwaway stress run at production sizes
then passed 6 probe blocks, a parity check, a phase end with the CPU
promotion screen and the next phase; peak GPU 5.4 GB of 10.2 GB. Resumed
from the 372,736-game save at 13:09 CEST; throughput unchanged (~98-100 g/s).

Phase 0 complete 2026-10-17 13:46 CEST (all periodic stages now verified on
the live run): trainees agent0 (250k start) + agent1 (500k start) vs frozen
agent2/agent3 ended at the 500k cap after 503,808 ring games without parity.
Pooled paired deficit across the 9 matchups: -37.13/round at 40,960 games,
-17.07/round (worst matchup z -7.14) at the final check (512 deals per
matchup). Phase-end screen vs BEST_CHECKPOINT (256 deals): agent0 -109.88,
agent1 -68.32/round, so no promotion gate. Phase 1 (agent2 + agent3 train vs
frozen agent0/agent1) started at 507,904 ring games, ~100 g/s.

Recipe change 2026-10-17 15:56 CEST (user request: mix in older checkpoints
so the ring is less of a pure metagame): from ring game 1,052,672 (phase 3,
after its first 4,096 games) the trainer runs 34e641f. Each update draws 2
snapshots from a pool of start + phase-end checkpoints (10 at the switch,
+2 per phase) and plays 20% of games as trainee partnership vs a snapshot
team; parity still uses the 9 frozen-pair matchups only. Swapped at the
phase boundary by `~/ring_runs/swap_at_phase_end.sh` (graceful SIGINT +
relaunch). Throughput after the switch ~96 g/s (was ~100); GPU 1.7 GB
between phases. Phases 0-2 (ring games 0-1,048,576) used the pure
trainee-vs-frozen mix.

Phase results so far (pooled paired advantage of trainees over the frozen
pair at the final check; screen vs BEST_CHECKPOINT):
- phase 0, agents 0+1 vs 2+3: cap, 503,808 games, -17.07 (worst z -7.14);
  screen -109.9 / -68.3
- phase 1, agents 2+3 vs 0+1: parity at the 40,960 minimum, +18.0
  (worst z +0.34); screen -84.3 / -59.3
- phase 2, agents 0+2 vs 1+3: cap, 503,808 games, -0.58 (worst z -2.75);
  screen -81.1 / -73.7

Status 2026-10-17 20:31 CEST (new session; run had been unwatched since the
previous session ended): no crash; trainer up since the 15:56 swap, 0
errors, phase 6 at 2,220,032 ring games, ~93 g/s; snapshot pool 16.
Agent self-play-equivalent games: 625,657 / 824,312 / 927,753 / 1,029,122.
- phase 3, agents 1+3 vs 0+2: cap, 503,808 games, -3.06 (worst z -4.31);
  screen -88.7 / -69.9
- phase 4, agents 0+3 vs 1+2: parity, 368,640 games, +3.52 (worst z -0.94);
  screen -78.2 / -58.0
- phase 5, agents 1+2 vs 0+3: parity, 163,840 games, +4.39 (worst z -1.13);
  screen -76.3 / -55.3

### 2026-10-17 — Ring vs lineage strength curve

All points vs BEST_CHECKPOINT (lineage 5,796,864), 1,024 shared deals in both
seat orders per point (same deal seeds for every point), current rules, CPU,
10 processes, ~7.5 min:

```bash
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G \
  python training/ring_strength_curve.py \
  local_evaluations/strength_curve_ring_vs_lineage_20261017.jsonl --deals 1024 --procs 10
python local_evaluations/ring_strength_page/build_strength_page.py \
  local_evaluations/strength_curve_ring_vs_lineage_20261017.jsonl OUT.html
```

Ring checkpoints copied to `ring_runs/ring1/agents/` (gitignored). Page:
a private page (private).
Points/round vs best (95% half-width ~7-9): lineage 250,880 -125.4;
500,736 -95.6; 750,592 -73.5; 800,768 -66.9; 1,000,448 -67.1; 1,250,304
-58.1; 1,500,160 -43.7; 2,000,896 -40.9; 2,500,608 -21.3; 3,000,320 -20.8;
4,538,368 -2.8; 5,550,080 -0.3. Ring: agent0 376,835 -102.8 / 502,777
-77.9 / 594,936 -78.0 / 646,136 -74.6; agent1 626,682 -78.2 / 752,632 -77.3
/ 793,592 -65.4 / 844,793 -75.5; agent2 760,831 -77.5 / 886,793 -68.6 /
927,753 -60.5; agent3 811,008 -72.6 / 936,961 -71.9 / 1,029,121 -64.4.
Reading: at matched own games the ring agents are level with or slightly
above the lineage (agent0 +17.7 at ~500k is clear, z ~3.2; agent2 +6.5 at
928k is within noise). At equal total training (ring 1.14M games beyond the
starts -> lineage 1.95M, about -41) the best ring agent (agent2, -60.5) is
~19 points/round behind. Lineage evaluated under post-2026-09-23 rules it
did not train on.

### 2026-10-17 — Ring play-style divergence (training/ring_diversity.py)

All checkpoints score the same 2,000 decision states (>= 2 legal moves,
sampled from fixed-seed self-play of BEST_CHECKPOINT); pairwise mean
Jensen-Shannon divergence (base 2) of move distributions and top-move
agreement. Agents reconstructed after every phase from ring_state.json.

```bash
OMP_NUM_THREADS=4 python -m training.ring_diversity \
  --state ring_runs/ring1/ring_state.json --agents-dir ring_runs/ring1/agents \
  --out local_evaluations/ring_diversity_20261017.json
python local_evaluations/ring_strength_page/build_strength_page.py \
  local_evaluations/strength_curve_ring_vs_lineage_20261017.jsonl OUT.html \
  local_evaluations/ring_diversity_20261017.json
```

Mean pairwise JS (agreement): start 0.100 (69.1%); after phase 0 0.087
(70.9%); phase 1 0.090; phase 2 0.091 (71.3%); phase 3 0.103; phase 4 0.106;
phase 5 0.105; now (2,371,584 ring games) 0.108 (70.0%). Scale: lineage
750k vs 800k = 0.057, 250k vs 800k = 0.117. Distance to best now 0.163-0.191.
Style now (start): bomb prob agent0 49.7% (32.8%), agent3 37.3% (50.1%);
pass prob agents 2/3 35.2%/37.1% (30.4%/33.2%), best 38.3%. Reading: no
collapse into one style; agents converged in phase 0, then drifted apart
again from phase 3 (the snapshot-mixing switch also happened then, so the
cause is not separable).

### 2026-10-18 — Ring strength update (phase 27, 6.0M ring games)

Same protocol and seeds as 2026-10-17 (appended to
`local_evaluations/strength_curve_ring_vs_lineage_20261017.jsonl` via
`training/ring_strength_curve.py --extra ...`). Run healthy: one process since
2026-10-17 15:56, 0 errors; phases 7-26 all ended at parity (40k-410k games);
no promotion. Agent games 1,294,335 / 1,359,854 / 1,286,156 / 1,356,801.
Points/round vs best (+-1.96 SE ~7.5-8.5): agent0 997,375 -56.4, 1,099,776
-49.5, 1,212,417 -43.6, 1,294,335 -50.9; agent1 1,083,377 -56.2, 1,206,256
-51.8, 1,349,614 -46.0, 1,359,854 -47.0; agent2 1,122,315 -55.2, 1,183,754
-57.0, 1,286,156 -41.1; agent3 1,152,000 -55.2, 1,285,121 -54.0, 1,346,562
-49.1, 1,356,801 -42.8.
Reading: every agent improved ~15-30 since 2026-10-17 (e.g. agent2 -60.5 ->
-41.1). At matched own games vs the lineage (interpolated: -55.8 at 1.29M,
-51.8 at 1.36M) agents are +4.7 (a0), +4.6 (a1), +14.9 (a2, z ~2.7), +9.0
(a3). At equal total compute (ring 2.99M beyond starts -> lineage ~3.79M,
~-11.5) the best ring agent is ~30 points/round behind; the gap grew from
~19 because the ring splits compute over four agents.
Play-style refresh 2026-10-18 (`local_evaluations/ring_diversity_20261018.json`,
same tool/bank): mean pairwise JS stays 0.10-0.12 through phase 27 (0.1105
now, agreement 72.3%); distance to best fell from 0.163-0.191 to
0.149-0.166; styles converging on best (pass 36-40% vs 38.3%, bomb 39-46%
vs 44.6%). Page republished (version 3) with both updates.

### 2026-10-18 — What the ring agents disagree on (training/ring_disagreements.py)

Current agents (latest.pt, ~1.29-1.36M games), same 2,000-state bank:
all four top moves agree in 54.9% of states; a pair disagrees 27.7%.
Disagreement kinds: single of a different rank 34.3%; pass vs play 33.7%;
same rank, different suit (single 14.0%, straight 4.7%, pair/full house/steps
1.9%) ~20.6% in total (mostly strategically equivalent); different combination
type 7.6%; pair of different rank 2.1%; Tichu call 1.3%; bomb vs non-bomb
0.1%. Context: 33.6% disagreement with 10-14 cards vs 20.2% with 1-4;
30.9% when an opponent holds the trick vs 17.4% when the partner does.
Pass-vs-play splits: agent2 is the passer 63%, agent1 42%, agents 0/3 47-48%.
Implication: ~1/5 of measured disagreement is suit choice; canonicalize
suit-equivalent moves before the divergence metric.

Suit-pooled diversity with 95% game-bootstrap intervals (2e8e1c7;
`local_evaluations/ring_diversity_20261018_pooled.json`,
`ring_disagreements_20261018_pooled.json`, 2,000 states): mean pairwise JS
start 0.090 [0.086, 0.095], after phase 6 0.088, after phase 13 0.088, now
(phase 27) 0.095 [0.090, 0.101] — no significant change (RETRACTS the
earlier unpooled "drifted apart from phase 3" reading). Top-move agreement
rose 74.9% [73.4, 76.2] -> 77.7% [76.5, 78.9] (significant). Pairwise JS now
0.087-0.100 (+-0.007-0.008); vs best 0.131-0.143 (+-0.010). Pooled pair
disagreement 22.3% (was 27.7% with suit splits): single rank 42.8%, pass vs
play 41.5%, combination type 9.5%, pair rank 2.6%, Tichu 1.6%. Page
version 4 shows the intervals. Plateau check: rerun with --states 8000.

### 2026-10-18 — Snapshot drawing by strength (3c34ea8), queued for the ring

Measured before the change (phase 28, pool 60, uniform drawing): last 400
snapshot draws were 10% from <0.6M-game checkpoints, 18% 0.6-0.9M, 72% from
>=0.9M (~2% / ~3.5% / ~14% of all games). User asked for a slight bias toward
stronger/recent opponents instead. New rule: draw weights
softmax(strength / 40) using each snapshot's phase-end screen vs best (starts
screened once at startup), excluding each agent's latest phase-end
checkpoint; 20% snapshot share unchanged. Projected on the phase-28 pool:
<0.6M 3.4%, 0.6-0.9M 12.0%, 0.9-1.1M 29.8%, 1.1-1.3M 41.6%, >=1.3M 13.2%
of snapshot draws. Deploys at the next phase end after 2026-10-18 12:02 via
`~/ring_runs/swap_at_phase_end.sh`.

### 2026-10-18 — GPU sharing, strength drawing live, throughput merge

- 12:32 the ring OOM-crashed in a GT probe: a llama-server/Ollama model loaded
  onto the shared RTX 3080 (5.3 GB). Launcher resumed 12:33 (attempt 2) and
  already ran 3c34ea8, so strength-weighted snapshot drawing went live then
  (mid phase 29), not at a boundary. 13:28 the armed swap restarted at the
  phase-29 end (launcher pid 964361). ~13:32 someone registered it with
  `~/llm/gpu-run --name tichubot-ring --pid 964361` (AGENTS.md "Shared GPU on
  training-server"); recorded need 6,168 MiB.
- `~/ring_runs/launch_ring1.sh` now wraps each attempt in
  `~/llm/gpu-run --name tichubot-ring -- ...` (replaced atomically 13:58).
- Throughput branch from the subagent (session tichubot-86) cherry-picked onto
  master as cdcc3b8..eada6bd: Phoenix-interpretation cache (e643af9), one
  device-to-host copy per batch (bbb3983), batched GT head (eada6bd), plus
  bench/digest/watch tools. Local CPU rollout 39.06+-0.51 -> 50.46+-0.74 g/s
  (x1.29). My independent legal_digest check: 1,500 games / 150,379
  decisions identical to f09c43f. 437 tests pass. Synced 14:00; deploys at
  the phase-31 end via the swap script. Server expectation ~118 g/s (from
  ~94-100); verify and report rollout profile to tichubot-86.
- Deployed 14:50 at the phase-31 end (launcher pid 974712, under gpu-run,
  6,168 MiB reserved). Server throughput 93.8 (sd 1.8, n=39 regular updates)
  -> 132.4 g/s (sd 5.7, n=6), +41%. Per-rollout seconds before -> after:
  legal_actions 48.6 -> 13.9, forward 26.5 -> 20.4, immediate_decisions
  17.6 -> 8.7, env_step 7.9 -> 8.4, obs/history encode unchanged (~5 each).
- Health verified on the new code by 15:57: phases 32-35 ended (parity) with
  parity checks and promotion screens, periodic checkpoints, and a full GT
  probe block (40,000) completed; 0 errors; gpu-run peak 5,838 MiB of the
  6,168 MiB reservation. Latest screens vs best: agent2 -26.4, agent3 -35.5
  (256 deals, noisy).
- 19:34: gpu-run had seen a 6,304 MiB peak against the 6,168 MiB recorded
  need; raised the live reservation in place with
  `~/llm/gpu-run --name tichubot-ring --pid 974716 --mib 6600` (LLM budget
  3,638 -> 3,206 MiB). gpu-run records the run's max peak at exit, so the
  next launch reserves at least 6,304 automatically. Status then: 8.38M ring
  games, phase 46, ~130 g/s, 0 errors since the 14:50 launch.

### 2026-10-19 — Status and snapshot-cache GPU leak fix

10:45 direct check: 13,570,048 ring games, phase 81, 136.5 g/s, 0 errors
since the 2026-10-18 14:50 launch (~20 h); agent games 2,239,490 /
2,233,329 / 2,210,818 / 2,394,117; snapshot pool 166. gpu-run had seen a
7,396 MiB peak (> 6,600 reserved): the trainer cached every drawn snapshot on
the GPU with optimizers (main process ~1.3 GB resident). Raised the live
reservation to 8,000 MiB at 10:46 (LLM budget 1,806 MiB). Fix 2beddda:
snapshots load on CPU, drop optimizers, LRU of 32 (`--snapshot-cache`);
behaviour unchanged; 437 tests pass; synced and armed for the phase-81 end.
gpu-run's recorded need keeps the 7,396 MiB maximum; once the fixed peak is
measured, lower it with --mib or a new job name (AGENTS.md).

### 2026-10-19 — Chart update (page version 5)

12 new ring points (same protocol, 1,024 deals, appended to the 2026-10-17
jsonl): agent0 1,611,775 -39.5, 1,837,057 -35.7, 2,259,970 -27.3; agent1
1,646,571 -38.5, 2,089,968 -30.7, 2,243,568 -23.8; agent2 1,624,079 -39.4,
1,934,345 -34.1, 2,210,817 -32.2; agent3 1,786,879 -33.6, 2,209,795 -26.9,
2,424,836 -28.6. Matched own games vs lineage (interpolated 2.0M -40.9 ->
2.5M -21.3; steep, so rough): a0 +3.4, a1 +7.7, a2 +0.4, a3 -4.2 — roughly
level now (earlier lead gone). Ring total 6.84M games beyond starts exceeds
the whole lineage (5.8M); best ring agent -23.8 vs the lineage endpoint.
Diversity (`ring_diversity_20261019.json`, --slice-every 6, suit-pooled):
mean pairwise JS 0.090 [0.086,0.095] at start -> 0.119 [0.113,0.126] now
(significant rise, mostly after phase 53); agreement 74.9% -> 78.8%
[77.6,80.1]; distance to best 0.13-0.14.

### 2026-10-21 — Strict parity (40f4d71), and gate results

Before the change (phase 228, 29.6M ring games, agents 4.2-4.4M own games):
phases 30-119 median 81,920 games (30% at the 40,960 minimum); last 60
phases median 81,920, 40% at the minimum, 0 caps; at the first check the
trainees were -0.3 pooled (sd 2.35), i.e. parity was decided by noise.
33 promotion gates (4,096 deals) since 2026-10-19 13:53, none promoted;
last 20 mean -5.5, best -0.32 +- 2.07 (agent2 phase218 4,094,963); gates
cost ~6.4 min each (~211 min total). User chose strict parity: a phase ends
only when the pooled advantage over the 9 matchups is >= 2 pooled SE (still
no matchup >= 2 SE behind). Synced 13:18; deploys at the next phase end via
the swap script. Expect longer phases, many to the 500k cap.
Auto-refresh: `python -m training.ring_report --html ring_runs/ring_strength.html`
then publish to a private page; session
cron job every 6 h at :17 (session-only, 7-day expiry).

### 2026-10-21 — Training-curve look: capacity question (training/ring_training_curves.py)

Strict parity since 14:16: phases 230, 231 hit the 500k cap; pooled lead at
checks -1.5..+2.9 (z -0.7..+1.4); no gates since. Expected: each trainee gets
~125k own games per capped phase, and late-stage improvement is ~4-5
points/round per 1M games (lineage 4.54M -> 5.80M: +5.7), so a +4 (2 SE)
lead within one phase is out of reach even for a healthy learner.
Per-update medians by own games, lineage vs ring (500k bins): value loss
0.130/0.145 at 1.0M, 0.142/0.158 at 2.0M, 0.194/0.193 at 3.0M, 0.196/0.194
at 4.0M; entropy 0.33/0.33 at 1.0M then 0.23 for both from 2.0M (entropy
coef reached its 0.01 floor at 2M); KL ~0.004 and clip ~3.3% steady for both.
Reading: ring trains like the lineage; losses flat from ~3M in both while the
lineage still gained strength -> flat curves do not show a capacity limit.
Candidate brakes: the 0.01 entropy floor (entropy pinned at 0.23 since 2M);
capacity needs a one-variable width test. Not run yet.

### 2026-10-21 — Ring paused; entropy-floor experiment launched (training-server)

Ring paused gracefully (SIGINT) at 18:17:37: 31,281,152 ring games, phase
232, state + weights saved. Resume: `RESUME=--resume setsid nohup
~/ring_runs/launch_ring1.sh >> ~/ring_runs/ring1.log 2>&1 < /dev/null &`
(then re-arm `ring_runs/watch_ring.sh` and re-create the 6-h report cron;
the session crons were cancelled while paused).
User asked to lower the entropy floor. Two matched arms from BEST (compact
5,796,864), 1.5M games each to 7,296,864, run sequentially by
`~/entropy_runs/entropy_queue.sh` (control starts only if low wrote final.pt):
- low:  `--entropy-coef 0.01 --entropy-coef-end 0.005 --entropy-decay-games
  1500000 --entropy-decay-start 5796864` (0.01 -> 0.005 linear; no jump at resume)
- ctrl: `--entropy-coef 0.01` (the lineage floor)
Common: `~/entropy_runs/launch_arm.sh <arm>` = resume loop (20 tries) around
`~/llm/gpu-run --name tichubot-entropy --mib 7500 -- ~/envs/tichubot/bin/python
-m training.train_history --resume <BEST|arm latest.pt> --checkpoint-dir
~/entropy_runs/<arm>/checkpoints --log-file ~/entropy_runs/<arm>/train.csv
--games 7296864 --device cuda` + compact recipe (`--hidden 128 --n-layers 3
--trunk-dims 192,160,128 --mlp-activation silu --mlp-layer-norm --embed-dim 32
--hist-d-model 96 --hist-nhead 4 --hist-n-layers 2 --hist-ffn-dim 192
--hist-norm-first --hist-activation gelu --split-actor-critic-history
--static-attention --games-per-update 1024 --workers 4 --synchronous
--batch-size 512 --optimizer-batch-size 2048 --train-gt --gt-probe-every
200000 --gt-probe-games 40000`), env `TICHU_WORKER_CUDA_ALWAYS_RELEASE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Logs ~/entropy_runs/{low,ctrl}.log.
Smoke (4 updates, checkpoint, pass step, 4,096-game probe block): clean,
peak 4,878 MiB (reserved 6,500), ~91 g/s with probes. Expect ~5-6 h per arm.
Launched 18:27. Readout: paired A-vs-B at 4,096 deals, each arm vs start;
promote only via promote_best.
23:17: low arm at 7.12M, no errors, entropy 0.183 (was ~0.22 at start).
Peak VRAM grew to 5,918 MiB over 4.9 h (smoke 4,878, first probe 5,358), so
the live reservation was raised 6,500 -> 7,500 (`gpu-run --pid 1391869 --mib
7500`) and the launcher default too (ctrl arm reserves 7,500).
Interim (23:40): low arm ckpt 7,200,768 vs start (BEST 5,796,864), 4,096
paired deals, CPU: +11.78 pts/round, simultaneous 95% CI [8.89, 14.52]
(z ~8). For scale, the lineage gained +5.7 over 4.54M -> 5.80M. Not yet the
experiment's answer: the control arm has to show how much of this plain
continued training at 0.01 also gives.
`python -m training.paired_checkpoint_eval low7.20M=ring_runs/entropy/low_07200768.pt
start5.80M=<BEST> --deals 4096 --device cpu --output
local_evaluations/entropy_low_07200768_vs_best_4096.json`
Final (2026-10-22): both arms finished clean (low 23:57, ctrl 05:29; 0
restarts; peak VRAM 6,174 MiB recorded). Final checkpoints at 7,297,024,
copied to ring_runs/entropy/{low,ctrl}_final.pt. 4,096 paired deals each, CPU:
- low vs ctrl:   +4.05 pts/round, 95% CI [-0.20, 8.41] (SE ~2.1, z ~1.9,
  block sign-flip p 0.09): suggestive, NOT significant at 2 SE.
- low vs start:  +8.96, CI [4.9, 12.6] (z ~4.2)
- ctrl vs start: +4.70, CI [0.9, 8.8]
  Consistent: 8.96 - 4.70 = 4.26 ~ direct 4.05. Interim low 7.20M vs start was
  +11.8 (within noise of +8.96).
Training curves: low entropy 0.215 -> 0.184; ctrl stayed ~0.22-0.23 (0.209 at
end); KL/clip equal (~0.004 / ~3%).
Reading: continuing at 0.01 gives ~+4.7 per 1.5M (lineage-like); the lower
floor roughly doubles that, but the head-to-head gap is at z ~1.9. A second
4,096-deal batch (new seed) would settle it.
evals: local_evaluations/entropy_low_final_vs_ctrl_start_4096.json,
local_evaluations/entropy_ctrl_final_vs_start_4096.json.
Promotion of low_final (z ~4.2 vs BEST) not done: the auto-mode classifier
denied running promote_best; left to the user:
`python -m training.promote_best ring_runs/entropy/low_final.pt --from-eval
local_evaluations/entropy_low_final_vs_ctrl_start_4096.json`

### 2026-10-22 — Ring GT call rate vs BEST (analytics session)

Ring paused at phase 232 (agents' `latest.pt` = state checkpoints); BEST =
compact-scratch 5,796,864. Scripts and results in
`local_evaluations/gt_rate_20261022/`.
Call rate (exact expectation over 1M hands, 250k random full deals; GT head
sees only the 8-card hand, so this is the head's true rate; SE ~0.03 pp):
`PYTHONPATH=. python local_evaluations/gt_rate_20261022/gt_rate.py CKPT...`
agent0 16.35%, agent1 17.87%, agent2 15.46%, agent3 17.10%, BEST 14.51%
(per hand; training adds GT_EXPLORE_EPS=0.10 coin flips). Corr(strength
proxy, P) +0.65..+0.68 ring, +0.63 BEST. Extra calling is in the middling
band (gt_check proxy 4-8: ring 11.1-13.7% vs BEST 10.5%; 8-10: 54-64% vs 48%).
Earlier MEMORY figures (7.0% at 3.26M, 7.8% std leg) are older checkpoints;
BEST itself already calls ~2x that, so the ring is only ~1-3 pp above BEST.
Outcome (self-play, model vs itself, both teams calling, 8,192 games each,
`python local_evaluations/gt_rate_20261022/gt_made.py LABEL CKPT 8192 OUT`):
made 62.0/62.1/62.8/61.9% (a0-a3, ±0.65 SE) vs BEST 65.1% ±0.7 (pooled ring
62.2% ±0.33, diff -2.9 pp, z ~3.8); net per call +48..+51 vs +60 pts;
net GT pts per team-round +15.4..+17.2 vs +17.4. Caveat: self-play, so each
model's GT faces its own opponents. Reading: the ring calls a little more and
makes a little less; GT profit per round is about the same, consistent with
no strength gain over BEST (gates: 33, none promoted, best -0.32 +-2.1).
Second batch low vs ctrl (seed 20261022, 4,096 deals): +1.35, CI [-2.9, 6.0].
Pooled direct low vs ctrl, 8,192 deals (32 blocks of 256): +2.70 +- 1.62,
z 1.66, 95% CI [-0.5, 5.9]. The two vs-start evals shared seed 20260814 (same
deals), so their +4.26 difference is not independent of batch 1.
Verdict: lower floor (0.01 -> 0.005) is at best a small edge (~+3/round per
1.5M games), not established. Entropy floor is not the main brake.
PROMOTED (user said "yes, promote", 2026-10-22 10:12 UTC): BEST_CHECKPOINT =
ring_runs/entropy/low_final.pt (7,297,024; +8.96 +- 1.77, z 5.07 vs the
compact 5,796,864). The file lives in gitignored ring_runs/ locally and is
copied to ~/tichubot/ring_runs/entropy/low_final.pt on training-server (sha256
5a089eb3...). ring_strength_curve now pins each curve file to its first
row's reference (legacy rows -> compact 5,796,864), so the existing ring
chart stays on the old reference; ring gates/screens use the new BEST once
the server's best_checkpoint.json is synced.

### 2026-10-22 — Step-size check and lr experiment (training-server)

Question (user): is the PPO clip the brake? Clip fraction ~3% at eps 0.2 and
KL ~0.004, so the ratio clip rarely binds. New logging (8f9ecda): pre-clip
grad norm per optimizer step -> train.csv `grad_norm`, `grad_norm_max`,
`grad_clipped_frac`. Probe from BEST (8 updates, lr 3e-4, entropy 0.005):
grad norm mean 0.25-0.35, clipped (>0.5) in 0-5% of steps (spikes up to 6.6)
-> max_grad_norm 0.5 is not the brake either; step size = lr x epochs.
Bug found and fixed (bd633de): Optimizer.load_state_dict restored the saved
lr, so --lr changes were silently ignored on resume (all runs so far used
3e-4, nothing affected). Resume now keeps the configured lr and prints it.
Experiment: from BEST (low_final 7,297,024), 1.5M games each, sequential via
`~/lr_runs/lr_queue.sh` (ctrl starts only if lr2 wrote final.pt):
- lr2:  `--lr 6e-4`   - ctrl: `--lr 3e-4`
Common: `~/lr_runs/launch_arm.sh <arm>` (resume loop under `gpu-run --name
tichubot-entropy --mib 7500`), same compact recipe flags as the entropy runs
plus `--entropy-coef 0.005` (where BEST ended; no jump), `--games 8797024`,
logs ~/lr_runs/{lr2,ctrl}.log, train.csv per arm. Launched 12:37.
Readout: lr2 vs ctrl at >=8,192 paired deals (two seeds), each vs start.
Watcher: ring_runs/watch_lr.sh (failures only; session-bound).

### 2026-10-22 — Saturation check and passing study (analytics session)

Files: `local_evaluations/saturation_pass_20261022/` (scripts, per-checkpoint
JSON, `saturation_table.txt`). BEST at the time = entropy low arm 7,297,024.
All CPU, local copies.

Saturation (`saturation.py`): fixed bank of 4,000 decision states from 506
games (seed 11, `build_state_bank`), 16,000 pass hands, 16,000 GT hands;
40 checkpoints: lineage 0.25M-5.8M, both entropy arms, ring agents at
phases ~0/30/90/150/210/latest.
- Trunk MLP healthy throughout: dead (on <1% of inputs) 3-5%, flat
  (pre-act < -4) <=3%, effective rank of trunk output ~43-47 since 1M.
  Pass and GT trunks: tanh saturation ~0%. No generic unit saturation.
- Actor history-fusion branch is being abandoned: dead units 2% (0.25M) ->
  77% (5.8M) -> 94-98% (7.3M arms); ring 27-48%. Mean-ablation
  (`fuse_ablation.py`, branch replaced by its bank mean) changes the argmax
  on 22.7% of decisions at 0.25M, 14.6% at 1M, 8.7% at 3M, 3.7% at 5.8M,
  1.4-2.1% at 7.3M; ring 6-10%. Critic fusion branch: 95%+ units dead from
  ~2.5M, but mean-ablation still moves V by ~6-16 pts (V sd ~110) -
  small, stable contribution. Static obs evidently carries most of what
  the actor needs; the stronger 7.3M arm uses history least.
- Play-net weight norm grows ~linearly with no plateau: 69 (0.25M) ->
  160 (5.8M) -> 176 (7.3M). Plasticity risk to keep in mind; no decay.
- Policy determinism: entropy 0.85 -> 0.38 by 2M, flat to 5.8M (entropy
  floor); 0.336 ctrl / 0.275 low at 7.3M; decisions with max p > 0.99:
  39% (5.8M) -> 43% ctrl / 50% low. Pass net: first-pick entropy 1.76 ->
  0.22, 52% of hands >0.99. GT: 84% of hands decided (<1% or >99%).
  Reading: more deterministic = stronger so far (low arm); no sign this
  is a limit yet.

Passing strength (`pass_study.py strength`): both teams BEST play/GT/
dragon/wish; candidate team swaps only the passer; 4,096 paired deals (both
orientations, 8,192 rounds), seed 2026, pts/round vs BEST's own passing:
random -155.0 +-2.3; heuristic (best card to partner, two lowest
non-pair cards to opponents, keep Phoenix/bombs) -52.8 +-2.3; pass nets of
lin1.0M -32.6 +-2.3, lin3.0M -17.5 +-2.2, lin5.8M -6.8 +-2.1, entropy
ctrl 7.3M -3.5 +-2.0, ring agent1 latest -8.6 +-2.2; greedy BEST pass
(argmax) +1.0 +-1.9. Null (BEST vs BEST) over 5 seeds: +4.6, +0.7, +0.2,
-2.2, -3.3 (mean 0.0 +-0.9). Caveat: older pass nets are paired with a
play policy that co-adapted with the newer pass net.
Reading: passing is strong (53 pts/round over a sane heuristic) and was
still improving at 5.8M -> 7.3M (+6.8, z~3.3), not saturated.

What it passes (`pass_study.py describe`, 20,000 deals, BEST GT phase):
opponents get 2-5 (left 96%, right 87%), Dog 3-4%, never Mahjong or
anything above 9; partner: A 22%, 6-9 30%, Dog 8%, Dragon 6.4%, Phoenix
5.6%, Mahjong 3.5%. Partner GT: A 47%, K 13%, Dragon 12%, Phoenix 9%,
Mahjong 8%. Own GT: partner gets 6-9 60%, Dog 18% (keeps strength).
Flaw (`bomb_pass.py`): BEST breaks its own four-of-a-kind in 33.9% +-0.5%
of bomb hands (8,315 of 240,000 hands = 3.46%): rank 2 100%, 3 83%, 4 63%,
5 29%, 6 38%, 7-K 13-23%, A 4%. Bomb-masked BEST pass (`bombsafe`) on deals
where any seat has a bomb (DEAL_FILTER=bomb), 4 x 4,096 deals: +9.2, +6.3,
+6.8, +3.4 -> +6.5 +-1.0 pts/round (z~6); nulls on the same deal filter
-2.2, -3.3. ~13% of deals have a bomb -> ~+0.85 pts/round overall from
masking alone.
Result (evaluated 2026-10-24; both arms clean, 0 restarts, finished
2026-10-22 17:52 / 23:17; finals 8,797,184 in ring_runs/lr/):
- lr2 vs ctrl: -4.35 [-9.0, +0.7] (seed 20260814) and -4.87 [-8.8, -1.4]
  (seed 20261024) -> pooled ~ -4.6 (SE ~1.5): 2x lr is WORSE.
- ctrl vs start (BEST 7.30M): +1.98 [-3.2, +7.1] (n.s.); lr2 vs start -3.36.
  1.5M more games at the standard recipe gave only ~+2: flattening.
Reading: step size is not the brake; KL 0.004 at lr 3e-4 is about right.
evals: local_evaluations/lr_lr2_vs_ctrl_start_4096.json, lr_ctrl_vs_start_4096.json,
lr_lr2_vs_ctrl_4096_seed20261024.json.

### 2026-10-24 — Open-ended plateau run (training-server, ~/plateau_runs/run1)

User: "leave it running indefinitely until I come back". Continues the lr
control (8,797,184, lr 3e-4, entropy 0.005 const), so the lineage is BEST
7.30M -> ctrl 8.80M -> run1. `~/plateau_runs/launch.sh` (resume loop, 50
tries, gpu-run tichubot-entropy 7,500 MiB), `--games 500000000`,
`--checkpoint-every 250000` (ckpt_*.pt every 250k kept for the curve), same
compact recipe + GT probes every 200k. Log ~/plateau_runs/run1.log, metrics
run1/train.csv. Stop: kill -INT <python pid>. Launched 2026-10-24 16:27.
Readout later: 4,096-deal paired evals of every ~1M ckpt vs BEST 7.30M (and
vs ctrl 8.80M); promote via promote_best if z>=2.

### 2026-10-24 — Why the history branch dies: unnormalized pre-LN encoder output

Follow-up to the 2026-10-22 saturation check (scripts `fuse_anatomy.py`,
`hist_scale.py`, `param_norms.py`, `token_norms.py` in
`local_evaluations/saturation_pass_20261022/`; same 4,000-state bank, seed 11).
- `TransformerHistoryEncoder` is pre-LN (`--hist-norm-first`) and
  `nn.TransformerEncoder` has no final `norm=`, so the residual stream is
  never normalized. Mean valid-token norm after layer 1: 19 (0.25M), 45
  (1M), 164 (3M), 387 (5.8M), 346-594 (7.3M arms), 164 (ring1); input
  tokens stay ~7-10.
- The pooling attentions have no norm either; pooled history norm (mean
  part): actor 3 -> 67 (5.8M) -> 154-525 (7.3M); critic 24 -> 516 (1M) ->
  48,276 (5.8M) -> 111,529 (ctrl 7.3M). Static features stay ~7.
  Weight growth is modest (critic Wv 8 -> 35, out_proj 7 -> 35); the token
  residual stream is the main driver.
- `history_fuse` = LayerNorm over the concatenation [static(128),
  hist(96)]: hist takes 97% of the LN variance from ~2M on, so inside the
  branch the static half becomes ~constant; W_static times that constant
  gives every unit a -2.2..-2.5 offset (bias itself only -0.02..-0.19),
  while the history part varies only ~0.1 per unit (0.5-0.6 at 0.25-1M).
  Units never cross zero -> 'dead' SiLU. Critic: after LN only 3-8% of
  the history vector is state-varying (a constant direction dominates).
- Not measured: whether this costs strength (the 7.3M arm is strongest
  while using history least). Fix candidates (code change, needs a
  one-variable training test): final LayerNorm on the encoder
  (`nn.TransformerEncoder(norm=...)`) and/or a separate LayerNorm on the
  pooled history before the concat; optionally weight decay on the encoder.
Plateau run1 stopped 2026-10-24 16:57 by SIGINT at 8,976,384 (latest.pt
saved; resume: `cd ~/plateau_runs && setsid nohup ./launch.sh >> run1.log
2>&1 < /dev/null &`), replaced by the history-norm run below (user's call).

### 2026-10-24 — History-branch scale bug and fixed-branch run (training-server)

Found by the analytics agent, verified independently (48 deals/ckpt, ~3.6k
decisions; scratchpad hist_check.py): pre-norm encoder (`--hist-norm-first`)
has no final LayerNorm, so token encodings grow without bound; attention
pooling passes it on; the fuse LayerNorm over [static(128) ; history(96)]
then gives history ~98% of the variance. Median norms static/token/actor
pool/critic pool: 1.0M 8.5/46/15/513 (hist share 76%/97%), 3.0M
8.3/158/38/4,704, 5.8M 6.9/430/142/44,084 (98%/98%), BEST 7.3M
7.4/342/256/60,820 (98%/99%).
Fix (2e82744): `--hist-final-norm` (encoder `norm=LayerNorm`), `--hist-pool-norm`
(LayerNorm per pooled actor/critic history vector before fusion); default
off, old checkpoints load unchanged. Resuming an unnormed ckpt remaps Adam
state by parameter name (6 new params without state). `--reset-history-fuse`
zeroes history_fuse.1 / critic_history_fuse.1 so delta = 0 at start; BEST
with zeroed delta vs BEST: -4.25, CI [-9.3, +0.7] (2,048 deals) = mild shock.
Run: `~/histnorm_runs/launch.sh` (open-ended, `--games 500000000`, resume
loop 50 tries, gpu-run 7,500 MiB; first launch from BEST with
--reset-history-fuse, resumes never), compact recipe + both norm flags,
lr 3e-4, entropy 0.005, ckpt every 250k, GT probes every 200k. Log
~/histnorm_runs/run1.log. Launched 17:06. Smoke (fresh + own-resume) clean,
peak 4.6 GB. Matched comparison: its ~8.80M ckpt vs ring_runs/lr/ctrl_final.pt
(same start/recipe/games, old architecture), then vs plateau run1.
Supervision: ring_runs/watch_run.sh histnorm_runs run1.log (session-bound).
Readout 2026-10-25 (run healthy, 0 restarts, 13.65M at 15:19; reservation
raised 7,500 -> 8,500 MiB live and in launch.sh after peak 6,802 MiB):
- Matched (same start/recipe/games), hn 8,750,080 vs old-arch ctrl 8,797,184:
  +7.62 [3.3, 11.9] (seed 20260814) and +1.37 [-2.3, 5.2] (seed 20261025);
  pooled 8,192 deals +4.50 +- 1.58, z 2.85.
- hn 13,500,416 vs BEST 7.30M +23.82 +- 1.83 (z 13.0); vs ctrl 8.80M +24.25;
  vs hn 8.75M +20.65 +- 2.13 (~+4.3/1M games, vs ~+1.3/1M for the old arch
  7.30 -> 8.80M): no plateau in sight with the fix.
- PROMOTED (promote_best --from-eval, z 13.04): BEST_CHECKPOINT =
  ring_runs/histnorm/ckpt_13500416.pt (server copy ~/tichubot/ring_runs/histnorm/,
  sha256 00eb55aa...). Architecture flags travel in its config.
- Scales now (median): token 10-11 (was 342), static 6.6-7.8; actor pool
  output before its norm still grows (191 -> 1,150) but is normalized before
  fusion; history share of fuse-LN variance 84-90% actor, 57-83% critic
  (was 98-99%). History-ablation move-change rate not re-measured yet.
evals: local_evaluations/histnorm_8750080_vs_ctrl_4096*.json,
histnorm_13500416_vs_best_ctrl_hn875_4096.json.
2026-10-25 21:13: 15.30M, 0 restarts, peak 7,040 MiB (res 8,500). hn 15,250,432
vs BEST 13.5M: +1.05, CI [-2.8, +4.8] (4,096 deals): gain slowed sharply
after 13.5M (vs ~+4.3/1M before); policy entropy 0.134 -> 0.121. No promotion.

### 2026-10-25 — History norms default everywhere; "noLN" naming (6642711)

User: insert the missing LN on all architectures going forward, and rename
the runs without it for clean comparisons. Done:
- TrainConfig + TransformerHistoryEncoder/TichuPolicyHistory default to
  final_norm / history_pool_norm = True.
- Layout of a checkpoint = its weights (checkpoints.history_norm_flags);
  loaders, rollout workers and ring snapshots build that; pickled configs
  read hist_*_norm from the instance dict (False if absent).
- Training resume of a noLN ckpt upgrades it automatically (Adam remap +
  fuse reset); `--legacy-history-norms` reproduces the old layout. Ring:
  trainees upgrade on resume, frozen snapshots stay as trained.
- Naming: every pre-fix history checkpoint is "noLN" (`is_legacy_history`);
  arena/paired-eval labels get a -noLN suffix automatically, paired-eval
  manifests carry `legacy_history`; ring chart series marked "(noLN)".
  Paths left unchanged (BEST history, evals and MEMORY reference them).
  noLN runs: all kaggle_runs history lineages, ring1 agents, entropy_runs,
  lr_runs, plateau_runs/run1. Their server launchers now pass
  --legacy-history-norms so a resume stays a clean noLN control.

### 2026-10-25 — Saturation pipeline (training/saturation.py, b31475a)

User: replace the throwaway saturation scripts with a clean pipeline. One
module + tests: `python -m training.saturation bank` (once; fixed bank
local_evaluations/saturation/bank_v1_n4000_s20261022.pkl, gitignored, 4,000
states from 465 games played by BEST 13.5M), `probe LABEL=CKPT ... |
--run-dir DIR --every 1000000 --prefix NAME` (records in
local_evaluations/saturation/records/, cached by file sha + METRICS_VERSION +
bank id; noLN labels tagged), `report [--filter]`.
Cross-check vs the analytics agent's 2026-10-22 numbers (different bank):
5.8M actor-fuse dead 77% (77), flip 3.5% (4%), |W| 160 (160), erank actor
78.6 (78.7); 7.3M flip 2.1% (1-2%). First readout:
- noLN: history ablation flip 7.5% (3.0M) -> 3.5% (5.8M) -> 1.0% (lr ctrl
  8.8M; token norm 884, critic pool 198k): the branch keeps dying.
- histnorm: flip 4.0% (8.75M) -> 7.3% (13.5M) -> 9.6% (15.25M), KL 0.03 ->
  0.19; actor-fuse dead 68 -> 51%; hist-ffn dead 14 -> 5%; token norm ~10.
  The history branch is being used more and more.
- Still: critic fuse 77-86% dead; weight norm grows faster (189 -> 265);
  policy entropy 0.256 -> 0.204, decisions with p>0.99 50 -> 58%.

### 2026-10-26 — Evaluator: 16k default, sharding, CRN flag (a5ebf6a)

User: 4x deals for late-stage precision (SE ~1.05); add CRN as a flag.
- Per-deal noise decomposition: BEST vs itself (plain, 2,048 deals) has
  per-deal sd 118.8 vs 128.6/140.3 for real pairs, i.e. ~85% of the
  variance is action-sampling noise.
- CRN (--crn): per-deal streams keyed (deal, seat, decision type) feed the
  Gumbel-max samplers of play, pass, GT, dragon, wish. Self vs self (BEST,
  4,096 deals, server): exactly 0.00, sd 0. BUT 13.5M vs 15.5M at 16,384
  deals: sd 127.85 plain vs 124.96 CRN (-2%): checkpoints 2M games apart
  diverge in the first decisions, so CRN buys ~nothing for typical pairs.
  Flag kept, default off.
- 13.5M vs 15.5M (histnorm run): plain -3.16 [-5.1, -1.2], CRN -0.94
  [-2.9, +0.9]; pooled ~ -2.0 +- 0.7: the histnorm run regressed after
  13.5M under the fixed entropy schedule. BEST stays 13.5M.
- Speed: server = Xeon W-2133, 6 cores / 12 threads; 16,384 deals take
  ~300 s with 10 workers (CPU floor; a single process ran ~63 g/s using
  torch's own threads). GPU does not help (Python game engine bound).
Files: local_evaluations/crn_validation/*.json; script ~/crn_validate.sh.
Pitfall again: `cat > f && ... &` backgrounds the whole chain (empty f).

### 2026-10-26 — MMD from-scratch run launched (training-server ~/mmd_runs/scratch1)

User: restart from scratch with the history norms, replace the entropy
schedule by a principled rule. Discussion (2026-10-25/26): entropy bonus =
pull to a fixed uniform magnet; Magnetic Mirror Descent (Sokota et al. ICLR
2023) + moving magnet (DeepNash R-NaD outer loop); exploration concern
(reverse KL to a peaked magnet suppresses dropped moves cumulatively; the
convergence proofs assume full feedback, PPO only sees sampled moves) ->
mixed magnet with uniform share eps. PBT judged too slow for now.
Implementation dcb8ebc (training/mmd.py; tests/test_mmd.py: round-0 update
== entropy update to 1e-6; controller move/restart/resume).
Knobs: alpha 0.03 (= lineage start entropy, validated range); eps 0.2
(lineage at its 0.01 floor spent 15% on non-preferred moves: 0.147 at 4.54M,
0.153 at 5.80M, 4,000 bank states; eps*mean(1-1/n)=0.15); magnet moves when
the per-update KL(pi||rho) OLS slope over 100 updates is not > 2 SE, round
>= 200k games, forced check at 3M; at each move the round's policy plays the
magnet at 16,384 deals (~5 min CPU): z <= -2 -> restart the round from the
magnet (user: no watcher alert for this, just restart). Pass/GT/dragon/wish
keep the lineage schedule 0.03 -> 0.01 over 2M. Everything else = compact
lineage recipe (lr 3e-4, 1024 games/update, GT probes 200k/40k, ckpt 250k).
Launcher: ~/mmd_runs/launch.sh (copy ring_runs/mmd_launch.sh), resume loop
50 tries, gpu-run tichubot-scratch-mmd 8,500 MiB, --games 500000000.
Smoke (12k games, tiny thresholds): round 0 forced move, round 1 check
+94.9 +- 4.7 then move, resume into round 2 clean; peak 5.6 GB.
Launched 2026-10-25 23:33. Files: checkpoints/mmd_state.json (events),
magnet.pt, mmd_round_NN.pt, mmd_evals/*.json. Comparisons later: vs the noLN
lineage at equal games (16k deals), per-round policies vs BEST 13.5M offline.
Open question to watch: whether the KL slope test fires or rounds run to the
3M cap (self-play moves the fixed point; lineage entropy at constant 0.01
was flat 2M-5.8M, suggesting it will plateau).
Readout 2026-10-27 11:41 (unwatched between sessions; healthy: 8.37M, 1
launch, 0 errors, peak VRAM 7.3 GB of 8.5): 30 magnet rounds, 1 restart
(round 15 at 3.92M: -4.36 +- 1.17 vs magnet, rerun then moved). Rounds end by
the KL slope test every ~200-350k games (never the 3M cap); magnet KL steady
~0.13-0.14. Round gains vs own magnet (16k deals): +34.8 (r1), +18.3, +7.2,
+7.8, +4.5 ... still +2.6..+4.0 in rounds 22-29 (r26/r28 ~ -0.7, n.s.).
Entropy mmd vs lineage at equal games: 0.34/0.42 (0.5M), 0.26/0.36 (1M),
0.23/0.24 (2M), then ~0.21-0.22 flat through 8M (no collapse; eps floor).
Throughput 78-80 g/s (lineage recipe ~92; magnet forward + ~5 min eval/round).
Fixed-reference evals (16,384 deals, local CPU, procs 9):
- mmd 5,821,440 vs lineage 5,796,864 (noLN, same recipe otherwise): +0.49 +- 1.09
  -> from scratch, LN+MMD is level with the lineage at equal games.
- mmd 8,163,328 vs lr-ctrl 8,797,184 (noLN, BEST-lineage continued): +5.86 +- 1.09
- mmd 8,163,328 vs BEST (histnorm 13.5M): -19.01 +- 1.07
Files: local_evaluations/mmd_scratch1/*.json; ring_runs/mmd/ (round ckpts).
Readout 2026-10-29 09:40 (unwatched 10-27 -> 10-29; healthy: 18.85M, 1
launch, 0 errors; 65 rounds, 2 restarts; recent round gains +0.5..+2.2 vs
own magnet; peak VRAM 7,924 MiB -> reservation raised 8,500 -> 9,200 live and
in launch.sh). 16,384 deals:
- mmd 18,758,656 (round 64) vs BEST histnorm 13.5M: +10.86 +- 1.13 (z 9.6)
- mmd 18.76M vs mmd 8.16M: +31.21
PROMOTED: BEST_CHECKPOINT = ring_runs/mmd/mmd_round_64.pt (server copy
~/tichubot/ring_runs/mmd/, sha256 865e9754...). From scratch, LN + MMD passed
the previous best (which had ~13.5M games incl. the noLN lineage) at 18.8M.
Tooling (user: "why can't we make scripts for that?"): training/run_report.py
— `status RUN`, `fetch RUN [--round N]`, `readout RUN [--promote]` (16k vs
BEST + previous readout, history in local_evaluations/<run>/readouts.jsonl).
For interest (user, 2026-10-29): BEST mmd 18.76M vs the 2026-08-01 advisor
model (kaggle_runs/2026-08-01_110222_standard, 14,790,656 games, noLN):
+54.65, CI [52.9, 56.4] (16,384 deals). Chain estimate via promotions was
~+64 (19.6 + 9.0 + 23.8 + 10.9); not transitive, the direct number counts.
Deployment export 2026-10-29: `python distribution/build_lightweight.py
--checkpoint ring_runs/mmd/mmd_round_64.pt` -> dist/{linux/TichuBot-Linux.zip,
macos/TichuBot-Mac.zip, windows/TichuBot-Windows-Source.zip}, 6.0 MiB each,
inference-only model 2.0 MiB (model/pass/GT weights verified identical to
BEST), smoke-tested. SHA-256 linux b1a6a4f5..., mac fed6dbdb..., win-source
4a1283d3.... Full TichuBot.exe / TichuBot-Windows.zip NOT rebuilt (separate
self-contained Windows build; still the old model). Builder default changed:
automatic selection = BEST_CHECKPOINT (was "most games", which would have
picked a non-best checkpoint).
Readout 2026-11-01 14:46 (unwatched 10-29 -> 11-01; healthy: 36.76M, 1 launch,
0 errors, 117 rounds, 5 restarts; entropy 0.21 (8M) -> 0.17; peak VRAM 8,044
of 9,200 MiB). Plateau check, 16,384 deals, latest round 116 (36,637,696):
- vs own 30.11M (round 99): +9.53 +- 0.97
- vs own 23.84M (round 80): +16.29 +- 0.97
- vs BEST 18.76M (round 64): +26.00 +- 1.00
=> no plateau: ~+1.1/1M (23.8->30.1M) and ~+1.5/1M (30.1->36.6M).
PROMOTED: BEST = ring_runs/scratch1/mmd_round_116.pt (server copy
~/tichubot/ring_runs/scratch1/, sha256 9e3a8c0a...). (promote_best reports
z 32 from block SE; per-deal SE gives z 26.)

### 2026-11-01 — GT scaffolding off in the MMD run (user: random declares poison play data)

User worry: random GT declarations teach the play net that a declaration says
little about the hand (opponents ignore, partners don't support, declarers
play out junk). Two channels were on: GT_EXPLORE_EPS=0.10 eps-declares in
the main stream (~1/4-1/3 of all declarations random, head itself calls
~14% of probed hands) and gt_probe_practice (play + pass nets trained on
probe rounds, 4/5 with a forced declare on a uniform hand, ~20% of games).
Both were scaffolding for the 2026-07 never-call deadlock, which is broken
(probe p 0.11-0.18, 40% of forced declares positive). New flag
--no-gt-explore (c.f. config gt_explore); launch.sh now passes
`--no-gt-explore --gt-probe-practice 0`. Probes still train the GT head
(regression) but never enter play/pass data. Switched by
~/mmd_runs/switch_at_round_end.sh at the round-119 -> 120 magnet move:
SIGINT 18:15:45, relaunch 18:16:05, resumed at 37,562,368 games, MMD state
intact (round 120, 5 restarts). Strength curve: change point at 37.56M.

### 2026-11-01 — Transitivity / metagaming check of the MMD run (training/transitivity.py)

User asked whether the agent metagames against itself / time for a ring.
Ladder (MMD rounds 29/64/80/99/116 = 8.2/18.8/23.8/30.1/36.6M) x outside
panel (histnorm 13.5M LN, lineage 5.8M noLN, Aug-1 standard 14.8M, ring
agent2 noLN), 16,384 deals per cell; report local_evaluations/transitivity_mmd/.
36.6M scores: +38.5 / +62.6 / +70.0 / +70.5 vs the panel; monotone up the
ladder vs every opponent, no cycles. Direct gaps vs via-panel gaps:
r99->r116 direct +9.9 | via hn +8.1, lin +4.4, aug +3.1, ring +6.0;
r80->r99 +8.3 | 10.4, 5.7, 6.7, 7.7; r64->r80 +8.2 | 7.4, 8.4, 7.0, 7.7;
r29->r64 +28.3 | 32.7, 28.5, 20.4, 30.9. Reading: transitive within noise vs
the strongest outside opponent; smaller gains vs weak opponents = margin
compression (already +60-70). No metagaming evidence -> no ring now.
Trigger for a ring: direct gains persistently > gains vs the strongest
outside opponents, or outside gains flattening while self gains continue.
run_report readout now also scores OUTSIDE_PANEL (histnorm 13.5M, ring agent2).
