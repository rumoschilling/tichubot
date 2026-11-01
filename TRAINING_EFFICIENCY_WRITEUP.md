# Training Rollout Efficiency and Synchronization

Date: 2026-07-30

## Scope

This change addresses three issues found while auditing the current
history-policy training run:

1. Action-vector caching spent more time measuring cache hits than it saved.
2. Every transition carried a dense copy of its full history prefix.
3. Adapt rollouts could be collected before later policy updates, most
   severely across Grand Tichu probe-practice blocks.

The fixes preserve the policy inputs and PPO equations. They change how
transient rollout data is stored and when a rollout is collected.

## 1. Action-vector cache

### Problem

`training/rollout_embed.py::_action_vec()` called `functools.lru_cache.cache_info()`
twice for every legal action so it could maintain separate hit counters. The
cache lookup itself was cheap, but those two statistics snapshots made the hot
path expensive.

The audit benchmark used 28,264 real rollout action requests with an 82% hit
rate:

| Mode | Time |
| --- | ---: |
| Cache disabled | 10.6 ms |
| Old cache enabled | 30.4 ms |

The old cache was therefore about 2.9 times slower than rebuilding the binary
57-element vector.

### Fix

- `_action_vec()` now performs exactly one normal LRU lookup.
- `get_action_vec_cache_stats()` derives calls, hits, and misses from
  `cache_info()` only when statistics are requested.
- Enabling or disabling the cache clears it, so statistics and stored entries
  cannot leak across modes.
- Cached arrays are marked read-only. Current callers copy them into rollout
  batches, and accidental mutation can no longer corrupt future cache hits.

### Result

Post-fix matched benchmarks:

| Workload | Hit rate | Cache disabled | Cache enabled |
| --- | ---: | ---: | ---: |
| 58,064 realistic one-pass requests | 84.5% | 22.2 ms | 22.1 ms |
| 229,604 repeated requests | 96.4% | 99.6 ms | 73.6 ms |

The cache is now neutral on the realistic cold workload and beneficial when it
is warm, instead of being dominated by instrumentation overhead.

## 2. Shared history rollout representation

### Problem

The history model consumes a sequence of 63-feature play tokens:

- 4 relative-player one-hot features;
- 59 objective play features.

The collector used to materialize a dense tensor shaped
`(transitions, maximum_history, 63)`. Most rows in one game are prefixes of the
same immutable final play log. Copying every prefix duplicated the same token
many times, then multiprocessing copied those dense tensors to the parent, and
the parent allocated another dense tensor while merging worker results.

The audit's production-sized 64-game sample had:

- 6,529 transitions;
- 110.9 MiB total rollout data;
- 96.5 MiB of history data.

The collector also allocated a full maximum-sized history tensor and
immediately replaced it with a second rebuilt tensor.

### New representation

The collector now stores each game's objective log once:

| Array | Shape | Meaning |
| --- | --- | --- |
| `history_tokens` | `(H, 59)` float16 | Objective token features |
| `history_players` | `(H,)` int8 | Absolute player for each token |
| `history_game_offsets` | `(G+1,)` int32 | Token range of each game |
| `history_game_ids` | `(N,)` int32 | Source game for each transition |
| `history_observers` | `(N,)` int8 | Observing player for each transition |
| `history_lens` | `(N,)` int32 | Visible prefix length |

For a PPO minibatch, only its selected prefixes are materialized. The relative
player feature for token `t` and transition `n` is reconstructed as:

```text
relative_player = (absolute_token_player - transition_observer) mod 4
```

That recreates the original 63 features exactly:

```text
[one_hot(relative_player, 4), objective_token_features(59)]
```

The eager path keeps the compact arrays on the training device and performs a
vectorized gather/scatter there. The lazy path builds only the current
minibatch on CPU and transfers that minibatch. The specialist dragon and wish
buffers remain dense because they are rare and are not the memory bottleneck.

Worker merging adjusts both token offsets and game IDs. It does not construct a
dense intermediate. The old dense `history` input remains supported by PPO and
the merger for compatibility with tests and external callers. Checkpoint
format is unaffected because rollout buffers are transient.

### Result

A post-fix production-sized 64-game sample had:

- 6,439 transitions;
- maximum history length 128;
- 15.014 MiB total rollout data;
- 0.779 MiB shared history data;
- 99.037 MiB for an equivalent dense float16 history tensor.

The shared representation reduced history storage by 127.1 times. Total
rollout size is about one seventh of the old audit sample. The two samples use
different stochastic games, so their transition counts differ slightly, but
the model size and 64-game workload are the same.

The tradeoff is a vectorized reconstruction for each PPO minibatch and epoch.
That deliberately exchanges inexpensive device computation for a large
reduction in worker RAM, IPC volume, merge allocations, and retained rollout
memory during long probe blocks.

## 3. Adapt rollout synchronization

### Problem

The shared training loop started the next asynchronous adapt rollout before it
checked whether a Grand Tichu probe block was due. A probe block currently
contains 20 chunks, and probe practice updates the play, pass, and GT policies
between chunks. The already queued adapt rollout was consumed after all those
updates.

Across all seven recent probe blocks, the first post-probe adapt update showed:

- approximate KL `0.032-0.046`;
- clip fraction `0.113-0.131`.

Normal updates were generally around KL `0.001-0.006`. The stale buffer also
kept roughly a gigabyte of old dense rollout data alive throughout a
approximately 20-minute probe block.

The synchronous branch had a related boundary issue: it collected the next
batch after the normal PPO update but still before the probe-practice updates.

### Fix

`training/play_training.py` now computes `probe_due` before starting another
rollout.

- Neither asynchronous nor synchronous mode starts an adapt rollout before a
  due probe block.
- The next adapt rollout starts after the entire block, using the final play,
  pass, and GT weights.
- The standard Kaggle GPU bundle now passes `--synchronous`. Every normal
  adapt rollout is collected at the start of its cycle, after all play,
  pass, GT, and probe updates from the preceding cycle, and is consumed by
  the immediately following play update.

The asynchronous mode remains available for other experiments, but its probe
boundary is now safe. It still intentionally overlaps normal PPO updates with
the next adapt collection when `--synchronous` is not selected.

### Tradeoff

Synchronous collection gives up rollout/PPO overlap. The audit measured an
average 22.65 seconds in PPO and 0.86 seconds waiting for an already-prefetched
rollout in the prior asynchronous run. Actual synchronous wall time will
include the full collection duration rather than only that wait, so throughput
will be lower. The benefit is a simple policy relationship: every adapt batch
was sampled from the exact weights it updates, with no normal one-update lag
and no multi-update probe lag.

## Validation

The regression suite includes:

- exact shared-history reconstruction against `encode_history()` snapshots;
- exact shared-history reconstruction across worker-buffer merges;
- dense versus shared PPO optimizer-step equivalence;
- eager versus lazy shared-history PPO equivalence;
- action cache contents, read-only safety, and LRU accounting;
- synchronous rollout start weights after each PPO/probe update;
- asynchronous probe-boundary ordering.

Validation command:

```bash
python -m pytest -q
```

Result:

```text
226 passed in 7.74s
```

## Files changed for this fix

- `training/rollout_embed.py`
- `training/rollout_history.py`
- `training/ppo_history.py`
- `training/play_training.py`
- `kaggle/prepare_standard_gpu_bundle.py`
- `tests/test_action_vec_cache.py`
- `tests/test_play_training_schedule.py`
- `tests/test_ppo_rollout_optimizations.py`

## Deferred audit findings

These were observed but intentionally not mixed into this change:

- Transformer gradient checkpointing measured about 26% overhead locally.
- The standard T4 bundle does not yet enable AMP.
- Async timing reports wait time rather than full collection wall time, and
  some rollout profile totals double-count nested phases.
- Periodic checkpoints are serialized twice and all numbered snapshots are
  retained.
- Embed rollout workers rebuild model objects on every update.
- Evaluation and arena passing decisions are scalar rather than seat-batched.

AMP and disabling gradient checkpointing need a measured GPU memory and
throughput comparison before changing the long-running Kaggle configuration.

## Next run

The standard bundle generator embeds the current `training/` and `tichu_env/`
source trees when it runs. The next normal push through
`bash kaggle/run.sh standard --resume <checkpoint>` will therefore include the
compact histories, cache fix, probe-boundary fix, and `--synchronous` setting.
