# Codebase audit — 2026-09-23

This audit records the working tree reviewed on 2026-09-23, including
uncommitted and untracked source. Existing changes were preserved. The
findings below describe the original failures; remediation status is tracked
separately so the original reproduction evidence remains available.

All 20 initial grouped findings below were addressed in the working tree after
the user requested fixes. Three follow-up passes fixed thirteen additional defects. The original findings and reproductions are retained
as audit evidence. P1 denotes high priority; P2 normal priority; P3 low.

## Remediation — 2026-09-23

| Findings | Status | Resulting behavior |
| --- | --- | --- |
| T1–T4 | Fixed | Async history dispatch copies all policy weights; workers restore actual model dimensions, legacy wish heads, and requested devices; migrated GT heads do not inherit classifier optimizer moments. |
| D1–D4 | Fixed | Runtime bundles include observations; wheels build with the supported backend and contain browser assets; checkpoint-dependent Kaggle pushes require resume input; documented and legacy entry commands use the intended interpreter. |
| I1–I5 | Fixed | Rejected advisor moves preserve state; fixed and counted-input policies work through their supported inference paths; Phoenix choice survives execution; CLI tuple actions work; known Mahjong passing determines leadership. |
| R1–R7 | Fixed | Legal Phoenix substitutions are generated, rank-1 substitutions rejected, malformed moves remain atomic, phase and wish rules are enforced, Dragon history advances, Mahjong wishing works in the wrapper, and history-space bounds match existing encoding. |

Integration validation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **373 passed, 1 skipped in 13.85 seconds**; diff check clean. The skip
is the CUDA worker check because CUDA is unavailable. The suite includes
regressions for dispatch isolation, worker reconstruction, optimizer migration,
rejected actions, advisor rollback, model-aware CLI execution, Kaggle guard
behavior, staged runtime import, and offline wheel construction/import.

The sampled-rollout test now settles a pending Dragon gift before requesting
ordinary card actions; a gift decision correctly has no ordinary legal actions.
The history feature scale remains unchanged. Rule corrections change the set
of legal plays, so historical strength measurements should not be treated as
results under the corrected engine without rerunning their controlled comparisons.
No training, checkpoint-strength evaluation, remote Kaggle execution, native
desktop build, or live browser test was performed during remediation. Existing
unrelated edits and checkpoints were preserved. Changes remain uncommitted.

## Follow-up remediation — 2026-09-23

The subsequent pass fixed six more reproduced defects. The three game/interface
boundary changes and the advisor reconciliation change were reviewed through
existing Herdr agents; rollout work remained in the main session.

| Finding | Reproduction and fix |
| --- | --- |
| T5 — Fixed-policy rollout skipped Mahjong wishes | The collector checked membership in `(cards, call_tichu)` instead of `cards`. It now inspects the played card list and declares wishes only while the round is active. A deterministic short game verifies the next observation contains the wish. |
| T6 — More workers than games crashed fixed-policy collection | One game across four workers produced zero-game buffers with incompatible observation dimensions. Empty worker assignments are now omitted, matching the embed/history collectors. The regression exercises collection, worker reconstruction, and buffer merging with an immediate test pool. |
| R8 — Invalid Dragon recipients could corrupt state or crash | Recipient `-1` selected the last hand; recipient `4` raised an indexing error. The engine now requires an integer opponent seat, rejecting booleans/floats and invalid indices atomically while retaining NumPy integer support. |
| R9 — Inactive players received enabled action masks | All observers received the active player's mask, including Mahjong wish choices. Only `agent_selection` now receives enabled actions; tests cover both the wish decision and the next player's turn. |
| I6 — Advisor reconciliation contradicted known passing | A different player could play a card whose recipient was known, or the reconciliation swap could remove a known passed card. Known recipients and retained passed cards are now protected. Without an unknown replacement, the move is rejected with state and RNG preserved; successful swaps preserve hand counts. |
| I7 — Negative HTTP body length caused an unbounded read | `Content-Length: -1` reached `read(-1)`. It is now rejected before reading any bytes; a request-parser regression verifies the stream remains untouched. |

Final follow-up validation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **380 passed, 1 skipped in 13.45 seconds**; diff check clean. The skip
still requires unavailable CUDA. No training, model-strength evaluation, remote
execution, or live browser/server test was performed. All changes remain
uncommitted, with existing unrelated work preserved.

## Passing-worker and pending-decision follow-up — 2026-09-23

Four more reproduced failures were fixed in the next pass:

| Finding | Reproduction and fix |
| --- | --- |
| T7 — Passing workers rebuilt different models | The standalone collector rebuilt the frozen value model and specialist networks using unrelated trainer defaults. Custom dimensions failed to load, and nondefault history activations/normalization could diverge. Workers now receive a frozen CPU copy of precisely the trunk and value head already used by the static reward calculation; passing and GT dimensions come from dispatched weights. Tests compare exact static values for fixed, embed, and history models, including a nonuniform SiLU/LayerNorm trunk. |
| T8 — Passing collection dispatched empty workers | One deal across four workers attempted a passing forward pass with a malformed empty hand tensor. Zero-deal assignments are now omitted. |
| T9 — Passing rewards rejected counted observations | The collector always built 196-feature value inputs; supported 200-feature policies failed matrix multiplication. Pre/post-pass inputs now use the loaded policy's observation encoder and width, in both local and worker paths. |
| I8 — Advisor accepted a play before the pending Mahjong wish | After Mahjong, an opponent could play while `waiting_for_wish` was true. The opponent-play boundary now rejects before any mutation; a regression verifies the same move succeeds once the wish is recorded. |

The six initial passing regression cases failed before repair. Seven passing
checks now pass, including an actual two-process `spawn` pool with custom
passing/GT heads and a counted-observation value model. This preserves the
existing static value-delta reward definition; it does not change the training
objective or claim improved playing strength.

Final validation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **388 passed, 1 skipped in 14.30 seconds**; diff check clean. CUDA remains
unavailable. No training, checkpoint-strength experiment, remote execution, or
live browser test was run. Changes remain uncommitted.

## Dragon scoring and interface-state follow-up — 2026-09-23

| Finding | Reproduction and fix |
| --- | --- |
| R10 — A bomb capturing the Dragon wrongly gave away the trick | Both ordinary completion and round-end cleanup checked whether Dragon was anywhere on the table. They now check whether Dragon is the winning combination. Two failing regressions reproduced the wrong gift/score; a positive regression retains the gift when Dragon wins. |
| I9 — Pending advisor Phoenix choice survived another play | Opponent recording accepted a different move while a Phoenix interpretation was pending, leaving its old prompt and undo baseline attached to an advanced state. Alternative plays are now rejected until the pending choice is resolved or cancelled. The regression checks rejection preserves state and cancellation enables the alternative. |
| I10 — Round-ending Mahjong opened a stale wish prompt | Human and advisor Mahjong plays prompted for a wish even after the round ended. Both paths now open the prompt only while the round remains active; terminal and ordinary wish flows are covered. |

R10 follows the publisher's rule that a Dragon gift is required when Dragon
wins the trick; a bomb can beat Dragon. See page 2 of the
[Rio Grande Tichu rules](https://www.riograndegames.com/wp-content/uploads/2013/02/Tichu-Rules.pdf).
This corrects gameplay and scoring; historical strength comparisons used the
previous engine and were not rerun here.

Final validation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
git diff --check
```

Result: **394 passed, 1 skipped in 14.51 seconds**; diff check clean. CUDA is
unavailable. No training, checkpoint-strength evaluation, remote execution,
or live browser test was run. Existing unrelated work is preserved and changes
remain uncommitted.

## Review allocation

Three `gpt-5.6-sol` agents were started through the installed Herdr skill
(`herdr --skill`), in sibling panes of the original tab with focus preserved:

| Reviewer | Herdr pane | Scope |
| --- | --- | --- |
| `tichu-rules` | `w1:p0` | Game rules, state, encoding, related tests |
| `tichu-interface` | `w1:p11` | Web, advisor, terminal play, replay |
| `tichu-distribution` | `w1:p12` | Desktop distribution, Kaggle, Colab, project packaging |
| Main session | `w1:pZ` | Training, checkpoint restoration, rollout orchestration, evaluation code |

The training/evaluation analysis stayed in the main session as required by
`AGENTS.md`. No training runs, checkpoint strength comparisons, downloads,
distribution builds, or publishing were performed for this audit.

## Original audit validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
```

Result: **336 passed in 12.55 seconds**.

Additional targeted checks used tiny synthetic models, temporary checkpoints,
and mocked rollout execution. These checks are described with each finding.
GPU execution and native Windows/macOS installation were not exercised.

## Confirmed training findings

### T1 — P2: CPU asynchronous dispatch does not snapshot model weights

Location: `training/rollout_history.py:1188` (also the GT and passing weights
at lines 1193 and 1196).

`detach().cpu().numpy()` shares storage with a CPU policy. `map_async` hands
these arrays to a background task feeder, while `run_play_training` starts
the next PPO update immediately after dispatch. Serialization can therefore
observe parameters after or during an update, and different workers can
receive different versions instead of a stable behavior-policy snapshot.

Reproduction: call `collect_rollouts_history(..., block=False)` with a CPU
policy and a fake pool that retains the queued arguments. Mutate one live
parameter under `torch.no_grad()`. `np.shares_memory` reports `True`, and
the already-queued weight array changes. This establishes the alias; the
frequency of the scheduling race in real training was not measured.

Suggested fix: copy each CPU NumPy weight array before submitting work,
including the specialist policies. Add a dispatch-isolation regression test.

### T2 — P2: History workers cannot reconstruct supported specialist shapes

Locations: `training/rollout_history.py:1091`, `:1104`, and `:1114`.

The parent trainer supports `--gt-hidden`, `--gt-n-layers`, `--pass-hidden`,
and `--pass-n-layers`, but workers instantiate the GT and passing models
with default dimensions. A multiworker run with nondefault specialist sizes
fails at `load_state_dict`. In addition, the parent supports resuming legacy
13-output wish heads by resizing its model, while workers always construct
the current 14-output head and fail loading that resumed policy.

Reproduced directly through `_worker_collect`, with rollout execution
mocked so no games were played:

- GT width 32: checkpoint `[32, 56]` versus worker `[64, 56]` mismatch.
- Passing width 64: checkpoint `[64, 60]` versus worker `[128, 60]` mismatch.
- Legacy wish head: checkpoint `[13, 32]` versus worker `[14, 32]` mismatch.

Suggested fix: pass the specialist architecture configuration to workers
and include it in cache identities; apply the same wish-head compatibility
handling as the parent before loading weights.

### T3 — P2: Embed resume restores classifier moments into a new GT head

Location: `training/train_embed.py:86`.

Loading an old GT classifier intentionally discards `head.*` and initializes
a new declare-advantage regression head. Its parameter shapes match the old
classifier, so the subsequent unconditional optimizer load silently maps old
Adam moments and step counts onto the new head. The history resume path
already guards against this migration. The separate embed GT attachment path
has the same failure and was included in the fix.

Reproduction: save a synthetic legacy classifier checkpoint with an Adam
`exp_avg` of 9 for its head bias, then call the embed `load_checkpoint`.
The loaded GT policy reports `migrated_from_classifier=True`, while the new
`adv_head.bias` has the stale `exp_avg=[9.0]`. No optimization was run.

Suggested fix: skip GT optimizer restoration when
`gt_policy.migrated_from_classifier` is true, matching history resume.

### T4 — P2: Fixed-head workers ignore architecture and device

Location: `training/rollout.py:173`.

The fixed-head trainer exposes a custom hidden width, but each worker creates
a default `TichuPolicy()` and loads the caller's state. A nondefault model
therefore fails before collecting its first rollout. The worker also never
moves the reconstructed model onto `device_str`, so a CUDA worker would
receive CUDA observations while keeping the model on CPU.

The architecture failure was reproduced with a 16-wide, one-layer policy:
`_worker_collect` raised missing-key/size-mismatch errors. The device problem
is established by inspection; no CUDA execution was performed.

Suggested fix: transmit the policy architecture and construct the worker
with it, then move the model to the requested device before collection.

## Confirmed distribution findings

### D1 — P1: Lightweight source bundles omit a required runtime module

Location: `distribution/source_bundle.py:29` (`RUNTIME_FILES`).

`training/observations.py` is absent from the copied runtime, although the
web/advisor modules import it. The Sol reviewer staged `_copy_runtime()` in
`/tmp` and imported `training.web` from that isolated directory. Import failed
with `ModuleNotFoundError: No module named 'training.observations'`. The
mandatory smoke test therefore aborts Linux, macOS, and Windows source ZIP
builds. Existing bundle tests mock the builders and miss the real import graph.

Suggested fix: include the module and test importing the actual staged runtime.

### D2 — P1: Python packaging names a nonexistent build backend

Location: `pyproject.toml:23`.

The configured backend is `setuptools.backends._legacy:_Backend`. Importing
that module with the project's interpreter and setuptools 70.2.0 raises
`ModuleNotFoundError: No module named 'setuptools.backends'`. Packaging tools
cannot load this backend to obtain build metadata or build the package.

Suggested fix: use the real setuptools backend (`setuptools.build_meta`, or
its `__legacy__` backend where appropriate), then verify package discovery
and included runtime assets. This audit did not build a wheel or install it.

### D3 — P2: Checkpoint-dependent Kaggle workflows permit missing input

Locations: `kaggle/run.sh:33`,
`kaggle/prepare_optimizer_batch_sweep_bundle.py:89`,
`kaggle/prepare_optimizer_runtime_bundle.py:92`, and
`kaggle/prepare_replay_age_bundle.py:94`.

The launcher accepts `optimizer-sweep`, `optimizer-runtime`, and `replay-age`
without `--resume`. Their generated kernels require exactly one checkpoint
under `/kaggle/input`. With no checkpoint dataset attached, the remote run
will fail its explicit zero-checkpoint check. This was traced through source;
no kernel was pushed or executed.

Suggested fix: validate required checkpoint input locally before preparation
or upload, and document the required invocation for each workflow.

### D4 — P3: Documented entry commands do not match this checkout's environment

Locations: `README.md:10`, `README.md:20`, `README.md:44`, and
`kaggle/push_v4_fresh.sh:11`.

README commands use `./rl_env/bin/python`, but this checkout has no such
interpreter. The configured interpreter is
`python`. The legacy v4-fresh launcher also
calls bare `kaggle`, which is absent from the current PATH, while the CLI
exists in that environment. These are environment-dependent entry failures.

Suggested fix: document environment activation consistently and resolve the
CLI from the configured environment, as the newer launchers already do.

## Confirmed interface findings

### I1 — P1: A rejected advisor move can mutate the simulated hands

Location: `training/advisor.py:280` (`force_play`).

Opponent cards are swapped into the simulated hand before `gs.play()`
validates the move. On failure, those swaps are retained. Terminal/browser
callers report the error while keeping the changed state. The Sol reviewer
reproduced an invalid two-card move returning `False` while hands changed
from `[{55}, {0,1,2}, {14}, {28}]` to
`[{55}, {0,14,28}, {2}, {1}]`. A requested Tichu declaration can also persist
after rejection because the game engine records it before validating cards.

Suggested fix: validate/reconcile on a copy and commit only successful moves,
or restore the complete state when validation fails. Test rejection without
any changes to hands, declarations, history, or provenance.

### I2 — P2: Fixed-head policies crash advisor suggestions

Location: `training/advisor.py:635` (browser caller: `training/web.py:1647`).

The fixed-policy branch assigns the `(logits, value)` result of `forward()`
to `logits` and then uses tensor indexing on that tuple. A direct call with
`TichuPolicy` raises `TypeError: tuple indices must be integers or slices,
not tuple`. Browser state generation uses the same helper on the human turn.

Suggested fix: unpack `logits, _ = policy.forward(obs, mask)` and cover a
fixed-head checkpoint in advisor/browser inference tests.

### I3 — P2: The LLM play helper cannot infer with some accepted model types

Location: `llm_play.py:149` (`ai_play`).

The shared loader accepts fixed-head and counted-trick-points checkpoints,
but this helper always uses the embedding-policy call signature and a
196-element static observation. Direct checks reproduced a fixed-head
`TypeError` (six positional arguments instead of three) and a counted-points
matrix-size error (`1x196` versus `200x256`). Its legal-action enumeration
also predates explicit Phoenix policy choices.

Suggested fix: reuse the model-aware action selection in `training.play`
and `encode_policy_static`, preserving all components of the selected action.

### I4 — P2: The top-level play CLI still treats actions as card lists

Locations: `play.py:96`, `play.py:166`, and `play.py:170`.

The engine returns `(cards, call_tichu)` tuples, but display/classification
code passes the whole tuple to card helpers. A seeded direct run failed on
its first display with `TypeError: '>=' not supported between instances of
'list' and 'int'`.

Suggested fix: unpack actions consistently, including human display and
Mahjong checks, or redirect this entry point to the maintained play CLI.

### I5 — P2: Advisor setup can contradict known Mahjong passing provenance

Locations: `training/web.py:791` and `training/advisor.py:744`.

The setup flow accepts any opponent as the Mahjong holder, even when the
human's recorded outgoing pass identifies its actual recipient. The reviewer
reproduced Mahjong moving from P1 to P2 while the provenance still recorded
that it was passed to P1. The approximate hands, opening player, and known
passing information then disagree.

Suggested fix: derive the holder from known outgoing provenance when possible,
or reject a conflicting answer before changing hands or the opening player.

## Confirmed game-engine findings

### R1 — P1: Phoenix substitutions are missing from legal actions

Locations: `tichu_env/game_state.py:1283` and `:1337`.

Straight/steps enumeration decides whether Phoenix is needed from every card
in the hand. When a natural card or pair is available, it omits combinations
that use Phoenix instead and retain a natural card. Both the ordinary and
policy action APIs consequently exclude moves accepted by the classifier.

The main session independently reproduced these exact card-index cases on
a leading turn:

| Hand | Selected cards | Classifier | Generated by either action API |
| --- | --- | --- | --- |
| `[0,1,2,15,3,4,54]` | `[0,1,54,3,4]` | `STRAIGHT` | No |
| `[0,13,1,14,54]` | `[0,54,1,14]` | `STEPS` | No |

These are 2–6 with Phoenix replacing the retained 4, and consecutive pairs
of 2s/3s with Phoenix replacing a retained 2. The reviewer also found omitted
classified subsets in 23 of 100 sampled hands; this is an audit sample,
not a prevalence estimate over gameplay.

Suggested fix: enumerate substitutions within the chosen combination,
including ranks whose natural cards remain in the hand. Compare generated
actions against classified subsets in focused completeness tests.

### R2 — P1: Phoenix is accepted as Mahjong/rank 1

Locations: `tichu_env/combinations.py:127`, `:212`, and `:345`.

The classifier accepts `phoenix_as=1`, and policy actions expose
`[2,3,4,5,Phoenix]` with Phoenix representing either 1 or 6. The classifier
also accepts `[Mahjong,Phoenix]` as a pair, and direct `GameState.play()`
accepts it. The publisher specifies substitution for normal cards; normal
cards range from 2 through Ace. Thus representing Mahjong/rank 1 is outside
that rule. See the [publisher's rulebook, pages 1–2](https://www.riograndegames.com/wp-content/uploads/2013/02/Tichu-Rules.pdf).

Suggested fix: restrict substituted ranks to 2–14, require physical Mahjong
for a straight beginning at 1, and reject special-card pairs/triples/steps.

### R3 — P2: A Dragon gift does not advance the history trick counter

Location: `tichu_env/game_state.py:706`; compare normal awards at `:493`.

`give_dragon_trick()` clears the trick without incrementing `trick_num`.
The reviewer reproduced the Dragon trick and the following trick sharing
ID 0, with play-log IDs `[0,0,0,0,0,0]`. History features therefore give
different tricks the same index.

Suggested fix: advance the history trick counter when resolving the gift,
consistent with ordinary trick completion.

### R4 — P2: The PettingZoo wrapper skips Mahjong wishing

Locations: `tichu_env/env.py:219` and `:226`.

The selected action is `(cards, call_tichu)`, but Mahjong membership is
checked against that outer tuple. Playing the generated `([52], False)`
action left `_pending_wish=False` and immediately advanced to the next seat.
The main session also confirmed this incorrect tuple membership by inspection.

Suggested fix: unpack the card list before detecting Mahjong and verify the
extra wishing turn through the public environment interface.

### R5 — P2: Card-play APIs do not enforce the game phase

Locations: `tichu_env/game_state.py:528` and `:918`.

After `reset(with_calling=True)`, the reviewer obtained 19 ordinary actions
for player 0 and successfully played an owned card while the game was still
in `GRAND_TICHU`. The phase checks therefore allow direct callers to play
before declarations and dealing are complete; the same missing guard applies
to passing. Normal callers that respect phases avoid the problem.

Suggested fix: require the play phase when generating/executing card plays.

### R6 — P2: Invalid actions can commit declarations or duplicate cards

Locations: `tichu_env/game_state.py:535` and `:545`.

The engine commits a requested Tichu declaration before validating the card
action. A two-rank pair rejected with `False` still changed declarations to
`[1,0,0,0]`; this also contributes to I1. Separately, membership validation
uses a set but classification uses the original list. An action repeating
one physical five twice was accepted as a pair while removing only one card.
These malformed actions are not emitted by the ordinary action generator,
but can reach the direct/manual input APIs.

Suggested fix: reject duplicate indices and validate the full action before
committing its declaration, hand changes, or history.

### R7 — P3: History observations can exceed their declared space bounds

Locations: `tichu_env/game_state.py:478` and `tichu_env/env.py:91`.

The encoded history uses `trick_num / 25`, while the observation `Box`
declares an upper bound of 1. The reviewer ran a deterministic singleton/pass
game that reached 41 tricks and produced 1.64. Consumers validating against
the advertised space can reject such observations.

Suggested fix: normalize by a valid upper bound or widen the declared space.

## Verification limits by area

The distribution reviewer inspected builders, launchers, Kaggle/Colab source
and generated metadata, packaging tests, and documented commands. Static
Python compilation, shell syntax checks, and metadata parsing succeeded;
those checks do not establish remote or platform-native execution.

The interface reviewer ran 51 relevant tests (all passed), plus direct
controller/model checks. A live HTTP-server reproduction could not bind a
socket in its sandbox; HTTP behavior and browser rendering were not validated
end to end. No HTTP-only suspicion is included as a confirmed finding.

The game reviewer read all `tichu_env/` source, focused tests, and principal
callers; 147 focused tests passed. Targeted state constructions and the
100-hand subset comparison supplemented those tests. This was not an
exhaustive traversal of all game states. The main training review emphasized
worker restoration, asynchronous dispatch, checkpoint migration, PPO buffer
handling, and evaluation orchestration; it was not a proof of learning or
statistical correctness for every experiment module.

The initial audit changed only this report. The subsequent authorized
remediation changed source and tests as summarized above. All three Herdr
reviewers have finished and remain available in their panes.
