# Pure-PPO Information-State Architecture Study

Date: 2026-08-03 to 2026-08-04

**Status, checked 2026-11-02:** the observer-conditioned event-memory model
recommended here has not been built; this remains a design study. Two of the
three limitations below were later reduced by smaller changes to the existing
model: the action score is no longer additive over cards (nonlinear action-key
encoder, 2026-08-12), and history pooling is now conditioned on the current
state (`--static-attention`), with separate actor and critic history pooling
(`--split-actor-critic-history`). The pooled history is still shared by every
legal action, and the history tokens are still not information-complete.

## Executive decision

A substantially stronger pure-PPO play model is worth testing before adding
auxiliary reconstruction or hidden-hand losses. The clean experiment is not a
wider version of the current history model. The current design has three
separate limitations that width alone cannot repair:

1. The recorded history is not information-complete. It omits observable events
   and private memories that can affect inference and scoring.
2. The entire history is compressed by a fixed learned query before the model
   sees the current private context or candidate action.
3. The action logit is exactly additive over the cards in an action, so the
   policy cannot represent general interactions among cards in a combination.

The recommended replacement is an observer-conditioned causal event memory.
Each observer gets a separately redacted sequence containing their own raw
private information and the complete public event stream. Shared lower causal
blocks encode basic event/card relations, followed by separate actor and critic
causal refinement blocks. Legal-action queries and critic queries then attend
directly to their respective entire valid prefixes. No learned state is forced
through one pooled vector, and legal combinations are scored with a nonlinear
permutation-invariant set decoder.

The first experiment should use only the existing PPO policy loss, entropy
term, and terminal-return value loss. It should retain raw state that the actor
can directly inspect now, especially its current hand, but not played-card or
captured-score summaries. It should not use dropout, ownership targets,
reconstruction targets, or other auxiliary losses. Frozen probes are evaluation
tools only.

The first implementation target is width 128, four heads, FFN 384, two shared
causal blocks, and one actor plus one critic refinement block: approximately
1-1.5 million parameters including decoders. The width-192 `2+2+2` model is a
scale candidate only after the reference passes correctness and T4 benchmarks.

This is still an inductive bias. There is no bias-free architecture. The
intended boundary is to encode the game's observable entities, chronology,
causality, and set symmetries without telling the model which strategic
conclusions to draw from them.

## Scope and meaning of "pure PPO"

For this study, pure PPO means that the play model and its representation are
updated only by:

- the clipped PPO policy objective;
- policy entropy regularization; and
- MSE between the critic and the realized team-relative round return, currently
  `(score_A - score_B) / 400` with sign flipped for team B.

The following are excluded from the first experiment:

- public-state reconstruction losses;
- captured-point prediction;
- opponent-card ownership labels;
- behavior cloning or expert targets;
- feature dropout or modality dropout;
- engineered captured-score, played-card, or historical finish summaries.

Passing and Grand Tichu are already separate learning systems. For an
architecture comparison, their policies should be held fixed so changes in
play strength are attributable to the play representation. Wish and Dragon
policies should also be held fixed initially, while their public outcomes are
recorded in the event stream.

The legal-action mask remains. It enforces the environment's action space and
does not supply privileged strategic information.

## Evidence from the present model

### The current history is not sufficient

The current 63-wide history token records only:

- the relative actor;
- the set of cards played, or a pass;
- normalized trick index; and
- whether the play opened a trick.

See `tichu_env/encoding.py:60-73` and
`tichu_env/game_state.py:455-481`.

It does not record several facts that were observed by the player. The
irrecoverable omissions are:

- the Mahjong wish selection, including a no-wish choice;
- the recipient of a Dragon trick;
- the resolved role/rank of an ambiguous Phoenix play; and
- which of the observer's cards were in the first eight-card deal versus the
  later six-card deal.

Tichu/Grand Tichu status, finishes, and many trick boundaries still exist in
the current snapshot or can be derived from the play log, so their absence is
not the same information-theoretic failure. They should nevertheless become
typed chronological events because that removes needless reconstruction
ambiguity and allows the derived snapshot fields to be tested and removed one
at a time.

The missing Dragon recipient creates a hard observation alias. Two clones can
give the same Dragon trick to the two legal recipients and subsequently
produce byte-identical current static observations, play histories, and pass
provenance even though `won_cards` differs. The recipient is written only to
`trick_history` by `give_dragon_trick()` and that structure is not encoded
(`tichu_env/game_state.py:695-725`). No amount of model capacity can recover a
public event that is absent from its input.

The missing wish is also strategically relevant after it is fulfilled and the
current wish field returns to `None` (`tichu_env/game_state.py:727-780`). A
wish is evidence about the Mahjong player's hand and intent, not just a rule
constraint while active.

The first-eight provenance is private rather than public, but it is part of
the player's perfect-recall information set. The environment discards the
held-back deal partition after Grand Tichu decisions
(`tichu_env/game_state.py:280-343`). Its strategic value is an empirical
question; correctness, rather than a claim about how a partner interprets it,
is the reason to retain it.

Adding these raw events and private memories is not an engineered scoring
shortcut. It repairs the observation contract.

### The pooling query asks the wrong question

The history encoder uses one learned constant query to pool all encoded events
(`training/model_history.py:149-190`). Only after that compression does the
model fuse history with the current hand, pass provenance, and static state
(`training/model_history.py:361-398`).

Consequently the history query cannot be conditioned on:

- the current private hand;
- which passed cards came from which player;
- the current candidate action;
- whether the actor or critic is asking; or
- which particular card/player relationship is relevant now.

It has to preserve every potentially useful fact in one vector before it
knows what will be queried. The compact model gives actor and critic separate
pool queries and fusion layers, but they still share the complete Transformer
encoder. That is useful separation, but it does not remove the compression.

### The action scorer is strictly additive

The current policy calculates:

```text
query = query_head(state)
key(a) = Linear(action_card_mask)
logit(s, a) = query dot key(a)
```

Because both maps involved in the action score are linear:

```text
logit(s, a) = q(s)^T W a
            = sum over cards c in a of q(s)^T W[:, c]
              + optional Tichu-call term
```

This is a modular set function (`training/model_history.py:426-448`). It
cannot express arbitrary synergy or interference among cards in pairs, steps,
full houses, straights, or bombs. It also makes the call/no-call logit
difference constant across every possible first play in the same state. The
choice of whether to call Tichu therefore factorizes from the choice of what
to play.

Legal-action generation guarantees validity, but it does not make additive
strategic valuation sufficient. A nonlinear action decoder is required even
if the state representation were perfect.

### History is used, but the static shortcut is stronger

The mature checkpoint is not ignoring history. On fresh rollout states,
ablating history changed the standard policy by mean KL about `0.29-0.34`,
changed the top action about 11% of the time, and moved value predictions by
roughly 31-34 points. Reversing history produced mean KL about `0.45`, showing
that chronology matters.

The played-card static mask has greater influence. Removing only that mask
from the already-trained standard model produced mean KL about `0.82-0.85`,
changed the top action about 15-16% of the time, and moved values by roughly
47-53 points. These are out-of-distribution ablations, so they measure
functional dependence rather than the quality of either representation.

One actor pooling head is particularly dominant. Across 10,000 fresh states,
removing that one head from the mature model produced mean policy KL `0.311`;
removing all four pooling heads produced `0.356`. The model technically has
multi-head pooling, but its learned policy use is much closer to a single
dominant retrieval channel.

The fusion layer has also grown a large static-input block instead of remaining
a history-led correction:

```text
checkpoint     static-block norm     history-block norm
standard             10.87                  6.77
compact               9.94                  4.46
```

These raw Frobenius norms are not comparable evidence of modality use: the
blocks have different column counts and activation distributions. They are
retained only as provenance for the earlier question about weight allocation.
Per-column normalization and activation-weight products would be needed for a
meaningful follow-up; the OOD ablations establish dependence more directly.

### Terminal value error is not only action randomness

Value errors below are rescaled by 400 into game points. The 512-round terminal
diagnostic measured roughly 29 points MAE, mean selected
terminal-action probability `0.9988`, and about 64% singleton legal-action
states. A separate probe dataset used 2,048 fresh terminal states with 410 held
out. On that held-out split the mature critic was approximately:

```text
metric                    mature critic
MAE                       28-30 points
RMSE                      about 36 points
correlation               about 0.975
```

Thus the `V(s)`-before-action distinction remains mathematically important,
but it does not explain most of the final-action error: the final action is
usually effectively deterministic.

A terminal-only ridge readout and small nonlinear readout were fit on the 1,638
training states with the existing model frozen. Both reached only about 25
points MAE on the 410-state holdout. A simple alternate readout therefore did
not close the gap. That does not prove the value head is optimal, but it shifts
attention toward missing events, lossy compression, and genuinely unobservable
variation.

For an information set `I`, squared critic error decomposes as:

```text
E[(G - v(I))^2]
    = E[Var(G | I)] + E[(E[G | I] - v(I))^2]
```

The first term is irreducible hidden-state and future-policy variance. Early
single-trajectory MAE therefore cannot diagnose architecture by itself.
Terminal deterministic subsets, observation-equivalent state groups, and
conditional rollouts are more informative.

### Current history computation repeats almost everything

A 512-round rollout contained:

```text
states                                      36,595
unique stored public event rows             36,083
event rows processed across all prefixes 1,312,699
prefix duplication                            36.38x
mean decisions per round                      71.47
95th-percentile decisions                     86
maximum decisions                             99
```

The compact shared-history buffer fixed storage and IPC duplication, but the
Transformer still recomputes prefixes for every transition during inference
and every PPO epoch. An episode-oriented causal architecture can spend some of
this redundant compute on a stronger model while remaining practical.

The present Transformer is allowed bidirectional attention inside each already
truncated prefix. It cannot simply encode the completed round once: doing so
would leak later events into earlier token representations. Strict causality is
what makes the proposed full-sequence reuse valid, not merely a buffer layout
change.

### Measurement provenance and limits

The mature diagnostics used
`kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt`
at 14,790,656 games. Compact comparisons used
`kaggle_runs/2026-08-03_082524_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch/latest.pt`
at 3,258,368 games. Their policy parameter counts are 738,289 and 402,897.
Cycle times come from `tichu-v3-final.log` and `tichu-compact-scratch.log` in
those run directories.

Local diagnostic rollouts used each bundle's trained passing/GT policies and
disabled GT exploration. The collector's deal RNG was not fully seeded, so the
reported ablations are ranges and should not be read as confidence intervals or
exactly reproducible benchmarks. Before making a training decision from them,
persist the diagnostic command/config, centralize per-game deal seeds, and
report sample counts and uncertainty.

## Information-set contract

The new input must first be correct independently of any neural architecture.
Each observer receives a separately redacted timeline. It contains all and
only what that observer was entitled to know at that time.

### Phase-timed private events

For observer `p`:

- `PRIVATE_DEAL_8(card_set)` for the first eight cards;
- `PRIVATE_GT_DECLINE` when the observer privately decides not to call;
- `PRIVATE_DEAL_6(card_set)` only after the Grand Tichu phase ends;
- `PRIVATE_PASS_OUT({relative_target, card})` for the observer's three choices;
- `PRIVATE_PASS_IN({relative_source, card})` only after passing resolves.

The sequence must never contain another player's private deal or pass. If the
architecture is initially scoped only to main play, the same private facts can
be placed in a play-start prefix, but retaining their phase provenance is still
important.

### Complete public events

The public stream should contain typed events rather than derived summaries:

- `ROUND_START` as a non-private beginning-of-sequence token;
- `GT_CALL(actor)` and `GT_PHASE_END`;
- `TICHU_CALL(actor)` or a call flag on the corresponding play;
- `PLAY(actor, card_set, resolved_phoenix_role)`;
- `PASS(actor)`;
- `WISH(actor, rank_or_none)`;
- `TRICK_WON(winner)` before any Dragon recipient decision;
- `TRICK_TRANSFER(captured_recipient)` after recipient resolution;
- `FINISH(actor, position)`; and
- explicit phase boundaries where timing or the absence of further decisions
  is public information.

Grand Tichu declines are not made public individually by the existing game
interface, so only a call and the public end of the phase belong in every
observer stream. An observer's own decline remains private. If a future ruleset
or interface explicitly announces each decline, the schema can version that as
a public event instead.

`TRICK_WON` and `TRICK_TRANSFER` distinguish the player who won a trick from the
player who physically receives its cards. They are consecutive and name the
same seat for a normal trick. For a Dragon trick, `TRICK_WON` is emitted before
the recipient decision and `TRICK_TRANSFER` afterward, possibly naming a
different seat. Neither contains a point total. The cards remain in preceding
play events, so scoring must still be learned from history.

Dog does not resolve a trick; its public `PLAY` plus the deterministic partner
lead/transfer rule is sufficient for the reducer.

The historical event tokens do not include combo type/rank/length. Those are
derivable from the card set and sequence. The resolved Phoenix role is included
because it is part of the public meaning of an otherwise ambiguous play. The
current leading combination remains in the direct snapshot as visible rule
context for both actor and critic.

Decision prefixes are explicit boundaries in the timeline, not events carrying
the sampled result. For every transition `j`, store `prefix_end[j]`, the number
of tokens in that observer's redacted stream immediately before sampling `a_j`.
It is not the public play-history length because private insertions make the four
offsets differ. Actor and critic attention may use only key positions
`< prefix_end[j]`. The resulting call, play, wish, finish, or trick transfer is
appended only after the old log probability and value have been
stored. `TRICK_WON` is already in the Dragon decision prefix because the winner
is public before choosing the recipient.
Never use a final-sequence representation as a decision state. `ROUND_START`
defines empty-history behavior and keeps every prefix nonempty. This convention
is required both for causal training and for rollout-cache equivalence.

The high-risk orderings are:

```text
GT decision       -> GT_CALL(public) or PRIVATE_GT_DECLINE
all GT decisions  -> GT_PHASE_END(public) -> PRIVATE_DEAL_6(to each observer)
pass decision     -> PRIVATE_PASS_OUT(to that observer)
all pass decisions-> PRIVATE_PASS_IN(to each observer)
Mahjong PLAY      -> wish decision -> WISH(public)
last trick PASS   -> TRICK_WON(public)
normal trick      -> TRICK_TRANSFER(public, immediate)
Dragon trick      -> recipient decision -> TRICK_TRANSFER(public)
```

`PLAY` precedes any resulting `FINISH`; all result events remain after the play
decision that caused them.

### Direct snapshot and deliberately removed summaries

Retain the raw current observations that a player can inspect without memory:

- the observer's current hand as a card set;
- the current leading card set/combination, leader, resolved Phoenix role/rank,
  passes since the lead, cards on the table, and active wish;
- visible remaining-card counts, declarations, and who has finished; and
- the decision kind. Legal candidates remain the actor's output constraint,
  while the critic receives the same directly visible rule context.

Provide the same snapshot to the actor and critic. In particular, do not let
the actor infer its hand or rule context from legal candidates while forcing
only the critic to reconstruct it. This snapshot contains no captured score or
hidden ownership. The more aggressive event-only version remains a later
ablation after the fair shared-input model works.

Do not provide the new model with:

- the all-played-cards mask;
- captured cards or point totals;
- normalized round progress; or
- summarized historical finish order or partner finish position.

All removed fields are derivable from the complete event stream. The retained
snapshot is deliberately redundant with recent events because it is directly
observable at the decision and keeps actor/critic information symmetric. The
history must still reconstruct captured tricks, call timing and surrounding
action evidence, fulfilled wishes, and evidence about unseen hands.

### Contract tests

Before model training, build a deterministic reducer from an observer timeline
to every derivable public/private state component. At every decision it must
match the environment for:

- observer hand;
- declarations and eligibility;
- current wish;
- current trick and leader;
- played-card set;
- captured-card ownership;
- remaining-card counts;
- finish order;
- current player; and
- observer pass provenance.

This reducer is a test oracle, not a model input. Additional required tests:

- the two Dragon recipients produce distinguishable observer timelines;
- `TRICK_WON` is present and `TRICK_TRANSFER` absent at the Dragon recipient
  decision prefix;
- a fulfilled wish remains recoverable from history;
- first-eight/later-six provenance survives into play;
- no opponent-private deal/pass information appears;
- actor and critic decision-context encoders receive identical raw snapshots;
- mutating every suffix event leaves all earlier logits and values bitwise or
  numerically unchanged; and
- padded tokens cannot affect valid prefixes.

## Architectures considered

### Wider current Transformer

This is the lowest-effort option and the weakest experiment. It retains the
incomplete log, fixed pool, static bypass, repeated-prefix compute, and additive
action head. A negative result would say little about pure PPO's ability to
learn a belief state. It is rejected as the primary design.

### Single-vector GRU or LSTM

A recurrent model processes history once and is computationally attractive,
but one hidden vector is an even more explicit persistent-memory bottleneck.
It may be a useful efficiency baseline, not the architecture on which to base
the main hypothesis.

### Recurrent card/player entity state

A matrix-valued recurrent state with persistent card, player, and global slots
is a strong longer-term option. Every event can update selected cards, all
player beliefs, and global state once; rollout inference is naturally cached.
It is also the most structurally biased and requires careful full-round BPTT,
state reset, and observer-private conditioning. It should remain a fallback if
generic causal memory cannot organize the required beliefs.

### Per-state card/player entity Transformer

Fifty-six card tokens, four player tokens, and global/trick tokens can
cross-attend the raw event prefix. This gives every entity a persistent output
location and makes candidate-specific retrieval natural. It is representationally
excellent but retains repeated prefix processing unless the training pipeline
is also redesigned.

### Observer-conditioned causal event memory

This is the recommended first pure-PPO architecture. It combines a relatively
generic sequence model with correct information boundaries and direct prefix
access. It provides less task-specific structure than persistent ownership
slots while avoiding the current fixed pool.

## Recommended architecture

### Four isolated observer memories

Construct four timelines per round, one for each observer. Public events are
duplicated, while private events are redacted and phase-timed independently.
The same network processes all timelines in relative-seat coordinates.

Four isolated streams are the simplest leakage-safe reference. A single
sequence containing one player's private hand would leak that hand into later
decisions made for other players. Four streams are not mathematically necessary:
a canonical public causal memory encoded once, followed by observer-private
query/refinement towers, is an important efficiency control. It is less costly
but delays private/public interaction, so it should be compared rather than
assumed equivalent.

Each observer stream is used only for that observer's policy/value decisions.
Other players' public actions are still appended to it.

### Tokenization and tied card identity

Use shared card embeddings everywhere a physical card appears:

- private deal tokens;
- private pass tokens;
- public play events; and
- candidate legal actions.

Ordinary card embeddings should expose raw rank and suit identity; special cards
receive their own identities. The same card basis must be tied across inputs
and action decoding. The current model uses separate static columns, history
projection columns, pass embeddings, and action columns, forcing PPO to learn
their equivalence repeatedly.

A multi-card event is a set. Encode it using a nonlinear DeepSets-style map:

```text
cards(event) = rho_event(sum over c in event of phi(shared_card[c]))
```

This is permutation invariant without being additive at the final event level.
Combine it with event type, relative actor/recipient, phase, and chronological
position. No captured value or strategic label is included.

Private deals are card sets as well. Pass events are sets of `(relative player,
card)` assignments. They should use the same set encoder, or card tokens with a
shared event timestamp and no within-event position, so arbitrary input order
cannot acquire meaning.

### Decision-context encoder

Encode the direct current snapshot separately at each decision, using the tied
card/set machinery for the current hand and table cards plus small embeddings
for wish, declarations, counts, finished seats, and decision kind. This is not a
second historical MLP path: it contains no all-played mask, captures, score, or
progress summary.

The resulting context conditions both sides symmetrically. Add it to every actor
candidate query and every critic latent query before prefix cross-attention.
This lets the memory answer a question about the current hand/state without
compressing the historical tokens themselves, and prevents legal candidates
from becoming the actor's private side channel.

### Causal event backbone

A reasonable reference configuration is:

```text
d_model       128
attention     4 heads
shared        2 pre-normalized causal blocks
actor         1 pre-normalized causal refinement block
critic        1 pre-normalized causal refinement block
FFN width     384
activation    GELU or SiLU
dropout       0
position      explicit public event index plus causal order
```

The four causal blocks contain roughly 660k parameters, with a likely complete
model size around 1-1.5 million. A scale candidate at width 192 with two shared,
two actor, and two critic blocks contains about 2.7 million block parameters and
likely 3-4 million total. The mature standard and compact policies have about
738k and 403k trainable parameters respectively. Benchmark both new sizes; do
not start with the larger one merely because it fits in memory.

The shared blocks produce low-level contextual memory for every token. The actor
and critic refinements each produce their own prefix memory; neither is pooled
into one state vector before the final decision-specific queries. The depth of
each section is a sizing parameter; the structural split is the invariant.

### Nonlinear candidate-action decoder

For every legal action:

1. Encode its selected card set through a nonlinear permutation-invariant set
   encoder using the tied card basis.
2. Add action kind, a learned pass token, the Tichu-call flag, the resolved
   Phoenix role where applicable, and the shared decision context.
3. Use the resulting candidate as a query that cross-attends all contextual
   event-memory tokens at or before the current decision.
4. Apply a small nonlinear candidate tower and scalar logit head.

This permits the action to retrieve card- and player-specific evidence directly
from history. Average legal-action count in a fresh rollout was `5.7`, with
median `2` and 95th percentile `19`; candidate-prefix cross-attention is
therefore practical despite rare states with hundreds of combinations.

Unlike the present dot product, the decoder can learn interactions among cards
and between the call bit and the chosen opening play.

The action identity must describe semantics, not only a card mask. Where the
same Phoenix-containing card set has multiple legal interpretations, each
`(card_set, phoenix_role, call_flag)` is a distinct candidate. The present AI
action generator keeps a historical deterministic Phoenix interpretation while
the web player can ask the human to disambiguate. Carrying the role in the new
candidate schema prevents the new policy from inheriting that hidden action-space
restriction. Carry the semantic identity through the rollout buffer as well, so
stored and current action indices remain unique during PPO.

### Critic queries and actor/critic sharing

Share raw card/event embeddings and the low-level causal blocks. Then branch
into separate actor and critic causal refinements and query towers. The
reference uses two shared blocks and one block in each branch; the scale
candidate uses two blocks in each branch.

The critic should add the decision context to several learned latent queries,
rather than use a single fixed pooling query, and attend the valid prefix. A
small relational block and nonlinear MLP reduce those latents to one value. The
actor never consumes that pooled critic representation; each candidate queries
event memory directly.

This is a deliberate compromise. Fully sharing the complete causal backbone
would allow critic updates to rewrite all actor reasoning, while fully
duplicating encoders removes the actor's only dense reward-derived
representation signal. Under pure PPO, value MSE is important supervision for
the shared low-level event semantics. Separate upper blocks provide real
reasoning specialization. A fully shared backbone is a cheaper ablation, not
the recommended split architecture.

### Specialist decisions

The first implementation should preserve fixed specialist behavior for a
controlled comparison. Passing and GT already have standalone checkpoints.
Wish and Dragon do not: their heads consume the old play model's fused features.
Run those decisions through one specified frozen mature play checkpoint and its
old encoder for every arm, or replace them with the same deterministic policy
in every arm. Do not silently attach the old heads to new features.

Disable pass/GT optimizer updates and probe/adaptation rollouts during the
architecture comparison. If the current GT exploration mixture is retained for
training coverage, freeze its coefficient and RNG schedule across arms and
disable it consistently for evaluation. Every specialist outcome must still be
logged in the new event stream.

Later, wish and Dragon choices can use the same typed candidate-query mechanism:
rank candidates for wish and recipient-player candidates for Dragon. Grand
Tichu and passing require queries at their correct information phases and must
not see later private tokens.

### Symmetry

Relative seats already enforce cyclic rotation of player identities. Arbitrary
player permutation is invalid because partnership and turn direction matter.

Suits are rule-symmetric. A shared rank/suit card representation is appropriate.
A consistent whole-round suit permutation is a valid later augmentation, but it
should not be mixed into the first purity experiment because the architecture
change is already large and the comparison should remain interpretable.

## Episode-oriented PPO dataflow

### Training

The rollout buffer should store each round's raw events once, the four observers'
private phase events, and transition-to-prefix mappings. During each PPO epoch:

1. Construct the four redacted observer sequences.
2. Causally encode each complete sequence once.
3. Gather all of that observer's decision prefixes from the boundary map.
4. Decode ragged legal candidates and critic queries with strict prefix masks.
5. Apply the unchanged PPO, entropy, and value equations.

A concrete episode buffer contract is:

```text
public_events                  typed fields, stored once per round
private_events[observer]       typed phase-timed sets/assignments
transition_game               source round
transition_observer           redaction/relative-seat anchor
transition_prefix_end         offset in that observer's constructed stream
transition_snapshot           raw current hand/table/rule context
legal_action_offsets          ragged semantic candidates
chosen_action_index           unique candidate index, including Phoenix role
old_log_prob, old_value       rollout policy outputs
return, advantage             unchanged PPO targets
```

The compiler may materialize stream tokens lazily per game batch, but the
prefix offsets and semantic candidates are immutable rollout facts.

Encoding a complete sequence once is safe only because every backbone block is
strictly causal and every decoder mask ends at the recorded pre-action boundary.
Tokens after a decision may be resident in the same tensor, but neither earlier
memory tokens nor its actor/critic queries may attend to them. Games, rather than
individual transitions, become the PPO shuffle/minibatch unit; otherwise the
shared sequence graph would be recomputed or retained across optimizer steps.
Normalize advantages once over the whole rollout, preserve per-transition loss
weighting, sum all decoder subchunk losses for a sequence batch, and backpropagate
once. Backpropagating each subchunk separately would require retaining the shared
graph and risks changing the effective weighting.

No rollout-time latent may be detached and reused for PPO training. Old action
log probabilities and values remain stored, but the current model must recompute
the complete differentiable representation for PPO ratios and value loss.

Bucket observer sequences by length and accumulate gradients across buckets, as
the current PPO implementation already does for transition minibatches.

Do not encode all 1,024 games at once. Start with 16-32 games per sequence
batch and measure 64 only if VRAM permits. At 64 games, four observers, four
heads, and sequence length 110, one layer's explicit FP32 attention matrix is
approximately 50 MB before autograd intermediates. At length 130 it is about
69 MB. Private deals and passes should remain set events, but the actual schema
length distribution must be measured before choosing a batch.

Legal candidates should ultimately remain ragged. Padding every action across
all decisions in a game batch wastes memory because the measured median is two
while rare early states exceed 300 legal actions. A flat scorer with segmented
log-sum-exp/entropy is the preferred endpoint. A transition-subchunk decoder
that retains the shared sequence graph is an acceptable reference path.

Segmented categorical math does not by itself make candidate-prefix attention
ragged. Every query still needs an `(observer_stream, prefix_end)` mapping, and
the implementation must avoid physically copying prefix keys/values per
candidate. Benchmark the event backbone, set encoder, candidate cross-attention,
critic queries, and segmented categorical separately. Arbitrary query-by-key
and padding masks can disable optimized SDPA kernels.

Before profiling, compare segmented log-sum-exp, log probability, and entropy
values and gradients with a padded `torch.distributions.Categorical` oracle.
Cover one candidate, 300+ candidates, extreme logits, rejected empty segments,
and FP32/AMP. Likewise compare candidate-prefix grouping against a simple
per-transition attention oracle; a fast wrong ragged kernel silently corrupts
PPO ratios.

### Rollout inference

For an initial correctness implementation, full prefixes can be recomputed at
each decision. That is acceptable for unit tests and a short smoke run, not a
full Kaggle leg.

The production collector should maintain four observer cache stacks per game,
each containing shared, actor, and critic causal keys/values:

- append a public event through every observer's cache stack;
- append a private event only through its observer's cache stack;
- query only the current observer's actor/critic memories for a decision; and
- reset every stack at the round boundary.

Decision queries remain external to the cached event stream. If decision-marker
tokens are introduced instead, current turn and decision kind are public and
the marker must be appended consistently to every observer stream.

This changes rollout history cost from repeated full-prefix encoding to
incremental token updates. Cache correctness must be tested against a full-prefix
reference for every decision type.

`nn.TransformerEncoder` does not expose a usable incremental cache, so the
production path needs custom pre-normalized attention blocks (preferably backed
by SDPA) and explicit cache invalidation at optimizer/load/reset boundaries. For
1,024 live games, four observers, length 110, and FP32 keys/values, backbone
caches alone are roughly 1.85 GB for the four-block width-128 reference and
4.15 GB for the six-block width-192 scale model, before projected decoder caches
and CUDA contexts. Actor candidate attention and critic latent attention must
also cache their projected prefix keys/values, or they reintroduce repeated
`O(L*d^2)` projection work at every decision. Those two caches raise approximate
minimum totals to 2.77 GB and 5.54 GB respectively at length 110, and 3.27 GB
and 6.54 GB at length 130. Measure per-worker memory and consider FP16 caches
only after equivalence tolerances are defined.

### Expected compute shape

The current collector/trainer processes approximately 36 times as many history
token instances as exist uniquely in a round because every transition carries
a longer prefix. Four observer sequences duplicate public events four times,
but each complete sequence is encoded once per PPO epoch. Holding token count
constant, projection and FFN repetition falls by about `36.4 / 4 = 9.1x`.
Actual savings will be smaller because the complete schema adds private,
declaration, wish, resolution, and finish events and because the backbone is
wider. It must be microbenchmarked rather than inferred from parameter count.

Recent compact T4 cycles were approximately 10.6-11.1 seconds of rollout plus
13.3-13.6 seconds of PPO, or roughly 42 games/s. Standard was about 9.1 seconds
of rollout plus 17.5-18.3 seconds of PPO, or 37-39 games/s. Halving trainable
parameters therefore bought only a modest cycle improvement; history execution
shape matters more than checkpoint size.

In an idealized per-layer calculation that holds the old approximately 70-token
round length fixed, replacing repeated prefixes with four dense causal observer
sequences reduces attention-matrix work by about 5.8x. End-to-end work does not:
the reference has more total causal blocks and the complete schema is plausibly
105-130 tokens. The width-192 six-block candidate is estimated to perform
roughly three times the compact model's linear/FFN arithmetic before its query
towers. It may be slower and must earn its place through equal-wall-clock
learning. There is no defensible PPO-time forecast before a T4 microbenchmark.

The action decoder adds candidate-prefix attention. On measured rollouts,
`legal_actions * history_length` averaged about `120` per transition, so this
is much smaller than worst-case legal-action counts suggest.

Use both equal-games and equal-wall-clock comparisons. A model that learns more
per game but processes too few games may still be a worse Kaggle system.

## Why pure PPO may train this representation

The critic receives the terminal team-relative return at every play decision,
not only at the end of the round. Its value loss therefore supplies a dense
reward-derived gradient to the shared event backbone.

The scoring problem is present in raw form but not solved for the network.
`PLAY` supplies card identities and `TRICK_TRANSFER` supplies physical
ownership; the value target is the only signal connecting those facts to card
points, last-hand transfer, Tichu bonuses, and the round result. Accurate
publicly determined terminal values are therefore a direct test of whether the
representation learned bookkeeping without a score feature.

The new actor gradient path is also materially better:

- selected actions update a nonlinear representation of their actual card set;
- softmax competition updates the other legal candidate queries;
- tied card identities connect a candidate to occurrences of the same cards in
  private and public history; and
- candidate-specific prefix attention retrieves evidence without requiring a
  generic state vector to anticipate every future question.

None of this guarantees that PPO will learn calibrated opponent beliefs. PPO is
free to learn a shallower sufficient policy if deeper inference does not change
advantages often enough. The purpose of the architecture is to remove obvious
information and expressivity barriers so that a negative experiment is
meaningful.

Nor is a full hidden-card posterior identifiable merely because the model is
large. Policy/value losses constrain distinctions that affect expected return;
two internal beliefs that induce the same values and action distribution are
equivalent to PPO. The realistic pure-PPO objective is an action-relevant
information state. Ownership calibration is useful evidence, not a required
internal representation theorem.

## Implementation sequence

### Stage 1: information correctness

- Introduce a typed raw event log independent of the existing fixed-width
  history cache.
- Retain first-eight/later-six private provenance.
- Record wishes, calls, Phoenix roles, phase-timed trick winner and recipient
  events, and finishes.
- Implement observer redaction and the deterministic contract reducer.
- Add alias, redaction, chronology, and suffix-causality tests.

Do not change training until this layer is proven sufficient.

### Stage 2: nonlinear action control

- Implement tied card embeddings and the nonlinear candidate set encoder.
- Represent distinct legal Phoenix interpretations as distinct candidates.
- Unit-test permutation invariance.
- Unit-test that pair/straight synergies and action-dependent Tichu-call
  differences are representable.
- Benchmark ragged decoding on real legal-action distributions.

This isolates a present hard expressivity restriction before a long run.

### Stage 3: reference causal model

- Add the observer-conditioned causal backbone and separate actor/critic query
  causal refinement blocks and query towers.
- Start with full-prefix rollout inference as the correctness oracle.
- As an intermediate systems control, sequence-batch PPO while retaining the
  non-played portions of the current per-decision observation. Prove that this
  regrouping alone preserves outputs, losses, gradients, and one optimizer
  step. This control is not the final maskless architecture.
- Then remove played-card, captured-score, and historical summaries for the
  clean pure-PPO run while retaining the direct current snapshot shared by
  actor and critic.
- For the regrouping control, compare old transition batching with new episode
  batching at identical weights/inputs. For the new model, compare full-prefix
  causal inference with full-sequence prefix extraction. Check logits,
  distributions, values, PPO ratios/KL, gradients, and one optimizer step within
  specified numerical tolerances; old/new architectures cannot match outputs.
- Run CPU and GPU smoke tests, NaN checks, and short overfitting diagnostics.
- Initialize the final actor scalar projection with small gain and zero bias;
  verify initial entropy is close to uniform over each legal set and that
  gradients already reach the candidate encoder and actor memory.
- On T4, benchmark complete end-to-end layouts rather than generic depth:
  width 96 with `1 shared + 1 actor + 1 critic`, the width-128 `2+1+1`
  reference, and the width-192 `2+2+2` scale candidate. Use game batches
  16/32/64 and FP32/AMP with the exact snapshot, candidate, critic-query, and
  production masks. Record forward, backward, peak allocated/reserved VRAM,
  and whether fused attention is used.

### Stage 4: cached rollout path

- Implement per-observer KV caches.
- Compare every cached action distribution and value against full-prefix
  inference across complete seeded rounds.
- Test GT, passing, wish, Dragon, finish/reset, finished-game compaction, model
  reload, and post-update cache invalidation.
- Profile rollout, PPO, peak VRAM, and total cycle time on the actual Kaggle GPU
  shape.
- Treat caches as an inference optimization only. PPO must always reconstruct
  a differentiable sequence under the current parameters.

### Stage 5: controlled training

- Start from scratch; the old static/history checkpoint is not structurally
  compatible.
- Store an explicit architecture/event-schema version in the checkpoint and
  reject cross-version resume rather than relying on permissive partial loads.
- Freeze the same standalone pass/GT policies and the same explicitly specified
  mature-oracle or deterministic Wish/Dragon behavior across controls.
- Save checkpoints at 100k-200k intervals.
- Run a short correctness leg before committing a 12-hour leg.

Likely new modules are `tichu_env/events.py`, `training/model_infoset.py`,
`training/rollout_infoset.py`, `training/ppo_infoset.py`, and
`training/train_infoset.py`. Existing history training should remain intact as
the control rather than being rewritten in place.

Do not launch the first long Kaggle leg until all of these are green:

- the event reducer matches seeded live states at every decision phase;
- observer-redaction and suffix-mutation leakage tests pass;
- Phoenix semantic candidates round-trip through rollout and PPO buffers;
- ragged categorical values and gradients match the padded oracle;
- full-sequence, full-prefix, and cached inference agree within recorded
  tolerances over complete rounds;
- one regrouped PPO optimizer step matches its reference weighting;
- the chosen T4 batch has measured VRAM headroom and no attention fallback
  surprise; and
- a short from-scratch run has finite losses, noncollapsed entropy, nonzero
  actor/critic/shared gradients, and improving publicly determined value error.

## Controlled experiment and falsification

Do not change event completeness, memory, action scoring, and retrieval in one
comparison. First run the following 2x2 with the same complete event log, direct
current snapshot, and one decision vector before action scoring:

```text
                         additive action score    nonlinear set score
fixed pooled memory              A1                       A2
causal prefix memory             B1                       B2
```

For the causal arms, derive the single decision vector with a
decision-context-conditioned query over the causal prefix. `A1` versus `B1` and
`A2` versus `B2` then isolate memory; `A1` versus `A2` and `B1` versus `B2`
isolate nonlinear set valuation. After that 2x2, compare causal `B2` pooling
against the recommended direct candidate-to-prefix cross-attention. That final
comparison isolates retrieval without changing the action set encoder.

All four arms must use the same semantic action space. The additive controls may
append Phoenix role and call as linear fields, so distinct interpretations stay
distinct even though card-card interactions remain additive.

Separately compare old versus complete event logging under the same `A1`
architecture. Once the best memory/action design is known, compare played-card
summary on versus off. Finally, event-only input can be compared with the
recommended direct current snapshot. Short screening runs can eliminate weak
arms before equal-budget long runs.

Use synchronous rollouts, fixed specialist policies, a centralized deterministic
per-game seed schedule, matched configuration, and both equal-game and
equal-wall-clock checkpoints. Evaluate against fixed external checkpoints
rather than only the moving self-play lineage.

### Strength and optimization metrics

- balanced-seat arena score and standard error;
- win rate as a secondary metric;
- games per second, rollout time, PPO time, and peak VRAM;
- entropy, KL, clipping fraction, actor loss, and value loss;
- gradient norms by shared backbone, actor tower, and critic tower;
- policy/value gradient cosine on shared blocks, measured diagnostically without
  gradient surgery; and
- effective rank and dead/saturated activation checks.

### Value diagnostics

- MAE, RMSE, bias, and correlation by true global actions remaining;
- singleton-action endpoints where the selected action certainly terminates the
  round, separating publicly determined double-win and normal scoring;
- states with nonterminal alternatives, where future policy adds uncertainty;
- repeated rollouts from cloned late states to isolate future-action variance;
- groups of observation-equivalent states where available to expose hidden
  uncertainty; and
- calibration bins comparing mean prediction with mean realized result.

A clone with fixed hidden hands estimates only future-policy variance. It does
not by itself integrate over hidden deals consistent with the information set;
that distinction must remain explicit.

### Frozen post-training probes

Probes are trained only after freezing a checkpoint and never feed gradients
back to the play model:

- exact prior played-card reconstruction;
- captured-card ownership and point bookkeeping;
- declaration timing/context, fulfilled-wish, and finish-order reconstruction;
- opponent-card ownership log loss and Brier score for currently unseen,
  non-own cards; and
- a constrained baseline conditioned on played cards, known directed passes,
  remaining seat capacities, and every hard ownership exclusion.

Use both linear and small nonlinear probes, and run the identical probe on an
untrained frozen encoder so probe capacity is not mistaken for learned state.
The critic latents are a well-defined probe location. The actor has no generic
state vector, so probe candidate-query outputs separately and label them as
action-conditioned; do not average them and call the result an actor belief.
Raw event tokens are unsuitable for public reconstruction because they already
contain the facts. Where valid environment transitions can produce paired
histories differing in one informative public event, also measure the resulting
candidate-logit change as behavioral evidence.

### Gates and falsification

Keep three hypotheses separate:

1. Complete causal memory improves public bookkeeping.
2. PPO learns action-relevant hidden-card beliefs beyond hard constraints.
3. Those changes improve playing strength per game and per wall-clock hour.

At 1-3 million games, report each hypothesis independently; this is a practical
sample-efficiency gate, not a general falsification of pure PPO. The mature
comparator trained for 14.79 million games:

- `H1` fails if late-round public bookkeeping and publicly determined terminal
  values remain inaccurate under frozen probes.
- `H2` fails if unseen-card ownership does not beat the constrained baseline.
  This does not prove the policy lacks an action-specific sufficient statistic.
- `H3` fails if matched-seat strength shows no improvement at equal games or
  equal wall clock with no favorable learning trend.

The primary practical pruning rule is `H3` plus compute: stop an arm that is
clearly weaker at both matched budgets and is not closing the gap. Do not keep a
weak policy merely because one ownership probe moved, and do not discard a
strong policy merely because it lacks a calibrated full posterior. History
interventions must be valid paired environment histories; shuffling events is
only an OOD dependence test, not a strength criterion.

A stronger conclusion about the pure-PPO signal requires multiple seeds and a
game budget comparable with the mature lineage.

If public bookkeeping becomes exact, optimization is healthy, matched-budget
strength is competitive, but hidden-card diagnostics repeatedly remain at the
constraint baseline, that is evidence that PPO did not reward a calibrated full
posterior. It is then reasonable to test an auxiliary belief objective as a
separate hypothesis rather than assuming more width will solve it.

## Risks and unresolved questions

- Complete public history removes known aliases but cannot make private hands
  observable. The critic must predict an expectation over compatible worlds.
- Self-play beliefs are policy-dependent and nonstationary. A convention learned
  against itself may not transfer to humans or older checkpoints.
- Mixed-policy training should later expose deployment-known relative identity
  (`same` versus `different`) for each seat to both towers. Defer it from the
  first homogeneous self-play comparison rather than encoding it as an event.
- A larger pure model may be less sample-efficient despite better asymptotic
  representation.
- Causal masking with four redacted streams is correctness-critical. One future
  token or one opponent-private pass is enough to invalidate a run.
- Softmax attention is good at retrieval but can be poor at exact counting. The
  first experiment should test whether depth and direct prefix access suffice
  before adding a recurrent accumulator or entity-state update.
- The correct balance of shared actor/critic layers is empirical. Fully shared
  reasoning risks interference; fully separate encoders remove dense critic
  shaping from the actor representation.
- The full model should not be pushed to Kaggle until local sequence-batching
  and KV-cache equivalence tests pass.

## Confidence split

Established from code and direct diagnostics:

- Dragon recipient, fulfilled wish, Phoenix interpretation, and private deal
  provenance are absent from the current learned history;
- the current card-mask action score is mathematically additive;
- the fixed history pool occurs before current context/action conditioning;
- history affects the trained policy, but the played-card snapshot affects it
  more strongly under OOD ablation; and
- current PPO repeatedly encodes overlapping prefixes.

Strong design rationale, not yet empirical fact:

- complete typed events plus direct snapshot are a fairer information contract;
- context-conditioned causal memory removes the fixed-pool bottleneck;
- nonlinear set scoring removes the action expressivity restriction; and
- shared lower plus separate upper actor/critic blocks are a reasonable
  compromise between representation shaping and gradient interference.

Unknown until implemented and measured:

- whether pure PPO learns better action-relevant hidden-hand beliefs;
- whether width 128 is enough or width 192 earns its compute;
- whether four observer streams beat a shared public-memory factorization; and
- whether the complete end-to-end system improves strength per wall-clock hour.

## Bottom line

Pure PPO is not impossible, but capacity is only useful after the observation
and output contracts are expressive. The current network is asked to infer
from an incomplete log, compress every historical question through one fixed
pool, and value combinations as sums of independent card utilities. Those are
hard restrictions, not merely insufficient width.

The recommended experiment repairs the information set, removes historical
played/scoring summaries, processes each observer's complete causal history,
and lets actor candidates and critic latents query raw contextual memory
directly. It remains reward-only learning. If that design still fails to
develop public bookkeeping
and useful hidden-card beliefs after matched-budget multi-seed training, we will
have evidence that PPO's scalar training signal, rather than this model's raw
capacity, is the next limitation to test.
