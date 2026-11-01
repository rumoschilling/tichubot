"""Training hyperparameters for PPO self-play."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ── Paths ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    log_file: str = "logs/train.csv"

    # ── Architecture ──────────────────────────────────────────────────────
    hidden_dim: int = 256
    n_layers: int = 2
    embed_dim: int = 64  # query/key dimension for action-embedding model
    action_hidden_dim: int = 128
    counted_trick_points: bool = False
    trunk_dims: tuple[int, ...] | None = None
    mlp_activation: str = "tanh"
    mlp_layer_norm: bool = False

    # ── History encoder ───────────────────────────────────────────────────
    hist_d_model: int = 256
    hist_nhead: int = 8
    hist_n_layers: int = 2
    hist_ffn_dim: int = 512
    hist_proj_gain: float = 0.1
    hist_lr_mult: float = 1.0
    freeze_static_trunk: bool = False
    hist_norm_first: bool = False
    # History-branch norms, on for every model since 2026-10-25. Without them
    # the pre-norm encoder's output grew without bound and drowned the static
    # features. Checkpoints trained without them are tagged "noLN"; their
    # architecture is read from their weights (checkpoints.history_norm_flags).
    hist_final_norm: bool = True  # closing LayerNorm on the encoder output
    hist_pool_norm: bool = True  # LayerNorm on each pooled history vector before fusion
    hist_activation: str = "relu"
    hist_checkpointing: bool = False
    split_actor_critic_history: bool = False
    static_attention: bool = False

    # ── PPO ───────────────────────────────────────────────────────────────
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    # Magnetic Mirror Descent for the play policy (training/mmd.py). When
    # mmd_alpha > 0 the play loss uses alpha * KL(pi || mixed magnet) instead
    # of entropy_coef * H; entropy_coef (and its schedule) then only applies
    # to the pass net, GT net and dragon/wish heads.
    mmd_alpha: float = 0.0
    mmd_epsilon: float = 0.2
    mmd_window_updates: int = 100
    mmd_min_round_games: int = 200_000
    mmd_max_round_games: int = 3_000_000
    mmd_eval_deals: int = 16_384
    mmd_eval_procs: int = 10
    mmd_reference: str | None = None  # fixed checkpoint each round end is also scored against
    entropy_coef_end: float | None = None  # if set, linearly decay to this value
    entropy_decay_games: int | None = None  # over this many games
    entropy_decay_start: int | None = None  # absolute game count where decay begins (default: resume point)

    max_grad_norm: float = 0.5
    n_epochs: int = 2  # gradient epochs per rollout batch
    batch_size: int = 1024  # forward/backward memory microbatch
    # Transitions averaged into one main-policy optimizer step. ``None`` keeps
    # the legacy full-rollout accumulation behavior. This is independent from
    # ``batch_size`` so padding/VRAM tuning does not alter learning semantics.
    optimizer_batch_size: int | None = None

    # ── Rollout ───────────────────────────────────────────────────────────
    games_per_update: int = 256  # complete games collected before each PPO update
    use_rollout_bucketing: bool = True
    use_ppo_bucketing: bool = True
    use_action_vec_cache: bool = True

    # ── Training ──────────────────────────────────────────────────────────
    total_games: int = 10_000_000
    max_hours: float | None = None  # wall-clock cap; graceful checkpoint-and-stop when reached

    # ── Checkpointing / logging ───────────────────────────────────────────
    checkpoint_every: int = 10_000  # games
    log_every: int = 512  # games (rounds to update boundaries)

    # ── Parallelism ───────────────────────────────────────────────────────
    n_workers: int = 1  # worker processes for game collection (1 = no multiprocessing)

    # ── Passing network ───────────────────────────────────────────────────
    pass_hidden_dim: int = 128
    pass_n_layers: int = 2
    pass_lr: float = 3e-4
    pass_every: int = 10_000  # update pass net every N play games (on decisions harvested from self-play)
    pass_deals: int = 50_000  # standalone bootstrap trainer (train_passing.py) only

    # ── Grand Tichu network ───────────────────────────────────────────────
    gt_hidden_dim: int = 64
    gt_n_layers: int = 2
    gt_lr: float = 3e-4
    gt_every: int = 10_000  # train GT network every N play games

    gt_deals: int = 25_000  # 25k deals = 100k player-decisions per GT-training round
    train_gt: bool = False

    # ── Grand Tichu shared-control probes (probe/adapt duality) ───────────
    # When gt_probe_every > 0, the GT head learns ONLY from dedicated probe
    # blocks: every gt_probe_every self-play games, training pauses and
    # gt_probe_games game-equivalents of probe groups are collected — each
    # group is the same deal 5 times (deepcopy before any decision): 1
    # all-quiet control + 4 branches with exactly one seat's declaration
    # forced on. Sharing the control gives 4 marginal measurements and 4
    # declared practice rounds per 5 rollouts (the old one-seat-toggle
    # pairing got 1 of each per 2 rollouts). The head updates on the four
    # control-differenced diffs per group. Between blocks the head is frozen
    # and rollouts declare purely on-policy (no explore eps), so the
    # ecosystem adapts to a stationary declaration policy.
    gt_probe_every: int = 0  # 0 = disabled (harvest-based GT training)
    gt_probe_games: int = 200_000  # game-equivalents per probe block (groups = /5)
    gt_probe_chunk: int = 2_000  # games per collect between head updates (%5==0)
    # Practice channel: after each chunk's head update, run the standard play
    # PPO update on that chunk's rollout buffer. The sole-declarer branches
    # are the only meaningful source of declared-round practice for the
    # play/value nets once declares are on-policy-rare ("crazy person"
    # tradeoff, accepted 2026-07-17: without it, declared-round play skill
    # freezes and declaring can never become +EV). Groups stay valid — all 5
    # branches of a deal are always collected under the same play weights;
    # play only steps between chunks.
    gt_probe_practice: bool = True
    # Random eps-declares (rollout_history.GT_EXPLORE_EPS) in the main self-play
    # stream. Scaffolding for the never-call deadlock (2026-07); off once the
    # head calls on its own, so declarations in play data are sensible ones.
    gt_explore: bool = True

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cpu"
    use_amp: bool = False
    synchronous: bool = False
    use_lazy_gpu_loading: bool = False
    worker_device: str | None = None
