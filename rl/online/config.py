import chex

from rl.config.common import AdamWConfig, BaseTrainingConfig


@chex.dataclass(frozen=True)
class Porygon2LearnerConfig(BaseTrainingConfig):
    # ANNOTATED, so it is a real dataclass field: without the annotation
    # this was a plain class attribute — absent from .replace() and from
    # the config serialised into every checkpoint.
    num_steps: int = 5_000_000
    # Actor pool size: the machine's actor budget. Idle threads stay alive
    # and wait at the gate
    # between games (no create/destroy churn, no inference contention).
    num_player_actors: int = 12
    num_builder_actors: int = 4
    # The service baseline both eval slots play (service/src/server/eval.ts:
    # 0=random, 1=default, 2=simple_heuristic, 3=potential_mcts); the index
    # travels in the env username suffix and the name in the metric key.
    # SimpleHeuristic is the established harder control; potential MCTS is
    # experimental.
    # Both evaluation slots use the same live snapshot; thresholding only
    # changes the evaluation policy, never replay targets or training actors.
    eval_baseline: int = 2
    # Half-life, in games, of the bias-corrected smoothed winrate/margin
    # series logged alongside the raw per-game values.
    eval_smoothing_halflife: int = 200
    # Loose per-game safety bound on the actor's env loop (rng keys are
    # pre-split to this count), NOT a target length: games run to their
    # natural outcome (Showdown's turn-limit/endless-battle clauses and the
    # service's 40-turn HP-stall detector are the backstops), and chunking
    # handles any length with fixed shapes. A game that somehow exceeds
    # this bound ends with no done row; its trailing partial chunk is
    # dropped (PlayerActor.unroll).
    unroll_length: int = 1024
    # Fixed-length chunked unrolls: every stored trajectory is
    # exactly player_chunk_length transitions; games longer than one chunk
    # are split with a one-row overlap (each chunk's final row is
    # bootstrap-only — trained as row 0 of the next chunk), so train_step
    # sees ONE shape forever. Targets bootstrap at the cut from the
    # critic — with player_lambda 0.95 the direct reward horizon is ~20
    # steps, so a 64-step window changes targets only in its last third.
    player_chunk_length: int = 64
    # Fixed trailing history window stored per chunk (field-history rows;
    # the packed caches store 2x this, matching process_state's
    # max_packed_history = 2 * max_history ratio). Tokens before the
    # chunk's own first request are burn-in context for the recurrent
    # history scan — the scan starts from h0 over a trailing window, which
    # is exactly the actor's own per-step computation, so training matches
    # acting with no stored-carry staleness. Sized ~2.5x the typical
    # tokens-per-request times chunk length; the
    # player_chunk_history_underrun telemetry says when it is too small.
    player_history_length: int = 256
    # Static shape lattice for the learner batch: a CHAIN of
    # (chunk_rows, history_rows) combos, ascending in both dims, last
    # entry ALWAYS (player_chunk_length, player_history_length). Each
    # batch is trimmed host-side to the first combo that fits its actual
    # content LOSSLESSLY (T: trailing padding rows only — padding is
    # copies of the terminal step, so [-1] outcome reads survive; H: only
    # when every chunk's valid field steps and packed rows fit, so no
    # real history is ever dropped). NEVER a data-derived shape family:
    # the variants are a fixed, enumerated set —
    # len(lattice) executables, no more — and every one is precompiled at
    # startup (fail-fast: an OOM happens at launch, not mid-run).
    # ((player_chunk_length, player_history_length),) alone restores the
    # single-shape behaviour exactly. Retune the combos from the
    # player_shape_T/H logs.
    player_shape_lattice: tuple[tuple[int, int], ...] = (
        (48, 128),
        (64, 192),
        (64, 256),
    )

    batch_size: int = 4

    # Capacity limits replay age; reuse and first-use scheduling are separate.
    player_replay_buffer_capacity: int = 256
    player_replay_ratio: int = 8
    # The fresh stream: the share of each batch's
    # slots scheduled for FIRST-use chunks, oldest admitted first (1/8 is
    # one slot every two batches of four). An unavailable fresh slot falls
    # back to replay, so retention never forces an early eviction; startup
    # may use more fresh chunks; first use consumes the reuse cap. 0
    # restores uniform capped sampling exactly.
    player_replay_fresh_fraction: float = 0.125
    builder_replay_buffer_capacity: int = 512
    builder_replay_ratio: int = 10
    # Fraction of replay buffer capacity that must be filled before training
    # starts. Valid range: [0.0, 1.0]. The formula works out to
    # replay_ratio * batch_size trajectories — enough sample budget for the
    # learner's first few batches without waiting on a full buffer.
    replay_buffer_min_fill_fraction: float = (
        player_replay_ratio * batch_size / player_replay_buffer_capacity
    )

    # Dynamic replay-ratio control: a PI loop (PID-Lagrangian style — the
    # reuse cap plays the dual variable of a staleness constraint) holds the
    # effective sample size of the learner/behaviour importance ratios on
    # replayed batches (player_learner_actor_ess) above a floor by
    # adjusting the store's per-trajectory reuse cap between the bounds
    # below. player_replay_ratio above is only the initial cap. The
    # controller runs off the critical path in the wandb log worker and
    # works in velocity form on log(cap) — clamping the output is then
    # inherently anti-windup. Buffer capacity independently bounds sample
    # age (state-distribution staleness), which no ratio control fixes.
    player_replay_ctrl_enabled: bool = True
    # Floor: a replayed batch worth this fraction of a fresh one. ESS rather
    # than a KL because it is driven by the LARGE ratios, the rows v-trace's
    # truncation and the behaviour-ratio clip distort, where the sampled
    # forward KL is driven by ratios near 0, rows that merely carry a small
    # weight. A judgement call, not a calibrated boundary (LESSONS "Replay
    # controller: actor KL 0.045 -> ESS floor"). A pathology threshold, NOT
    # a desirable operating point — hence the asymmetric bounds below: the
    # controller throttles the cap below the nominal player_replay_ratio
    # when ESS falls under this, and recovers back to nominal when it
    # rises, but never raises reuse above nominal chasing the floor (staler
    # data per learner step is never a win under a strength-per-step
    # objective).
    player_replay_ess_floor: float = 0.75
    player_replay_ratio_min: int = 1
    # Upper bound of the controlled cap. Kept at the nominal ratio so the
    # controller is purely protective; raise above player_replay_ratio
    # only if learner throughput (not strength-per-step) is the priority.
    player_replay_ratio_max: int = 8
    # Velocity-form PI gains on log(cap) per controller tick, applied to the
    # normalised error (ess − floor)/(1 − floor): the lost fraction 1 − ess
    # against its ceiling 1 − floor. At ki=0.02 and one tick per
    # player_replay_ctrl_interval steps, a sustained 2× overshoot of the
    # lost fraction halves the cap in ~35 ticks.
    player_replay_ctrl_kp: float = 0.1
    player_replay_ctrl_ki: float = 0.02
    player_replay_ctrl_interval: int = 100

    save_interval_steps: int = 20_000
    league_winrate_log_steps: int = 1_000
    # How often (learner steps) the run publishes fresh live params for
    # its actors (update_live). Every interval mints a NEW params version,
    # and versions stay referenced until their in-flight games end, so this
    # directly sets the inference server's params-cache working set: a
    # longer interval keeps fewer versions live, so
    # inference_params_cache_size covers the whole working set without LRU
    # thrash. Staleness cost: at 50 the actors act on params up to ~30s
    # old, and the replay-KL controller cuts reuse if the actor KL climbs.
    main_player_update_steps: int = 50
    add_player_min_frames: int = int(4e5)
    # Backstop ("overdue") add interval. The healthy path — "dominant" adds
    # when main beats every member >0.7 — is ungated above min_frames, so
    # this clock only paces snapshots while the agent is NOT visibly
    # improving. Too short and the league fills with near-copies of main
    # (mirror play with extra staleness) and the stagnation clock goes
    # hair-trigger; 3.6e7 is ~176k steps at the live batch shape.
    add_player_max_frames: int = int(3.6e7)
    # Learner steps before the first historical snapshot joins the league.
    # Kept low enough that a ~400k step run still trains against a
    # populated league rather than pure mirror self-play, which produces
    # self-exploiting policies that don't transfer to stylistically alien
    # opponents. Setting it above num_steps is the off switch.
    minimum_historical_player_steps: int = int(1e5)
    league_size: int = 16
    # Once an add pushes the roster past league_size, the lowest-retention
    # snapshots (main's win-rate against them, less a UCB under-sampling
    # bonus) are culled down to this in one go, so the roster saws between
    # the two and main's games go to the half it does not already farm.
    # league_size itself is the old one-eviction-per-add behaviour. Read from
    # config on resume, not from the checkpoint's serialised copy.
    league_cull_size: int = 8
    manage_league_interval: int = 10
    # Disk-backed league: max materialised opponents held in RAM at once, and
    # the UCB exploration coefficient governing which stay hot.
    league_cache_size: int = 16
    league_ucb_c: float = 1.0

    # Best-response child runs (process-level: the parent run exits, the BR
    # trains as its own process, parent resumes and imports the result —
    # runs form a tree of directories under the generation root).
    # Subdirectory under ./ckpts/gen{N}/ holding THIS process's checkpoint
    # tree ("br/<tag>" for a best-response child). None = the flat parent
    # root, today's behaviour. The parent's most-recent-checkpoint scan is
    # non-recursive, so a child subtree is structurally invisible to the
    # parent's resume.
    ckpt_subdir: str | None = None
    # Frozen target checkpoint this run best-responds to. None = a normal
    # main run. When set: params-mode init from this path (fresh
    # optimiser), ALL matchmaking pinned against it (a stationary MDP —
    # the self-play self-confirming loop is broken for the run's
    # duration), league self-adds suppressed, and on every stop the
    # latest params are published into the parent's players/ dir,
    # overwriting this BR's own previous entry (the parent's league
    # always holds the latest version of the BR for a given target).
    br_target_ckpt: str | None = None
    # BR winrate stop: end the run once the payoff-table winrate against
    # the target clears this with >= exploit_ctrl_min_games_per_opponent
    # games behind it. 0.0 = off; the CLI defaults
    # a BR launched WITHOUT --num-steps to 0.7 so "train until the target
    # is beaten" is the no-flags behaviour.
    br_stop_winrate: float = 0.0
    # BR init policy — first launch of a BR child only; a resume keeps
    # whatever the subtree holds. "target" inherits the frozen target's
    # params verbatim (the probe searches only the target's own basin, and
    # its blind spot is the collapsed switch axis it inits from).
    # "head-reset" grafts a fresh-init
    # action_head onto the inherited trunk: uniform over legal cells at
    # step 0 (the flat readout's init contract), so the exploit axis has
    # full supply. "shrink-perturb"
    # interpolates every PLAYER param toward fresh init by
    # br_perturb_frac (Ash & Adams, arXiv:1910.08475).
    # "scratch" ignores the target's params entirely — recorded
    # as expected-nonviable (no curriculum against the strongest frozen
    # policy), kept as the control arm. The builder inherits under every
    # mode except "scratch".
    br_init: str = "target"
    # shrink-perturb only: 0.0 = pure inherit, 1.0 = pure fresh init.
    # The fresh component is drawn under a key folded off the lineage
    # seed (apply_br_init) — a same-seed draw is the target's own
    # ancestor and rotates nothing.
    br_perturb_frac: float = 0.5

    # RAM-attribution diagnostics (log_memory_diagnostics in
    # rl/online/training/diagnostics.py), logged through main's periodic
    # wandb logs: process RSS + OS-vs-python thread
    # census + exact replay-buffer/league-cache byte counts. 0 disables.
    # Cost per tick is one /proc read + a walk over stored trajectories' array headers —
    # negligible at this interval.
    memory_diag_interval: int = 5_000

    # Actor step-timing drain cadence (rl/environment/actor_stats.py):
    # every N learner steps the shared ActorStats sink is drained into
    # the logs as per-timer means. 10 steps is ~2.5s, ~100+ actor steps
    # across the pool — enough samples per point without a per-step
    # dict merge. It feeds the actor-step decomposition (service wait /
    # decode / history clip / inference, and the inference server's own
    # phases).
    actor_stats_log_steps: int = 10
    # Actor-side history carry (rl/online/player_actor.py, PlayerActor.unroll):
    # each request runs the history scan over only the steps SINCE the last
    # request, resumed from the carried post-window state, instead of the
    # whole window from h0 — the same function (the minGRU scan takes h0 as
    # an argument), exact up to the bf16 GEMM leading-dim class. Recompute
    # from scratch at game start, after the service rewrites past rows
    # (`history_rewrite_count`, an Illusion |replace|) or when the window no
    # longer contains the carried step. False = today's full-window request
    # on every step, bit-for-bit (the learner never carries either way) —
    # the control arm and the abort switch.
    player_actor_history_carry: bool = True
    # Where the ACTORS' forward runs (rl/online/main.py). "cpu" = every
    # PlayerActor (training and eval) and BuilderActor runs its own batch-1
    # f32 forward on the host through Agent.step_player — no inference
    # server, no queue, and the GPU stream belongs to the train step
    # alone. f32 because XLA:CPU only
    # emulates bf16; params are stored f32 either way, so the actors'
    # mu differs from the learner's bf16 recompute by bf16 rounding —
    # player_learner_actor_forward_kl is the watch. "gpu" = the batched
    # bf16 InferenceServer (rl/online/inference.py), today's path — the
    # control arm and the abort switch.
    player_actor_device: str = "cpu"

    # OOM guard (learner.py: Learner._check_oom_guard). A self-monitoring
    # safety valve, not a leak fix. Checks available system RAM every
    # oom_guard_check_interval steps; if it drops below
    # oom_guard_min_available_fraction, saves a checkpoint and stops the
    # whole process (main.py's orchestration loop treats this like a Ctrl-C
    # interrupt) rather than letting the kernel's OOM killer SIGKILL this
    # process at an arbitrary, possibly mid-write, moment. Deliberately
    # does not try to continue in the same process after triggering —
    # freeing Python objects doesn't guarantee the OS reclaims that memory,
    # so only a fresh process actually gets back to a clean memory state.
    oom_guard_enabled: bool = True
    oom_guard_min_available_fraction: float = 0.15
    oom_guard_check_interval: int = 1_000

    # Retain the established Adam moments and epsilon while changing the
    # actor estimator; small-gradient behaviour depends on epsilon too.
    player_adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-05, weight_decay=0)
    builder_adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-08, weight_decay=0)
    player_learning_rate: float = 3e-5
    builder_learning_rate: float = 3e-5
    player_clip_gradient: float = 10.0
    builder_clip_gradient: float = 10.0
    builder_ema_update_rate: float = 1e-3

    # Terminal-only reward, so gamma=1: every step of a game shares the
    # outcome and there is nothing to discount toward. Kept as a field
    # because it is a real RL knob, not because anything has moved it.
    player_gamma: float = 1.0
    # Value-target lambda. AlphaStar's own choice: TD(lambda=0.8), a
    # short (~5-step) bootstrap horizon. Lower = more bootstrapping and
    # less Monte-Carlo variance.
    player_lambda: float = 0.95

    # The replay ESS floor is fixed at player_replay_ess_floor; the
    # worst-matchup win-rate lives on in _should_add_new_player's
    # "dominant" gate, which actuates nothing.
    #
    # Both fields below serve main's VERIFICATION branch
    # (player_actor._concerning_opponents) exclusively; names kept from
    # the removed controller, which shared them.
    #
    # A historical opponent counts as a real, current weak spot when
    # main's win-rate against it is below this. 0.3 mirrors the
    # "dominant" league-addition threshold (win-rate > 0.7).
    exploit_ctrl_target: float = 0.3
    # ...AND it has this many effective games against main, so the
    # reading is trustworthy — a freshly-added (or lightly-played)
    # snapshot reads near 0.5 by construction (main vs. a near-identical
    # recent self), which looks exactly like a real hole.
    exploit_ctrl_min_games_per_opponent: float = 20.0

    builder_lambda: float = 0.99

    # The builder retains its SPO surrogate and EMA estimator.
    builder_ppo_clip_threshold: float = 0.3

    player_value_head_loss_coef: float = 1.0
    # The privileged critic: trained beside the deployable head
    # on the SAME win_returns; its CE carries this coefficient.
    player_priv_value_head_loss_coef: float = 1.0
    # The public critic: the same labels again, over the public tier only.
    player_public_value_head_loss_coef: float = 1.0
    # True routes the v-trace value bootstraps -- and therefore
    # pg_advantages -- through the privileged head. False is the exact
    # deployable-head estimator, the live fallback: the
    # run continues on the deployable estimator without a lineage break and
    # the privileged head stays an observer.
    player_privileged_targets: bool = True
    # nGPT normalised residual in the trunk (rl/model/trunk.py): each block's
    # two sub-layer steps become alpha-weighted moves on the RMS-1 sphere and
    # the block kernels are projected back to unit embedding-space vectors
    # after every update. The plain residual stream grew 20x over ijk4nyi4
    # and the input's direction was gone from the trunk output by 514k
    # (cos 0.009). False = today's forward, bit for bit. Reaches the ACTOR
    # through artifact.player_model_config_for as well -- the two sides must
    # agree or the actor runs plain residuals over alpha-trained weights.
    player_trunk_normalised_residual: bool = True
    # rl/model/config.py cfg.encoder.history_recurrence: "loop" | "stacked".
    player_history_recurrence: str = "loop"
    # PBRS as a potential channel: eta, the scale on the
    # service's unit position potential Phi (INFO_FEATURE__STATE_POTENTIAL,
    # the human-replay outcome fit). > 0 runs a second v-trace channel beside
    # the win channel -- reward gamma * Psi' - Psi with Psi = eta * Phi (0 on
    # done rows, uncentred), bootstrapped by the learner-only potential_head --
    # and adds its advantage to pg_advantages. The channel's exact value is
    # -Psi under any policy, so a fitted head makes it inert: the head starts
    # at 0 and its lag IS the shaping. The win critics never see it. 0.0
    # builds neither the head nor the channel -- today's learning rule.
    player_potential_strength: float = 0.0
    # APPO actor (RLlib appo_torch_policy.loss, FootsiesGym's parent class):
    # V-trace advantages against pi_old = old_policy_params, then SPO's
    # quadratic surrogate (arXiv 2401.16025, the builder's too) on
    # clip(mu/pi_old, 0, player_behaviour_ratio_clip) * pi_live/mu, with
    # entropy and FootsiesGym's KL(live || reference). No batch
    # normalisation of the advantage.
    player_pg_coef: float = 1.0
    # SPO's eps: the band edge 1 +/- eps where the quadratic's restoring
    # force balances the advantage -- an optimum, not a boundary, so this is
    # more permissive than PPO's clip at the same number. The number is
    # RLlib's APPO clip_param (FootsiesGym does not override), kept across
    # the objective switch so only the objective moves.
    player_ppo_clip: float = 0.4
    # RLlib target_worker_clipping: the cap on mu/pi_old, IMPACT's pull-back
    # of a stale worker's ratio before the surrogate sees it.
    player_behaviour_ratio_clip: float = 2.0
    # IMPACT Algorithm 1 line 11: pi_old <- live every t_target accepted
    # updates, a hard copy, t_target = N*K = 4*2 = 8 for its discrete tasks
    # (RLlib new-stack APPO carries the same 4 x 2 circular buffer). Also
    # our replay cap, so a chunk mostly sees one pi_old across its reuses.
    # 1 makes pi_old the pre-update live params: the clip never engages.
    player_old_policy_snap_steps: int = 8
    # FootsiesGym's fixed EMAgnet example; these are numerical defaults, not
    # a reward-scale conversion to our game (see the 2026-09-14 source audit).
    player_mag_coef: float = 0.025
    # Off: beside the uniform-KL floor below it only flattens. Its logit force
    # carries the action's own mass, so it cannot hold a floor, and per unit
    # of flattening among moves it held switching up half as well as the
    # floor does (LESSONS "Entropy bonus off — 2026-09-18").
    player_ent_coef: float = 0.0
    # KL(uniform over legal cells || policy), the one force on a cell that
    # does not vanish with its mass: the entropy and magnet floors are
    # exponential in the advantage gap and closed on switching at 0.005
    # mass on three lineages. Its floor is c / (k (Delta + c)); the formula
    # with the run's own k and Delta gives ~.015 and the measured
    # calibration (0.07 held 0.02-0.03 on the March lineage, 0.005 held
    # nothing) gives ~.05, the gap being the shared-feature and momentum
    # routes the formula omits. On this lineage .05 overshot: switch mass
    # .16 against a .03-.10 band 30k steps on, with the league winrate
    # against the 206k snapshot .55 -> .33, so the pre-registered rule
    # (above .10, halve) applies. 0 removes the term.
    player_uniform_kl_coef: float = 0.01
    # Evaluation only: prune low-probability actions in the thresholded slot.
    # Training actors and V-trace always use the full legal distribution.
    player_prune_threshold: float = 0.005
    # Weight on pre-update live parameters per accepted optimiser update.
    # FootsiesGym's fixed example uses 6e-4 / 16; 0 freezes the magnet.
    player_reg_ema_rate: float = 3.75e-5
    builder_value_loss_coef: float = 0.5
    builder_policy_loss_coef: float = 1.0
    builder_kl_loss_coef: float = 0.1
    builder_conditional_entropy_loss_coef: float = 1.0
    builder_entropy_coef: float = 0.01
    builder_entropy_prediction_normalising_constant: float = 100
    builder_entropy_advantage_scale: float = 1e-3

    builder_human_loss_coef: float = 1e-2


def get_learner_config():
    return Porygon2LearnerConfig()
