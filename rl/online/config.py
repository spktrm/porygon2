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
    # experimental. Random/Default were saturated (93%/72% at 163k steps).
    # The slate itself is fixed at two slots (rl/online/main.py, 2026-09-09):
    # `plain-t1`, the EMA params sampled exactly as the training actors
    # sample them, and `thresholded`, the same params sampled with
    # player_prune_threshold applied -- their gap prices the threshold in
    # play. The earlier T=0.5 slots and the search eval actor are gone
    # (LESSONS.md "Removal ledger — 2026-09-09 search eval actor").
    eval_baseline: int = 2
    # Every Nth eval game per thread uses the live (main) params instead of
    # the EMA target as a divergence sanity check. The target lags the live
    # params by only ~1/player_ema_update_rate steps, so alternating every
    # game (the old behaviour) logged two near-duplicate series at half the
    # effective sample size each. 0 = EMA params only.
    eval_main_params_every: int = 16
    # Half-life, in games, of the bias-corrected smoothed winrate/margin
    # series logged alongside the raw per-game values.
    eval_smoothing_halflife: int = 200
    # Loose per-game safety bound on the actor's env loop (rng keys are
    # pre-split to this count), NOT a target length: the service's
    # MAX_REQUEST_COUNT force-tie at 96 requests was removed alongside the
    # chunked-unroll change (2026-08-16) — games now run to their natural
    # outcome (Showdown's turn-limit/endless-battle clauses and the
    # service's 40-turn HP-stall detector are the backstops), and chunking
    # handles any length with fixed shapes. A game that somehow exceeds
    # this bound ends with no done row; its trailing partial chunk is
    # dropped (PlayerActor.unroll).
    unroll_length: int = 1024
    # Fixed-length chunked unrolls (2026-08-16): every stored trajectory is
    # exactly player_chunk_length transitions; games longer than one chunk
    # are split with a one-row overlap (each chunk's final row is
    # bootstrap-only — trained as row 0 of the next chunk), so train_step
    # sees ONE shape forever instead of a geometric bucket family (each
    # bucket was a separate compiled variant with its own workspace; the
    # first top-bucket batch ~20min into a session is what OOM'd
    # 1786537634, the Aug-15 03:26 run, and the Aug-15 23:33 run alike).
    # Targets bootstrap at the cut from the critic — with player_lambda
    # 0.8 the direct reward horizon is ~5 steps, so a 64-step window
    # changes targets only within a few steps of the boundary.
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
    # Static shape lattice for the learner batch (2026-08-20): a CHAIN of
    # (chunk_rows, history_rows) combos, ascending in both dims, last
    # entry ALWAYS (player_chunk_length, player_history_length). Each
    # batch is trimmed host-side to the first combo that fits its actual
    # content LOSSLESSLY (T: trailing padding rows only — padding is
    # copies of the terminal step, so [-1] outcome reads survive; H: only
    # when every chunk's valid field steps and packed rows fit, so no
    # real history is ever dropped). This is NOT the geometric bucket
    # family that OOM'd three runs: that compiled a data-derived variant
    # per shape, with the first top-bucket batch arriving as a SURPRISE
    # compile ~20min in. Here the variants are a fixed, enumerated set —
    # len(lattice) executables, no more — and every one is precompiled at
    # startup (fail-fast: an OOM happens at launch, not mid-run).
    # ((player_chunk_length, player_history_length),) alone restores the
    # single-shape behaviour exactly. Combos chosen from the Aug-20
    # measurement (batch_size 4): batch-max chunk fill mean ~42 of 64,
    # history fill mean ~85 of 256 — retune from the player_shape_T/H
    # logs.
    player_shape_lattice: tuple[tuple[int, int], ...] = (
        (48, 128),
        (64, 192),
        (64, 256),
    )

    # Batch iteration params
    batch_size: int = 4

    # Replay buffer params
    # Kept small on purpose: steady-state throughput is set entirely by
    # replay_ratio (samples per trajectory), so capacity only controls how
    # stale a trajectory is when sampled. 256 keeps mean sample age well
    # inside one EMA-target time constant (1/player_ema_update_rate steps).
    player_replay_buffer_capacity: int = 256
    player_replay_ratio: int = 8
    # The fresh stream (2026-09-08, 224582c): the share of each batch's
    # slots scheduled for FIRST-use chunks, oldest admitted first (1/8 is
    # one slot every two batches of four). An unavailable fresh slot falls
    # back to replay, so retention never forces an early eviction; startup
    # may use more fresh chunks; first use consumes the reuse cap. 0
    # restores uniform capped sampling exactly. Reads: LESSONS.md "Fresh
    # replay stream and decision accounting — 2026-09-08" and the
    # 2026-09-09 first-18.1k-update read.
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
    # measured learner-vs-behaviour KL on replayed batches at a setpoint by
    # adjusting the store's per-trajectory reuse cap between the bounds
    # below. player_replay_ratio above is only the initial cap. The
    # controller runs off the critical path in the wandb log worker and
    # works in velocity form on log(cap) — clamping the output is then
    # inherently anti-windup. Buffer capacity independently bounds sample
    # age (state-distribution staleness), which no ratio control fixes.
    player_replay_ctrl_enabled: bool = True
    # Ceiling: the actor-KL level the buffer-capacity plateau diagnosis
    # identified as the healthy/stale boundary. This is a pathology
    # threshold, NOT a desirable operating point — hence the asymmetric
    # bounds below: the controller throttles the cap below the nominal
    # player_replay_ratio when KL exceeds this, and recovers back to
    # nominal when it drops, but never raises reuse above nominal chasing
    # the ceiling (staler data per learner step is never a win under a
    # strength-per-step objective).
    player_replay_kl_target: float = 0.045
    player_replay_ratio_min: int = 1
    # Upper bound of the controlled cap. Kept at the nominal ratio so the
    # controller is purely protective; raise above player_replay_ratio
    # only if learner throughput (not strength-per-step) is the priority.
    player_replay_ratio_max: int = 8
    # Velocity-form PI gains on log(cap) per controller tick, applied to the
    # normalised error (kl_target − kl)/kl_target. At ki=0.02 and one tick
    # per player_replay_ctrl_interval steps, a sustained 2× KL overshoot
    # halves the cap in ~35 ticks.
    player_replay_ctrl_kp: float = 0.1
    player_replay_ctrl_ki: float = 0.02
    player_replay_ctrl_interval: int = 100

    # Self-play evaluation params
    save_interval_steps: int = 20_000
    league_winrate_log_steps: int = 1_000
    # How often (learner steps) the run publishes fresh live params for
    # its actors (update_live). Every interval mints a NEW params version,
    # and versions stay referenced until their in-flight games end, so this
    # directly sets the inference server's params-cache working set: at 10
    # (~6s of main training), main alone kept 5-10 versions live at once;
    # 50 (~30s) collapses that to ~2, letting inference_params_cache_size
    # =12 cover the whole working set without LRU thrash. Staleness cost:
    # actors act on params up to ~30s old — measured actor-KL is 0.005-
    # 0.006 vs the 0.045 replay target, ~5x headroom, and the replay-KL
    # controller cuts reuse if that ever stops being true.
    main_player_update_steps: int = 50
    add_player_min_frames: int = int(2e5)
    # Backstop ("overdue") add interval. The healthy path — "dominant" adds
    # when main beats every member >0.7 — is ungated above min_frames, so
    # this clock only paces snapshots while the agent is NOT visibly
    # improving. At 3e6 (~11.5k steps) it filled the league with
    # ~0.5-winrate near-copies of main (mirror play with extra staleness)
    # and made the stagnation clock hair-trigger. 9e6 (~44k steps at the
    # live batch shape) fired every add of irqeetfg to 640k — the dominant
    # gate never did — and was doubled 2026-09-04 (~88k steps) together
    # with the batch cull below.
    add_player_max_frames: int = int(1.8e7)
    # Learner steps before the first historical snapshot joins the league.
    # Kept low enough that a short (~200k step) run still trains against a
    # populated league rather than pure mirror self-play — mirror-only runs
    # measured 93% vs Random but ~10% vs SimpleHeuristic at 163k steps,
    # the signature of self-exploiting policies that don't transfer to
    # stylistically alien opponents.
    minimum_historical_player_steps: int = int(5e4)
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
    # games behind it (n=20 puts the SE at ~0.11, so 0.7 is a ~1.8 SE
    # signal — the old promotion-bar lesson). 0.0 = off; the CLI defaults
    # a BR launched WITHOUT --num-steps to 0.7 so "train until the target
    # is beaten" is the no-flags behaviour.
    br_stop_winrate: float = 0.0
    # BR init policy — first launch of a BR child only; a resume keeps
    # whatever the subtree holds. "target" inherits the frozen target's
    # params verbatim (the pre-2026-08-30 behaviour — the probe searches
    # only the target's own basin, and its blind spot is the collapsed
    # switch axis it inits from). "head-reset" grafts a fresh-init
    # action_head onto the inherited trunk: uniform over legal cells at
    # step 0 (the flat readout's init contract), so the exploit axis has
    # full supply while the world model carries. "shrink-perturb"
    # interpolates every PLAYER param toward fresh init by
    # br_perturb_frac (Ash & Adams, arXiv:1910.08475 — the ~179k
    # perturbation is the one event observed to revive collapsed switch
    # mass). "scratch" ignores the target's params entirely — recorded
    # as expected-nonviable (no curriculum against the strongest frozen
    # policy), kept as the control arm. The builder inherits under every
    # mode except "scratch".
    br_init: str = "target"
    # shrink-perturb only: 0.0 = pure inherit, 1.0 = pure fresh init.
    # The fresh component is drawn under a key folded off the lineage
    # seed (apply_br_init) — a same-seed draw is the target's own
    # ancestor and rotates nothing. Measured calibration (2026-08-30,
    # vs ckpt_00254992): full-tree fresh/trained norm ratio 0.925, so
    # frac maps near-linearly onto direction — 0.75 lands at cos 0.40
    # to the target (0.34 predicted orthogonal; the excess is
    # structural, e.g. LayerNorm scales ~1.0 in both nets).
    br_perturb_frac: float = 0.5

    # RAM-attribution diagnostics (Learner._log_memory_diagnostics), logged
    # through main's periodic wandb logs: process RSS + OS-vs-python thread
    # census + exact replay-buffer/league-cache byte counts. Added after
    # session 1786537634's RSS climbed 5.9->17GB (threads 478->775) with
    # no way to attribute it from wandb alone. 0 disables. Cost per tick
    # is one /proc read + a walk over stored trajectories' array headers —
    # negligible at this interval.
    memory_diag_interval: int = 5_000

    # Actor step-timing drain cadence (rl/environment/actor_stats.py):
    # every N learner steps the shared ActorStats sink is drained into
    # the logs as per-timer means. 10 steps is ~2.5s, ~100+ actor steps
    # across the pool — enough samples per point without a per-step
    # dict merge. The actor-step decomposition it feeds (service wait /
    # decode / history clip / inference, and the inference server's own
    # phases) is the baseline the history-carry pass is judged against.
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
    # alone. Measured 2026-09-03 beside a live learner: per-actor CPU
    # inference 23-60 ms against 147 ms through the GPU server (76 of it
    # queue wait, none of it compute: the server's forward shared one
    # device stream with the train step). f32 because XLA:CPU only
    # emulates bf16; params are stored f32 either way, so the actors'
    # mu differs from the learner's bf16 recompute by bf16 rounding —
    # player_learner_actor_forward_kl is the watch. "gpu" = the batched
    # bf16 InferenceServer (rl/online/inference.py), today's path — the
    # control arm and the abort switch.
    player_actor_device: str = "cpu"

    # OOM guard (learner.py: Learner._check_oom_guard). A self-monitoring
    # safety valve, not a leak fix — added after 1361 crashed, though that
    # specific crash turned out to be an unrelated websocket failure to the
    # game server, not RAM exhaustion. Checks available system RAM every
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

    # NOTE the representation-health probe (dormant-unit fraction,
    # srank@0.99) moved OFFLINE 2026-08-21 — rl/model/capacity.py, run
    # against a saved checkpoint by tests/test_checkpoint_collapse.py. It
    # cost an extra encoder forward plus an eigendecomposition per probe
    # inside the train loop, and no training decision read it. The
    # per-step fresh-vs-replayed value-error gap below stays: it is
    # computed from tensors train_step already has.

    # Learning params. Player b1 back to 0.9 (2026-08-26): the b1=0
    # detour was specific to the previous prefactor-free logit force
    # (momentum carried each push ~1/(1-b1) steps past the stiff
    # equilibria its analytic shifts created — the dx65cpwp runaway).
    # The player now runs the same trust-regioned PPO surrogate as the
    # builder, the exact case the pro-momentum argument was always
    # about; NashPG's own optimiser is AdamW with default moments.
    # Player eps 1e-5 (2026-08-31): the NashPG reference explicitly
    # overrides optax's 1e-8 (`optax.adamw(lr, eps=1e-5)`) and the
    # reference-diff ledger flagged it as "the one to test" — Adam is
    # scale-invariant, so a param whose gradient has gone tiny (a starved
    # switch cell's) still steps at ~full lr along a noise-dominated
    # direction, and eps is the ONLY damper; 1e-5 engages 1000x sooner.
    # Builder keeps 1e-8: the divergence concerned the player bracket.
    player_adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-05, weight_decay=0)
    builder_adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-08, weight_decay=0)
    # 3e-5. A 1e-4 trial (Aug 2026, zany-leaf-1305) collapsed: pre-clip grad
    # norms 10-100x the clip, action-emb srank at 0.27 by 13k steps (vs
    # 0.82 at 3e-5), value CE degrading and eval regressing from ~40k —
    # all while actor-KL sat quietly at 0.002, so KL headroom is NOT
    # evidence the LR can rise (the trust region bounds per-update policy
    # movement, not representation damage).
    player_learning_rate: float = 3e-5
    builder_learning_rate: float = 3e-5
    player_clip_gradient: float = 10.0
    builder_clip_gradient: float = 10.0
    # Fast EMA target (IMPACT-style): supplies the clipped-target ratio in
    # the surrogate, the v-trace reference policy, and the value bootstraps,
    # so it must track the learner closely for stability under replay reuse.
    # (Reference systems likewise keep a fast target purely for v-trace
    # stability, separate from their slow regularisation anchors.)
    player_ema_update_rate: float = 1e-3
    builder_ema_update_rate: float = 1e-3

    # Terminal-only reward, so gamma=1: every step of a game shares the
    # outcome and there is nothing to discount toward. Kept as a field
    # because it is a real RL knob, not because anything has moved it.
    player_gamma: float = 1.0
    # Value-target lambda. AlphaStar's own choice: TD(lambda=0.8), a
    # short (~5-step) bootstrap horizon — they could afford heavy
    # bootstrapping because supervised init gave them a sane critic from
    # step one. This project starts from scratch AND the 1328 five-arm
    # sweep pointed the same direction (monotone lower-lambda-better,
    # confounded but directional), so 0.8 is adopted as-is. NOTE: the
    # lambda=1.0 MC-anchor row of the aux spectrum used to keep a live
    # bootstrap-bias readout (player_bootstrap_gap) on this
    # bootstrap-heavy target; the aux heads went 2026-08-21, so that
    # instrument is gone with them (LESSONS.md ledger).
    player_lambda: float = 0.8

    # No adaptivity/entropy controller fields anymore. The
    # AdaptivityController was removed entirely 2026-08-13 (hard to tune,
    # harder to predict — see LESSONS.md 10
    # for the bug history). Its entropy sensors are still logged from
    # train_step (player_action_normalized_entropy,
    # player_normalized_modality_entropy); modality collapse (1330 died
    # at 0.08 on that axis) is now watched on the dashboard, not
    # auto-corrected.

    # No ExploitabilityController anymore (removed 2026-08-14, the last
    # adaptive hyperparameter loop — see rl/online/controllers.py's
    # module docstring). The replay KL target is fixed at
    # player_replay_kl_target; the worst-matchup win-rate it sensed still
    # exists as _should_add_new_player's "dominant" gate, it just doesn't
    # actuate anything.
    #
    # Both fields below now serve main's VERIFICATION branch
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
    # recent self), which looks exactly like a real hole (1338: two
    # snapshots 5.5k/26.9k steps old, win-rate never left 0.48-0.54 — a
    # false positive from exactly this).
    exploit_ctrl_min_games_per_opponent: float = 20.0

    builder_lambda: float = 0.99

    # Builder policy objective: ratio-based surrogate with a trust region
    # (SPO's smooth quadratic; the player runs the PPO clip — see
    # player_pg_objective).
    builder_ppo_clip_threshold: float = 0.3

    # Loss coefficients
    ## Player
    # (`player_kl_loss_coef`, the actor backward-KL force, was REMOVED
    # 2026-09-09 -- LESSONS.md "Removal ledger — 2026-09-09 actor
    # backward-KL force".)
    player_value_head_loss_coef: float = 1.0
    # The privileged critic (2026-09-01): trained beside the deployable head
    # on the SAME win_returns; its CE carries this coefficient.
    player_priv_value_head_loss_coef: float = 1.0
    # True routes the v-trace value bootstraps -- and therefore
    # pg_advantages -- through the privileged head. False is the exact
    # pre-2026-09-01 estimator (deployable head), the live fallback: the
    # run continues on the deployable estimator without a lineage break and
    # the privileged head stays an observer.
    player_privileged_targets: bool = True
    # Belief-state shaping (2026-09-01): CE from the matched public rows'
    # belief logits to the sg'd opponent code. Bounded (<= log K per group),
    # pi-free, touches representations not logits; 0.0 is an inert-loss off
    # (predictor params stay in the tree).
    player_belief_coef: float = 0.25
    # The latent transition model (2026-09-05, rl/model/transition.py;
    # supersedes the delta dynamics head of 2026-09-03/04): g(h_t, a, z)
    # over the post-trunk policy-readable rows with a chance code z. This
    # coefficient brackets the whole term -- consistency (normalised MSE
    # of the imagined rows against the real next rows, per sequence group,
    # copy predictor = 1), grounding (the old head's pre-trunk label at
    # t+1, normalised the same way), value (the shared critic on the
    # imagined CLS row, CE to the t+1 win_returns), policy (the shared
    # readout on the imagined rows, forward KL to the sg'd target policy
    # at t+1 over the real next mask), the next-mask BCE + request kind
    # CE, done BCE, and the two KL halves below. pi-free; shapes the trunk
    # and the readout's operands. 0.25 is the retune if the temp-1 eval wr
    # falls -0.03 over the hold; 0.0 is an inert-loss off (params stay).
    player_dynamics_coef: float = 0.5
    # DreamerV3's KL balancing: the prior is pulled to the sg'd posterior
    # at dyn_coef, the posterior to the sg'd prior at rep_coef, each half
    # clipped below at free_nats per transition (summed over code groups)
    # so a code that is already predictable pays nothing. The floor is
    # sized PER GROUP: DreamerV3's 1 nat is over a 32-group code (1/32 nat
    # each), and at 1.0 over our 2 groups it sat above the KL on 93% of
    # transitions (irqeetfg 1266k-1312k: kl 0.64, kl_free_frac 0.93) --
    # zero gradient on both halves, the prior never trained
    # (prior_grad_norm -> 0, prior_post_agree 0.58 -> 0.30) and the
    # posterior drifted through the straight-through decode alone
    # (perplexity 2.35 of 16, falling). 1/32 x 2 = 0.0625. The floor
    # never goes UP.
    # rep_coef 0.1 -> 0.0, 2026-09-06 (Step 3b D): Stochastic MuZero's
    # posterior form -- no pull of the posterior toward the prior, the
    # prior still chases the sg'd posterior at dyn_coef. Triggered as
    # pre-registered: with the consistency force out and v_head live
    # (the B relaunch, 1654k-1674k) every decode-side bar stayed at its
    # launch value -- out_proj_rms 0.0168 -> 0.0178 (bar > 0.03),
    # gain_public / gain_hp_moved 0.51 / 0.56 (bars 0.528 / 0.588),
    # value_delta_r2 0.29 with the prior-mode read at -0.11 (below copy),
    # kl 0.17 flat (predicted 0.3-0.6) -- with the matched control fine
    # (player_value_head_r2 0.926). A code the rep half keeps pinned to
    # the prior can only encode what the prior already predicts. 0.1 is
    # the DreamerV3 form and the abort's restore: kl > 4 or
    # prior_post_agree -> 1/16 for 5k puts it back (a reference-form
    # toggle, never a retune). The old "posterior collapse -> rep 0.05"
    # rung is retired with it.
    player_transition_dyn_coef: float = 0.5
    player_transition_rep_coef: float = 0.0
    player_transition_free_nats: float = 0.0625
    # Latent consistency against the real next rows, over rows present at
    # either endpoint. The copy-movement normaliser and nonempty-group mean
    # are owned by transition_losses. Restored 2026-09-08 after fixing
    # absent-row amplification; measurements and removal history: LESSONS.md.
    # 0.0 retains the diagnostics while removing this gradient contribution.
    player_transition_cons_coef: float = 1.0
    # 2026-09-06, Step 3b: the shared v_head TRAINS on the imagined CLS
    # row (MuZero's value target: the real t+1 win_returns through the
    # imagined state -- the same labels the head already fits, on a wider
    # input distribution). The trunk stays unreachable (g's input is
    # sg'd) and the action readout stays FROZEN on imagined rows (launch
    # check 2's anti-switch smoothing lived in that term). False = the
    # frozen clone, bit-identical to the 2026-09-05 observer form -- the
    # abort switch if `player_value_head_r2` leaves 0.90 +- 0.02 while
    # the imagined-side bars pass. Read into the model config at the
    # learner's construction sites (the model forward branches on it
    # statically).
    player_transition_value_trains_v_head: bool = True
    # Latent actions (2026-09-07). The action encoder q(u | h, a) is held
    # to the EXACT action-discrimination objective at observed states:
    # H_w(A | U, h) under uniform reference weights over the legal cells,
    # summed over all 64 codes (a sampled straight-through CE drops the
    # derivative of the sampling distribution). The loss is bounded by
    # log(n) in expectation and invariant to a common logit shift (no
    # restoring force along that direction -- the mean logit is on a
    # panel); it reaches the encoder only. 1.0 is the initial weight, 0
    # the matched control (does the alphabet collapse without it? the
    # symmetric collapsed encoding is a stationary point of the objective,
    # not a repelled one). A persistent collapse is diagnosed, never
    # escalated by coefficient.
    player_transition_decode_coef: float = 1.0
    # The recorded action's code at an IMAGINED node (the encoder on
    # hhat_k with the recorded cell) held to its real-state distribution
    # (CE to sg q(u | h_{t+k}, a)): a state-local code is not guaranteed
    # to keep its meaning after model error, so the unroll trains it to.
    # A hypothesis with its own panels (the teacher-vs-imagined KL); 0 is
    # the control.
    player_transition_align_coef: float = 1.0

    # THE policy gradient (2026-08-26): NashPG (arXiv:2510.18183, TMLR
    # 8/2026) — a PPO-clipped surrogate on the taken action's ratio
    # pi/mu with a batch-normalised v-trace advantage from V, plus a
    # DIFFERENTIATED reverse KL(pi || pi_reg) magnet and an entropy
    # bonus, the reference hard-snapped from the target params every
    # player_reg_snap_steps. Their section 5.4 ablation is the reason
    # for the operator choice: swapping PPO into the older reward-
    # transform framework closes most of its gap in larger games, i.e.
    # the inner update rule, not the regularisation cycle, was the
    # bottleneck.
    # The whole bracket shares this coefficient; 1.0 is the reference's
    # implicit value (the advantage is unit-std by construction).
    player_pg_coef: float = 1.0
    # PPO clip epsilon (NashPG/paper Table 4). The clip is the trust
    # region: the surrogate's gradient is exactly zero once the ratio
    # leaves the band in the push direction, so no force persists at a
    # stiff equilibrium — the structural fix for the runaway class the
    # previous logit-force loss needed a force clip, centred logits and
    # b1=0 to contain.
    player_ppo_clip: float = 0.2
    # Which surrogate policy_gradient_loss runs for the player: "spo"
    # (the smooth quadratic the builder also runs, 2026-08-30) or "ppo"
    # (NashPG's own rule) for an A/B. Static config: switching costs one
    # recompile at launch.
    player_pg_objective: str = "spo"
    # Differentiated REVERSE KL(pi || pi_reg) magnet, NashPG's mag_coef —
    # their Algorithm 4 line 8 / eq. 12 verbatim, D_KL(pi_theta(.|o) ||
    # rho(.|o)) under E_{o~pi}, i.e. the OPTIMISED policy is the first
    # argument. Called "forward" here until 2026-08-26; that was wrong by
    # this package's own convention (loss.py's approx_forward_kl is the k3
    # estimator for KL(actor || learner), reference first). Reverse =
    # mode-seeking, which is exactly why it cannot refill a dropped
    # modality (the removed support-anchor family was built for that; see
    # the note below player_ent_coef).
    # alpha = 0.2 is their U-shaped sensitivity optimum (fig. 1) and
    # DeepNash's eta; never anneal it (their Appendix C: annealing alpha
    # diverges). Own-side only, as NashPG's objective also is. The
    # gradient is pi-prefactored — with the PPO surrogate there is no
    # prefactor-free refill force anywhere any more; the bet (theirs) is
    # that the magnet cycle plus entropy keep pi interior so starvation
    # never starts. switch_ratio through the 13k wire is the acceptance
    # gate; the analytic-shift form is in git history if it fails.
    player_mag_coef: float = 0.2
    # Entropy bonus, differentiated — NashPG's ent_coef verbatim
    # (2026-08-30): the plain JOINT entropy over legal cells, one static
    # coefficient. Up to a constant this is the reverse KL to uniform.
    # The per-axis split (H(macro) + H(micro|taken), 2026-08-27) and the
    # SAC-style dual temperatures holding each at a normalised target
    # (2026-08-28) are removed with the forward-KL-to-uniform term — the
    # per-level entropies survive as OBSERVER panels only
    # (loss.factorised_entropies). Revert handles in the LESSONS.md
    # ledgers.
    player_ent_coef: float = 0.01
    # DeepNash's FineTuning threshold (rnad.py FineTuning._threshold, its
    # reference value .03): a legal action whose probability is below it is
    # REMOVED and the rest renormalised (rl/model/utils.py prune_log_policy,
    # with the reference's guard that a row entirely below the line keeps
    # its legal set). Where it applies, and nowhere else: (1) the
    # `thresholded` eval slot samples the thresholded policy
    # (HeadParams.prune_threshold); (2) the learner thresholds the TARGET
    # policy entering v-trace (targets.thresholded_target_ratio), so a
    # taken action the target has dropped below the line gets rho 0 and
    # its row is discarded -- variance control on the target estimator,
    # the reference's own placement (rnad.py:798 post-processes pi for
    # v_trace only; acting_policy and the policy loss's pi stay raw). rho
    # only: the trace ratio c stays raw, the pre-registered restriction
    # the offline cut audit fired (7.8% of chunks cut before the midpoint
    # against the 5% gate; targets.compute_player_targets). The
    # learner ratio, the surrogate, the magnet, the entropy term and the
    # support hinge read the raw policies; the training actors never
    # threshold -- mu stays the policy as trained. 0.0 is bit-identical to
    # no threshold everywhere.
    #
    # Three deliberate divergences from the reference, owned: always on
    # from the restart (rnad gates it on from_learner_steps, off by
    # default, as a late strength fix for a converged policy); .005 not
    # .03, so it bites on far fewer actions; no 1/32 discretisation. The
    # support hinge below holds every legal cell at twice this line, so in
    # equilibrium the discard zone is empty -- player_discard_taken_frac
    # above 1% sustained is the hinge failing and this hiding it, the
    # whole-set revert trigger.
    player_prune_threshold: float = 0.005
    # tau (.01) is twice player_prune_threshold: an action must lose half
    # its supported mass before the v-trace threshold discards it, and that
    # factor of two is the hysteresis band. Calibration: 5-6 bench cells at
    # .01 induce a switch-mass floor of 5-6%, just under the .07171 the KL
    # was holding -- low enough that evidence, not the floor, sets the
    # resting level. Confirm against player_switch_mass_choice in the
    # first 250k fresh decisions.
    player_support_tau: float = 2 * player_prune_threshold
    # The hinge's smoothing width, in LOG-PROBABILITY space (2026-09-11, the
    # user's call on docs/porygon2_support_loss_recommendations.md; the hard
    # hinge's kink was not measured to cost anything first): each legal cell
    # scores T * softplus(log(tau_row / pi) / T). At T = .1 a cell at tau/2
    # takes .999 of the lift, at tau .5, at 1.25 tau ~.1 and at 2 tau ~.001,
    # so the term is no longer exactly silent above the line and a cell the
    # critic is indifferent to rests near 1.2-1.5 tau rather than at tau.
    # 0.0 is exactly the hard hinge (loss.support_hinge_loss, no softplus).
    player_support_temperature: float = 0.1
    # The FLAT SUPPORT HINGE (2026-09-09, loss.support_hinge_loss): over a
    # row's legal cells as flat complete actions, (1/N) sum_a max(0,
    # log(tau / pi_a)) -- every legal action held at tau, the term exactly
    # silent once all clear it. Its per-logit derivative
    # active_fraction * pi_b - below_b / N is bounded, zero-sum over the
    # row and carries no pi prefactor on the cell it lifts, so it is the
    # one force still acting on an abandoned action; above tau it says
    # nothing and the critic ranks. It REPLACES the modality-marginal
    # uniform KL (player_uniform_kl_coef .025, 2026-08-31 to 2026-09-09):
    # at a starved action the two are the same order and equally
    # pi-independent, but the KL kept pulling toward uniform modality mass
    # at every probability -- and, with CELL_MODALITY_MASK separating MOVE
    # from WILDCARD, pulled P(tera | move) toward one half. The flat form
    # imposes no hierarchy at all (LESSONS.md "Removal + addition ledger —
    # 2026-09-09 flat support hinge").
    #
    # Screened OFFLINE over recorded chunks at .001 / .0025 / .005
    # (rl/offline/support_screen.py, ckpt_02014000, 64 chunks): the encoder
    # gradient norm moved by at most +0.01% at any of them against the 10%
    # ceiling, and the hinge's directional pull on the switch logits was
    # restoring at every value (-.00008 / -.00019 / -.00039); one restored
    # Adam step lifts no exposure measurably at any coefficient (min legal
    # probability .0197 before and after, 27.3% of legal cells under .01),
    # so the cost half of the criterion decided it and the top of the
    # screened range, .005, landed for the 2026-09-09 relaunch. Raised to
    # .05 the next morning on the live directional read: at .005 the
    # hinge's pull on the switch logits was -.0005 to -.0013 against the
    # retired KL's -.0065 to -.0081 and the policy gradient's +-.02 swing,
    # and its per-cell lift (coef / N ~ coef * .15) held a cell at the
    # .005 discard line only against an adverse normalised advantage of
    # ~.15 -- it could not be tested at that size. At .05 the lift is
    # .0075 (holds against ~1.5), the switch-axis pull ~ -.009 (the KL's
    # order), and the encoder-gradient cost extrapolates to ~0.1% against
    # the 10% ceiling. Never swept in a live learner -- config is a jit
    # static argname and a host-varied coefficient compiles one
    # executable per value. 0.0 is exactly off (no term at all).
    player_support_hinge_coef: float = 0.05
    # The support-anchor family (forward KL toward a temperature-raised /
    # advantage-tilted reference; player_support_{coef,temperature,
    # adv_temperature}) was REMOVED 2026-08-27 after phases 1-4: every
    # mass-restoring variant either erased within-modality discrimination
    # (mode-covering targets + the snap ratchet) or taught the mean
    # switch's losing value. Replaced by the per-level ENTROPY terms above
    # (the PPO surrogate was split per-level in the same pass and
    # re-joined 2026-08-28 — see that revert commit) — see train_step's
    # policy bracket and the LESSONS.md ledgers for history and handles.
    # Snap period of the reference: reg_params <- target_params, in
    # place, every N steps (NashPG's K inner updates; their paper runs
    # re-clone every 10k for 25 outer rounds). Frozen between snaps —
    # the continuous EMA it replaced never reset, so the KL gap
    # compounded with policy speed (2wvnlsz3: ref_kl 2.07 nats). A
    # shorter period approaches an EMA magnet, which chases the policy
    # and degenerates into a short-horizon trust region (LESSONS 4).
    player_reg_snap_steps: int = 10_000
    ## Builder
    builder_value_loss_coef: float = 0.5
    builder_policy_loss_coef: float = 1.0
    builder_kl_loss_coef: float = 0.1
    builder_conditional_entropy_loss_coef: float = 1.0
    builder_entropy_coef: float = 0.01
    builder_entropy_prediction_normalising_constant: float = 100
    builder_entropy_advantage_scale: float = 1e-3

    # Human
    builder_human_loss_coef: float = 1e-2


def get_learner_config():
    return Porygon2LearnerConfig()
