from typing import Literal

import chex

from rl.config.common import AdamWConfig, BaseTrainingConfig

PolicyObjectiveT = Literal["spo", "ppo"]
QBoostVariantT = Literal["multistep", "onestep"]


@chex.dataclass(frozen=True)
class Porygon2LearnerConfig(BaseTrainingConfig):
    num_steps = 5_000_000
    # One pool size for every population (docs/three-population-league.md,
    # block-sequential scheduling): all three populations spawn pools of
    # this size, but the per-population run_gate (Learner._set_active)
    # means only the block owner's actors actually play — so this is
    # simply "the machine's actor budget", handed whole to whoever is
    # training. Idle pools' threads stay alive but wait at the gate
    # between games (no create/destroy churn, no inference contention).
    num_player_actors: int = 12
    num_builder_actors: int = 4
    # One eval thread per entry; each plays the service baseline at that
    # index (service/src/server/eval.ts: 0=random, 1=default,
    # 2=simple_heuristic). The index travels explicitly in the env username
    # suffix and the baseline name in the metric key, so eval coverage no
    # longer depends implicitly on actor count. All threads point at the
    # strongest baseline: Random/Default were saturated (93%/72% at 163k
    # steps) while winrate-vs-simple-heuristic — the series runs are judged
    # by — was starved at ~1 game per 80 learner steps.
    eval_baselines: tuple[int, ...] = (2, 2, 2)
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

    # Batch iteration params
    batch_size: int = 4

    # Replay buffer params
    # Kept small on purpose: steady-state throughput is set entirely by
    # replay_ratio (samples per trajectory), so capacity only controls how
    # stale a trajectory is when sampled. 256 keeps mean sample age well
    # inside one EMA-target time constant (1/player_ema_update_rate steps).
    player_replay_buffer_capacity: int = 256
    player_replay_ratio: int = 8
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
    cloud_save_interval_steps: int = 100_000
    league_winrate_log_steps: int = 1_000
    # How often (own steps) each population publishes fresh live params for
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
    # Backstop ("overdue") add interval, ~35k learner steps at the current
    # batch shape. The healthy path — "dominant" adds when main beats every
    # member >0.7 — is ungated above min_frames, so this clock only paces
    # snapshots while the agent is NOT visibly improving. At 3e6 (~11.5k
    # steps) it filled the league with ~0.5-winrate near-copies of main
    # (mirror play with extra staleness) and, because overdue adds are the
    # plasticity trigger's input, made the stagnation clock hair-trigger.
    add_player_max_frames: int = int(9e6)
    # Learner steps before the first historical snapshot joins the league.
    # Kept low enough that a short (~200k step) run still trains against a
    # populated league rather than pure mirror self-play — mirror-only runs
    # measured 93% vs Random but ~10% vs SimpleHeuristic at 163k steps,
    # the signature of self-exploiting policies that don't transfer to
    # stylistically alien opponents.
    minimum_historical_player_steps: int = int(5e4)
    league_size: int = 16
    manage_league_interval: int = 10
    # Disk-backed league: max materialised opponents held in RAM at once, and
    # the UCB exploration coefficient governing which stay hot.
    league_cache_size: int = 16
    league_ucb_c: float = 1.0

    # Three-population league (docs/three-population-league.md, updated
    # 2026-08-13 to block-sequential scheduling): MainPlayer,
    # MainExploiter, and LeagueExploiter are live populations sharing one
    # process/GPU — never torn down/refounded. Scheduling is
    # block-sequential, not fraction-interleaved: main trains until a
    # routine league addition closes its window, then ONE exploiter
    # population (alternating MainExploiter/LeagueExploiter) owns the GPU
    # for a full attempt — to its own promotion or frame-budget timeout,
    # AlphaStar's ready_to_checkpoint shape — before main's next window.
    # There are deliberately no frame-split fractions: each population
    # trains to its own terminal condition against a (mostly) frozen
    # target, sequentialising AlphaStar's concurrent league on one GPU
    # rather than simulating concurrency by interleaving (main does still
    # train as filler on ticks where the active exploiter's smaller actor
    # pool has no batch ready — see Learner._select_population).
    # MainExploiter targets main's own lineage (and live main directly,
    # once reliable — AlphaStar's actual rule); LeagueExploiter targets
    # the whole historical population via linear_capped PFSP,
    # unrestricted. False means only main exists, exactly as before.
    auto_exploiter_enabled: bool = True

    # Centralized batched actor inference (rl/online/inference.py): all
    # training PlayerActors submit per-step forwards to one server thread
    # that runs a single vmapped apply over whatever is queued — zero-wait
    # adaptive batching (no min-batch, no max-wait timer; see the module
    # docstring for why those knobs are deliberately absent). False
    # restores the per-actor batch-1 Agent.step_player path exactly.
    inference_server_enabled: bool = True
    # Compile-shape/latency cap on one batched forward. Padded to powers
    # of two, so traced batch sizes are log2(cap)+1 distinct values.
    inference_max_batch: int = 16
    # Device-resident params LRU in the server, keyed by params version —
    # each entry is one device copy of the actor player params (~81MB at
    # 20.3M f32), so this is a VRAM knob. It REPLACES the old per-actor
    # copies (every actor device_put its own copy per game — up to ~2x num
    # actors live at once), and it must cover the WORKING SET of versions
    # simultaneously referenced by in-flight games: ~2 per live population
    # (paced by main_player_update_steps above) + ~5-7 concurrent
    # historical PFSP opponents. 12 covers that with a slot or two spare;
    # sizing BELOW the working set causes LRU thrash — a serial ~81MB
    # host->device transfer per miss in the server thread, plus alloc/free
    # churn in XLA's pool (fragmentation, the OOM class that killed
    # session 1786537634).
    inference_params_cache_size: int = 12

    # Promotion bar, shared by both exploiter types: win-rate vs. EVERY
    # opponent currently pinned must clear this, not just the average — a
    # strategy that crushes one pinned target and loses to another hasn't
    # generalized. Matches the existing "dominant" league-addition bar
    # (win-rate > 0.7) exactly — raised from an initial 0.55 (deliberately
    # looser on the theory that a specialist beating a narrow target set
    # didn't need "beats everything" stringency). 0.55 turned out to be a
    # weak statistical bar at the exploiter_promote_min_games floor:
    # standard error at n=20 games, p~0.5 is ~0.11, so 0.55 is under half
    # a standard error above a coin flip — barely distinguishable from
    # noise right at the reliability floor. 0.7 (~1.8 SE above a coin flip
    # at n=20) is a real signal.
    exploiter_promote_winrate: float = 0.7
    # Same reliability floor as exploit_ctrl_min_games_per_opponent /
    # bandit_min_games_per_opponent, applied here for the identical reason:
    # a handful of lucky games isn't a real win-rate.
    exploiter_promote_min_games: float = 20.0
    # Minimum dwell before a population's promotion bar is even consulted.
    # AlphaStar's MainExploiter.ready_to_checkpoint has an explicit
    # minimum-dwell floor (min 2e9 steps) before either a promotion or a
    # timeout can fire; the pre-redesign code had none at all (a promotion
    # could fire on the very first check after creation). Shared by both
    # exploiter types.
    exploiter_min_dwell_frames: int = int(1e6)

    # Per-population frame budget before a non-promoted attempt times out
    # and resets (main_exploiter_reset_to_main / exploiter_hard_reset_prob
    # below). Sized the same as add_player_max_frames: roughly one main
    # "overdue window" worth of dedicated search.
    main_exploiter_frame_budget: int = int(9e6)
    league_exploiter_frame_budget: int = int(9e6)

    # No pinned-opponent-set width knob: neither exploiter population pins
    # to a fixed target set at all — AlphaStar's own exploiters PFSP-sample
    # fresh from their candidate pool every single match (whole population
    # for LeagueExploiter, origin=="main" lineage for MainExploiter's
    # fallback branch below), never freezing a subset for the population's
    # whole lifetime. A fixed k (the pre-redesign single-exploiter role's
    # mechanism) would just be re-inventing that narrower, less faithful
    # design under a new name — see rl/online/player_actor.py's
    # get_match().

    # MainExploiter's live-target branch (AlphaStar's actual MainExploiter
    # rule, not a simplification of it): if main_exploiter's own win-rate
    # against LIVE main exceeds this floor, play live main directly instead
    # of falling back to the lineage-restricted historical draw above.
    # Reliability-gated by main_exploiter_live_target_min_games — same
    # pattern as exploit_ctrl_min_games_per_opponent/bandit_min_games_per_
    # opponent: until enough games are recorded, the signal is
    # untrustworthy, so the fallback (never the optimistic) branch is
    # taken. main_exploiter's every game (either branch) records into
    # League's shared payoff table, so this signal keeps updating with no
    # new statistics code.
    main_exploiter_live_target_winrate_floor: float = 0.1
    main_exploiter_live_target_min_games: float = 20.0

    # How often (learner steps) a running exploiter checks its own
    # promotion bar. A fraction of save_interval_steps so a clear win
    # doesn't sit undetected for long; the check itself is cheap (a
    # handful of dict lookups over a tiny league).
    auto_exploiter_check_interval: int = 5_000

    # On a terminal outcome (promoted or timed out): main_exploiter ALWAYS
    # resets to a fresh fork of main's then-current live params — this is
    # AlphaStar's actual rule for this role, not a tunable roll, hence a
    # bool rather than a probability.
    main_exploiter_reset_to_main: bool = True
    # league_exploiter instead rolls this on a terminal outcome: per-attempt
    # probability of shrink-and-perturbing toward random init instead of
    # continuing to train its current (already-trained) weights un-reset.
    # Adapted from AlphaStar's LeagueExploiter.checkpoint(), which has a
    # flat 25% chance of resetting to its ORIGINAL init — this project
    # deliberately reuses shrink_and_perturb_player_state (the same
    # mechanism plasticity resets use) rather than a literal fresh random
    # init: the shrink keeps the perturbed policy anchored to
    # pre-perturbation behaviour via target_params/KL, instead of
    # discarding everything learned so far the way a literal reinit of an
    # already-trained network would.
    exploiter_hard_reset_prob: float = 0.25

    # Plasticity (shrink-and-perturb) params. Triggered when the main player
    # keeps failing to dominate its own league history: after
    # `plasticity_overdue_trigger` consecutive overdue-only league additions,
    # player params are interpolated toward a fresh init draw. Unrelated to
    # the three-population redesign above: this is main's own weight
    # perturbation mechanism, orthogonal to whether exploiter populations
    # exist.
    plasticity_enabled: bool = True
    # Historically paired with the old phase-based exploiter mechanism
    # (suppress main's own perturbation while a main-pausing exploiter
    # phase ran); kept for the manual, non-automated workflow
    # (rl/online/promote_exploiter.py) where that still applies. Under the
    # three-population redesign, exploiter populations never pause main, so
    # this has no effect when auto_exploiter_enabled is set.
    plasticity_defer_to_exploiter: bool = False

    # RAM-attribution diagnostics (Learner._log_memory_diagnostics), logged
    # through main's periodic wandb logs: process RSS + OS-vs-python thread
    # census + exact replay-buffer/league-cache byte counts. Added after
    # session 1786537634's RSS climbed 5.9->17GB (threads 478->775) with
    # no way to attribute it from wandb alone. 0 disables. Cost per tick
    # is one /proc read + a walk over stored trajectories' array headers —
    # negligible at this interval.
    memory_diag_interval: int = 5_000

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

    # Consecutive overdue-only adds before a perturbation. At 1 (the old
    # value) a single stalled add window (~6k steps of not dominating the
    # league) fired a 50% reset: the Aug 2026 run perturbed during a
    # consolidation phase and dropped below its own 50k-step snapshot
    # (winrate 0.485), paying a multi-10k-step recovery tax. Require a
    # sustained stall instead.
    plasticity_overdue_trigger: int = 3
    # Fraction of the old weights kept (lambda). Higher = milder perturbation.
    plasticity_default_shrink: float = 0.5
    # Per top-level module overrides; the encoder holds expensive
    # representations, so it is perturbed more gently than the heads.
    plasticity_module_shrink: tuple[tuple[str, float], ...] = (("encoder", 0.5),)
    # Recovery gate: no further perturbations until the main player beats the
    # pre-perturbation snapshot at this winrate and the cooldown has elapsed.
    plasticity_recovery_winrate: float = 0.6
    plasticity_cooldown_frames: int = int(1e6)
    # Plasticity instrumentation: every N learner steps, run an
    # encoder-only forward on the current batch and log trunk
    # representation health — dormant-unit fraction (ReDo criterion) and
    # srank@0.99 — alongside the per-step fresh-vs-replayed value-error
    # gap. Together these say whether a plateau is actually plasticity
    # loss (dormant/rank degrading, memorisation gap opening) before a
    # league-stagnation trigger fires a perturbation. 0 disables the
    # probe (the value-error gap is always on).
    plasticity_probe_interval: int = 1000

    # Player magnet regularization (MMD-style). The policy is pulled toward a
    # fixed hierarchical magnet over legal actions (uniform over valid
    # modalities, uniform within each modality — the composed head's init
    # policy), so the KL is per-state entropy regularization that is
    # invariant to per-modality option counts. The
    # magnet is deliberately stationary — a fixed anchor is what gives the
    # regularized self-play dynamics a stable fixed point (QRE), whereas an
    # EMA magnet chases the policy and degenerates into a short-horizon trust
    # region. The EMA target is reserved for the v-trace/IMPACT reference and
    # plays no regularization role. The coef sets the softness level.
    # 0.05 (up from 0.01): the per-modality policy head removed the shared
    # grid's accidental sharpness cap, and at 0.01 the magnet lost the
    # arm-wrestle — chocolate-silence-1307 collapsed to normalised entropy
    # 0.27 (modality 0.17) by 190k while magnet KL climbed to 1.44, and
    # eval strength regressed from its 56k peak. 0.1 (up from 0.05,
    # 2026-08-17): the entropy-regularisation timeline showed the longest
    # stable lineages (Oct-Nov 2025, 400-580k steps, 2.0-2.9 nats whole
    # lifetime) ran ~2.8-5.4x today's effective entropy pressure (mostly a
    # 4x per-head structural factor), while the current 0.7-0.9 nat
    # operating point is the lowest outside the collapse regimes and the
    # switch-collapse pattern reads as under-regularisation. Judge by
    # normalised entropy holding in the ~0.5-0.65 band through 100k
    # (magnet KL level flat, not climbing); revert to 0.05 on overshoot.
    # If a static coef can't hold the band, the proper fix is a PI
    # controller on the coef with a target-entropy schedule (same pattern
    # as the replay-KL controller).
    player_magnet_kl_coef: float = 0.1

    # Learning params. Momentum (b1=0.9) is on: stability under replay reuse
    # is already provided by the SPO trust region, the behaviour-KL penalty
    # and the replay-KL controller (which throttles reuse if actor-KL
    # exceeds its 0.045 ceiling) — momentum-free Adam was leaving all three
    # guardrails idle (actor-KL 0.013–0.044, grad norm 1–4 vs clip 10).
    adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-08, weight_decay=0)
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
    gradient_accumulation_steps: int = 1
    # Fast EMA target (IMPACT-style): supplies the clipped-target ratio in
    # the surrogate, the v-trace reference policy, and the value bootstraps,
    # so it must track the learner closely for stability under replay reuse.
    # (R-NaD likewise keeps a 1e-3 target purely for v-trace stability,
    # separate from its slow anchors.)
    player_ema_update_rate: float = 1e-3
    # Advantage-normaliser statistics EMA — deliberately much faster than
    # the target-params EMA above (the stats are per-batch scalars, not
    # parameters): ~100-step time constant so PG mis-scaling after a
    # distribution shift (plasticity event, league addition) lasts ~0.5%
    # of a rating window instead of ~10%.
    player_adv_ema_rate: float = 1e-2
    # Floor for the normaliser's std divisor: below this, stop rescaling
    # so converged-policy noise is not amplified into the actor. Sized
    # ~10-15% of the healthy running adv-std (0.34-0.6 observed).
    player_adv_std_floor: float = 0.05
    builder_ema_update_rate: float = 1e-3

    # Advantage estimation params — AlphaStar's v-trace + UPGO recipe
    # (2026-08-14, replacing the LambdaGapController; see targets.py):
    # the value head trains on TD(lambda)-style v-trace targets at
    # player_lambda, the policy gradient takes v-trace advantages with NO
    # lambda of its own (AlphaStar's vtrace_advantages is
    # unparameterised — clipped IS weights only, i.e. lambda=1), and a
    # separate UPGO term (below) carries the per-step outcome-conditional
    # credit the old runtime-tuned advantage lambda was trying to
    # approximate globally.
    player_gamma: float = 1.0
    player_alpha: float = 1.0
    # Value-target lambda. AlphaStar's own choice: TD(lambda=0.8), a
    # short (~5-step) bootstrap horizon — they could afford heavy
    # bootstrapping because supervised init gave them a sane critic from
    # step one. This project starts from scratch AND the 1328 five-arm
    # sweep pointed the same direction (monotone lower-lambda-better,
    # confounded but directional), so 0.8 is adopted as-is. The aux
    # spectrum's lambda=1.0 MC-anchor row (player_aux_lambdas) keeps the
    # bootstrap-bias readout (player_bootstrap_gap) alive regardless.
    player_lambda: float = 0.8

    # UPGO (AlphaStar rl.py upgo_returns): policy-gradient-only return
    # that follows the actual trajectory while the continuation performs
    # at least as well as the critic expected (lambda_t = 1 where
    # Q_hat >= V, else cut to the critic's value) — full Monte Carlo
    # credit along successful lines, truncation at the first
    # worse-than-expected step. Coefficient mirrors AlphaStar's equal
    # weighting of the v-trace and UPGO PG terms. Passed to train_step
    # as a RUNTIME scalar, zeroed while plasticity recovery is active
    # (an optimistically-wrong post-perturbation critic would cut in the
    # wrong places — the same regime the old lambda controller handled
    # by forcing lambda to its ceiling).
    player_upgo_coef: float = 1.0

    # No adaptivity/entropy controller fields anymore: the magnet KL
    # coefficient is exactly player_magnet_kl_coef, always. The
    # AdaptivityController was removed entirely 2026-08-13 (hard to tune,
    # harder to predict — see rl/online/controllers.py's module docstring
    # for the bug history). Its entropy sensors are still logged from
    # train_step (player_action_normalized_entropy,
    # player_normalized_modality_entropy); modality collapse (1330 died
    # at 0.08 on that axis) is now watched on the dashboard, not
    # auto-corrected.

    # BT-rating telemetry (rl/online/ratings.py): every bandit_window_steps
    # the learner fits a Bradley-Terry rating for main against the frozen
    # league snapshots from the payoff table the league already keeps,
    # and logs it plus the exploitability auditors (worst-matchup drift,
    # BT non-transitivity residual). Pure self-play signal: mirror games
    # carry no reward and the scripted eval baselines are never
    # consulted. Before bandit_min_rated_opponents snapshots have
    # bandit_min_games_per_opponent effective games against main, no
    # rating exists and rating_logs reports bandit_rating_valid=0.
    # These fields used to also configure LambdaBandit, a discounted-UCB
    # bandit that retuned the advantage lambda from this same rating —
    # retired in favour of the lambda gap-controller (itself since
    # removed, 2026-08-14, for UPGO + fixed player_lambda) and the
    # exploitability controller (the rating itself needs hundreds of
    # games per point, so it stays an auditor, never a control signal —
    # see ratings.py). The bandit_ field and metric prefixes are kept for
    # wandb continuity across lineages.
    bandit_window_steps: int = 20_000
    bandit_min_games_per_opponent: float = 20.0
    bandit_min_rated_opponents: int = 2

    # No ExploitabilityController anymore (removed 2026-08-14, the last
    # adaptive hyperparameter loop — see rl/online/controllers.py's
    # module docstring). The replay KL target is fixed at
    # player_replay_kl_target; the worst-matchup win-rate it sensed still
    # exists as _should_add_new_player's "dominant" gate and the
    # league_main_winrate_min auditor, it just doesn't actuate anything.
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

    builder_gamma: float = 1.0
    builder_alpha: float = 1.0
    builder_lambda: float = 0.99

    # Player policy objective: ratio-based surrogates with a trust region.
    player_policy_objective: PolicyObjectiveT = "spo"

    player_ppo_clip_threshold: float = 0.3
    builder_ppo_clip_threshold: float = 0.3

    # Advantage EMA normalization. When disabled, raw advantages are used;
    # the EMA statistics keep updating either way so re-enabling is smooth.
    player_advantage_ema_enabled: bool = True

    # Loss coefficients
    ## Player
    player_policy_loss_coef: float = 1.0
    player_kl_loss_coef: float = 0.05
    player_value_head_loss_coef: float = 1.0
    # Multi-lambda auxiliary value heads: K extra categorical value
    # readouts trained by CE against per-lambda v-trace targets at the
    # main gamma — a target bias/variance spectrum that shapes the shared
    # representation (the main head keeps sole ownership of the policy's
    # advantages). See player_aux_lambdas. Coefficient kept modest — the
    # grad-norm lesson from the integrated-critic era: heavy aux gradient
    # globally clips everything.
    # Multi-lambda aux value heads: all at the main gamma (=1, win prob),
    # differing in target construction — lambda=1 is the Monte Carlo
    # anchor, low lambda leans on the critic. A gamma spectrum would
    # degenerate here (terminal-only reward: gamma^45 kills the signal).
    # Spectrum chosen ~geometric in effective horizon 1/(1-lambda):
    # 2, 10, 20, inf turns against a ~45-turn mean game, bracketing the
    # main head's own lambda=0.8 ~ 5-turn horizon. Two rows dropped as
    # redundant-with-another-head, same logic both times:
    # - lambda=0.2 (2026-08-14): near-pure next-step self-distillation
    #   with terminal-only reward; its R2 series correlated 0.984 with
    #   lambda=0.5's over 223k steps (run 1786583261-main).
    # - lambda=0.8 (2026-08-14, after player_lambda moved 0.99->0.8):
    #   the main head's target became exactly lambda=0.8 v-trace at the
    #   same gamma — a copy, not a horizon.
    # The lambda=1.0 row is the MC anchor player_bootstrap_gap reads —
    # the safety instrument for the bootstrap-heavy lambda=0.8 value
    # target — and must stay. Length must match the model config's
    # aux_v_head.num_heads.
    player_aux_lambdas: tuple = (0.5, 0.9, 0.95, 1.0)
    player_aux_value_coef: float = 0.5

    # Counterfactual value ladder (2026-08-16): shared coefficient for the
    # own-info (no opponent sheet) and public-info (history-only) value
    # heads' CE losses. Critic-only representation/diagnostic heads like
    # the aux spectrum — the policy reads the main (privileged) head's
    # advantages exclusively.
    player_value_ladder_coef: float = 0.25

    # Observer all-action Q critic (stage 1, docs/q-critic-plan.md):
    # Retrace(lambda)-trained categorical Q over the flat action grid,
    # read off the same action embeddings as the policy heads. ZERO policy
    # influence at this stage — it exists for diagnostics (player_q_r2,
    # player_q_ev_gap, player_q_switch_move_gap: the direct "what is
    # switching worth" readout) and representation shaping. Enabling adds
    # q_head params to the tree, so a strict checkpoint-mode resume across
    # the flip fails — resume with LOAD_STATE_MODE=params (merge) or start
    # fresh. Singles only (asserted in get_player_model_config).
    player_q_enabled: bool = True
    # CE weight — same modest scale as the aux value spectrum, and for the
    # same reason (heavy aux gradient globally clips everything).
    player_q_coef: float = 0.5
    # Retrace trace parameter; matches player_lambda's 0.8 default.
    player_q_lambda: float = 0.8
    # Stage 2 (docs/q-critic-plan.md §5): forward KL from the Boltzmann
    # distribution over the EMA target's deployable-rung action values
    # to the learner policy — the anti-ratchet policy-improvement channel
    # (its pull toward an action p_q rates well does not scale with pi's
    # current mass there, unlike the reverse-KL magnet, whose restoring
    # force vanishes exactly as a modality starves). Enabled 2026-08-18 as
    # a deliberate early unlock of backlog item 8: the voluntary-switch
    # crossover re-formed on the 1786951032 lineage (realised post-switch
    # returns positive while the critic gap stayed negative), and at
    # tau=0.1 the Boltzmann target assigns the collapsed switch modality
    # ~10x the policy's mass even under today's switch-averse critic, so
    # the term's first-order push is pro-switch data generation. MAIN
    # POPULATION ONLY (host-gated in Learner._train_step): the exploiter
    # blocks stay clean as a within-run contrast.
    player_q_improve_enabled: bool = True
    # COEF ZEROED 2026-08-19 (fresh no-checkpoint lineage): the stage-2 KL
    # is superseded by the stage-3 Q-boosting advantage below — enabled=True
    # with coef 0 is the plan's own observer posture, keeping the p_q
    # diagnostics (player_q_improve_pq_* vs pi switch mass) live as the
    # within-run "what would the KL have pushed" readout with zero loss
    # influence. RUNTIME scalar into train_step, so flipping it back
    # never recompiles.
    player_q_improve_coef: float = 0.0
    player_q_improve_ramp_steps: int = 2000
    # Boltzmann temperature over the +/-1 categorical value support.
    # 0.1 = sharp but not argmax; tuned against p_q entropy / switch-mass
    # diagnostics on the live checkpoint before launch.
    player_q_improve_tau: float = 0.1
    # Stage 3 Q-boosting advantage (docs/q-boosting-plan.md; Fan & Farina,
    # arXiv 2605.19235): cross-fade the PG advantage from the v-trace
    # channel to retrace_g − v_exp — unbiased at lambda=1 for ANY critic
    # accuracy (their Thm 3.1) and lower-MSE than the GAE family exactly
    # where the policy must keep randomising (Var_a[Q] > 0), i.e. the
    # mixed-strategy stay/switch states the collapse forms in. Blended
    # PRE-normalisation so the existing ema_adv_mean/std fields serve
    # verbatim (zero new pytree leaves). LIVE from launch on the
    # 2026-08-19 fresh lineage, in place of the stage-2 KL (coef zeroed
    # above): boosting repairs the credit signal itself where the KL
    # propped the policy up with a Boltzmann prior. MAIN POPULATION ONLY
    # (host-gated), exploiters stay clean as the within-run contrast.
    # Abort = zero this flag; q_boost_mix forces to 0 next step, no
    # recompile. Judge by switch_ratio, player_q_boost_adv_* agreement,
    # player_q_calibration_r2_fresh vs player_value_r2_fresh, and the
    # entropy-cliff watch inherited from stage 2.
    player_q_boost_enabled: bool = True
    # Loss-free Stage-3a diagnostics (player_q_boost_adv_*,
    # player_q_action_var, calibration r2 fresh/replay); on permanently.
    player_q_boost_diagnostic_enabled: bool = True
    # multistep = retrace_g − v_exp (the paper's Thm 3.1 estimator);
    # onestep = q_taken − v_exp fallback arm. Config-static: switching
    # variants recompiles.
    player_q_boost_variant: QBoostVariantT = "multistep"
    # Linear cross-fade 0→1 over this many main-pop steps from first
    # activation, mirroring player_q_improve_ramp_steps' host pattern.
    player_q_boost_ramp_steps: int = 2000
    # Agent57/Ape-X-style exploration ladder (replaces stage 4's
    # cross-population intake, removed 2026-08-15: it conflated another
    # agent's policy evidence with main's own action values, and its
    # frozen-between-blocks stock went stale — foreign-row Q R² 0.27 vs
    # 0.84 own). Every player actor independently makes each game it
    # plays with its own live params an explore game with this
    # probability, drawing a fresh temperature log-uniform from
    # explore_temp_range (the continuous analogue of R2D2's
    # geometrically-spaced epsilon ladder, assigned per game like
    # Agent57's per-episode picks rather than per dedicated actor slot;
    # base games sample at 1.0). Per-game draws make the explore share
    # of trajectories equal this probability BY CONSTRUCTION — the prior
    # dedicated-slot design's 2/12 actors bypassed the InferenceServer
    # full-time (it has no per-request head_params, so tempered games
    # take the direct batch-1 path) and out-produced the server-queued
    # base pairs ~4x, inflating the intended ~17% row share to ~44% and
    # halving the PG/value effective batch. Sides draw INDEPENDENTLY:
    # tempered play is graded against the true temp-1 policy it will
    # actually face, and the untempered side of a mixed game pushes
    # ordinary PG/value rows played under opponent-switch pressure —
    # coverage the old explorer-vs-explorer pairing kept locked inside
    # Q-only rows. Frozen-opponent sides (nothing trainable, and league
    # payoff reads would be polluted) and eval actors never temper;
    # tempered PFSP games are also skipped from payoff updates. Because
    # the temp is applied to the logits BEFORE the policy metrics are
    # computed, the recorded behaviour policy IS the tempered
    # distribution, so v-trace/Retrace ISRs are automatically correct.
    # Explore trajectories are tagged at the actor: own-masked out of
    # every PG/value/builder loss at the existing choke points, consumed
    # ONLY by the observer Q critic. 0 disables. No parameter change, so
    # the flip is checkpoint-safe.
    # Range log-symmetric around 1 (median temp = 1.0). Not wider: 0.5
    # already matches eval's sharpened temp, and both ends stay within
    # 2x of the base policy so Retrace's ISR truncation keeps most of
    # every trace.
    explore_game_prob: float = 1.0 / 6.0
    explore_temp_range: tuple[float, float] = (0.5, 2.0)
    # Exploiter blocks run for hours while main's step (the checkpoint
    # pacing basis) barely moves; the active exploiter's own periodic save
    # paces on ITS step counter at this tighter interval so a mid-block
    # kill (the 2026-08-15 09:38 machine shutdown lost a block segment)
    # costs minutes, not the whole block since its boundary save.
    exploiter_save_interval_steps: int = 5000

    ## Builder
    builder_value_loss_coef: float = 0.5
    builder_policy_loss_coef: float = 1.0
    builder_kl_loss_coef: float = 0.1
    builder_entropy_loss_coef: float = 0.01
    builder_conditional_entropy_loss_coef: float = 1.0
    builder_entropy_coef: float = 0.01
    builder_entropy_prediction_normalising_constant: float = 100
    builder_entropy_advantage_scale: float = 1e-3

    # Human
    builder_human_loss_coef: float = 1e-2


@chex.dataclass(frozen=True)
class RuntimeScalars:
    """The TRACED complement of Porygon2LearnerConfig: per-step scalars the
    host computes each call (ramps, plasticity gating, per-population
    zeroing) and passes into the jitted train_step as ONE pytree argument.
    These must never move into the static config above — config is a jit
    static_argname, and static scalars retained ~5GB of executables per
    distinct value and OOM-killed run 1326; as traced leaves they change
    freely with zero recompiles. None falls back at the use site (to the
    static coef for upgo/magnet, to 0 for the ramped Q terms); the live
    Learner path always fills all four."""

    # config.player_upgo_coef, zeroed by the host during plasticity
    # recovery (a freshly-perturbed critic cuts UPGO returns in the wrong
    # places).
    upgo_coef: chex.Array | None = None
    # config.player_magnet_kl_coef (the entropy watchdog escalates it
    # host-side).
    magnet_coef: chex.Array | None = None
    # Host-side ramp of player_q_improve_coef; zero for the exploiter
    # populations.
    q_improve_coef: chex.Array | None = None
    # Stage-3 Q-boosting cross-fade weight (docs/q-boosting-plan.md);
    # zero for exploiters and while player_q_boost_enabled is off.
    q_boost_mix: chex.Array | None = None


def get_learner_config():
    return Porygon2LearnerConfig()
