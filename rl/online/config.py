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
    # Backstop ("overdue") add interval, ~35k learner steps at the current
    # batch shape. The healthy path — "dominant" adds when main beats every
    # member >0.7 — is ungated above min_frames, so this clock only paces
    # snapshots while the agent is NOT visibly improving. At 3e6 (~11.5k
    # steps) it filled the league with ~0.5-winrate near-copies of main
    # (mirror play with extra staleness) and made the stagnation clock
    # hair-trigger.
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
    player_adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-08, weight_decay=0)
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
    # instrument is gone with them (CLAUDE.md ledger).
    player_lambda: float = 0.8

    # No adaptivity/entropy controller fields anymore. The
    # AdaptivityController was removed entirely 2026-08-13 (hard to tune,
    # harder to predict — see CLAUDE.md 10
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
    player_kl_loss_coef: float = 0.05
    player_value_head_loss_coef: float = 1.0

    # All-action Q critic (docs/q-critic-plan.md) — STRUCTURAL since
    # 2026-08-20 (no enable flag): the hierarchical advantage head is
    # part of the model, its loss always trains, and every consumer
    # assumes it exists. Singles only (asserted in
    # get_player_model_config).
    # Huber weight — deliberately modest, per the grad-norm
    # lesson from the integrated-critic era: a heavy auxiliary gradient globally clips
    # everything (CLAUDE.md 5).
    player_q_coef: float = 0.5
    # No trace parameter since 2026-08-23 (Step 3): the residual critic
    # regresses on the TD(0) label r + gamma*V_win_target(s'). Retrace at
    # q_lambda 1.0 / pi lambda 0.8 (the outcome chain within the chunk)
    # is in git history; the Step-6 probe showed the categorical head
    # fitting those labels through a state-only route, which the residual
    # form closes. Since 2026-08-26 the policy no longer reads this stack
    # — it is the matched-control observer (same modules as the policy
    # head under a different loss) and the Retrace value baseline.
    #
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
    # Which surrogate policy_gradient_loss runs for the player: "ppo"
    # (NashPG's own rule) or "spo" (the builder's smooth quadratic) for
    # an A/B. Static config: switching costs one recompile at launch.
    player_pg_objective: str = "ppo"
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
    # Entropy bonus, differentiated — FACTORISED 2026-08-27 (the Oct–Nov
    # 2025 form rebuilt on the composed head): the coefficient now scales
    # H(macro) + H(micro | taken modality), each masked-AVERAGED over its
    # own row set (macro: >= 2 live modalities; micro: taken modality has
    # >= 2 legal cells). The joint H it replaces decomposes as H(macro) +
    # sum_m pi_m * H(micro|m) — its within-switch pressure died in
    # proportion to switch mass, defunding the which-axis exactly as the
    # modality shrank, and offered one budget payable wherever entropy was
    # cheapest (the measured blindness: global 0.755 while modality 0.22).
    # Unit weights are per-axis budgets; the masked average makes a rare
    # taken-modality's term inverse-frequency amplified; and as a BONUS
    # (not a target) it is a temperature — per axis the equilibrium is
    # pi ∝ exp(A/coef), so live evidence beats it wherever it exists
    # (the injection post-mortem's temperature-vs-target law). 0.05 kept
    # across the transition; the sum's scale differs from the joint H, so
    # read player_entropy_{macro,micro_taken} at first batches before any
    # retune, and retune on the equilibrium condition if at all.
    player_ent_coef: float = 0.05
    # The support-anchor family (forward KL toward a temperature-raised /
    # advantage-tilted reference; player_support_{coef,temperature,
    # adv_temperature}) was REMOVED 2026-08-27 after phases 1-4: every
    # mass-restoring variant either erased within-modality discrimination
    # (mode-covering targets + the snap ratchet) or taught the mean
    # switch's losing value. Replaced by the FACTORISED objective: the
    # PPO surrogate and the entropy bonus each split into macro (whether)
    # and micro (which, within the taken modality) terms with their own
    # row masks — see train_step's policy bracket and the CLAUDE.md
    # removal ledger for the full history and revert handles.
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
