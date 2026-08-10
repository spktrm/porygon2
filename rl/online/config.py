from typing import Literal

import chex

from rl.config.common import AdamWConfig, BaseTrainingConfig

PolicyObjectiveT = Literal["spo", "ppo"]


@chex.dataclass(frozen=True)
class Porygon2LearnerConfig(BaseTrainingConfig):
    num_steps = 5_000_000
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
    unroll_length: int = 128

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
    main_player_update_steps: int = 10
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

    # PSRO-style exploiter phase (docs/exploiter-phase-plan.md). Set only on
    # a dedicated exploiter run — a process forked from a historical main
    # checkpoint (FORK_FROM_CKPT env var) whose entire purpose is a bounded
    # best-response search against a fixed subset of the league. When set,
    # PlayerActor.get_match() skips the mirror coin toss and restricts the
    # existing PFSP draw to exactly these step_counts (piece 2) — start a
    # phase at k=1, widen the tuple later in the same phase once win-rate
    # clears exploiter_promote_winrate, as a generalization check. None on
    # every ordinary main run (the default): matchmaking is unrestricted.
    pin_opponent_steps: tuple[int, ...] | None = None
    # Promotion bar (piece 5): win-rate vs. EVERY opponent currently in
    # pin_opponent_steps must clear this, not just the average — a
    # strategy that crushes one pinned target and loses to another hasn't
    # generalized. Matches the existing "dominant" league-addition bar
    # (win-rate > 0.7) exactly — raised from an initial 0.55 (the plan
    # doc's original placeholder, deliberately looser on the theory that a
    # specialist beating a narrow target set didn't need "beats everything"
    # stringency). 0.55 turned out to be a weak statistical bar at the
    # exploiter_promote_min_games floor: standard error at n=20 games,
    # p~0.5 is ~0.11, so 0.55 is under half a standard error above a coin
    # flip — barely distinguishable from noise right at the reliability
    # floor. 0.7 (~1.8 SE above a coin flip at n=20) is a real signal.
    exploiter_promote_winrate: float = 0.7
    # Same reliability floor as exploit_ctrl_min_games_per_opponent /
    # bandit_min_games_per_opponent, applied here for the identical reason:
    # a handful of lucky games isn't a real win-rate.
    exploiter_promote_min_games: float = 20.0

    # Plasticity (shrink-and-perturb) params. Triggered when the main player
    # keeps failing to dominate its own league history: after
    # `plasticity_overdue_trigger` consecutive overdue-only league additions,
    # player params are interpolated toward a fresh init draw.
    plasticity_enabled: bool = True
    # docs/exploiter-phase-plan.md piece 7: when the overdue-stagnation
    # trigger fires, spend a bounded exploiter-phase budget first instead
    # of auto-perturbing. Purely a suppression switch on its own — pairs
    # with auto_exploiter_enabled below for full automation, or can be set
    # alone for the manual v1 workflow (watch plasticity_consecutive_overdue,
    # launch by hand via FORK_FROM_CKPT/EXPLOITER_RUN_ID, promote by hand via
    # rl/online/promote_exploiter.py). Flip back to False once a manually-run
    # episode concludes so a real stall still gets a perturbation.
    plasticity_defer_to_exploiter: bool = False
    # Full automation of the exploiter-phase lifecycle, all within one
    # `python -m rl.online.main` invocation: main pauses itself (saving a
    # checkpoint and raising a control-flow signal) the moment the overdue
    # trigger fires; the orchestration loop in main.py forks an exploiter
    # against increasingly-wide pinned opponent sets IN THE SAME PROCESS
    # (strictly sequential — never two learners live at once, since this
    # hardware can't run distributed/concurrent training anyway); each
    # exploiter self-checks its own promotion bar and self-promotes or
    # self-discards; main then resumes automatically. No manual
    # promote_exploiter.py invocation and no separate launch commands needed
    # once this is on. Implies plasticity_defer_to_exploiter — no need to
    # also set both.
    auto_exploiter_enabled: bool = True
    # k values tried in sequence per stagnation episode. Each rung is an
    # independent fresh fork from the SAME paused-main checkpoint (not a
    # continuation of the previous rung's training) — width escalates only
    # after a narrower attempt fails to clear the bar within its budget.
    # Matches piece 2's k=1 (sharpest signal) / k=3-5 (generalization check)
    # sizing, simplified for automation: the doc's "widen after success
    # within the same phase" nuance is folded into "widen after failure,
    # across independent attempts" here, for a much simpler state machine.
    # main._pick_pin_opponent_steps fills each rung with the k opponents
    # main is CURRENTLY WEAKEST against (by win-rate, reliability-gated),
    # not the k most recent snapshots — recency was only ever a proxy for
    # "resembles current main," and the league's own win/loss table says
    # directly where the actual blind spot is.
    auto_exploiter_ladder: tuple[int, ...] = (1, 3, 5)
    # Frames (not steps — consistent with add_player_max_frames /
    # plasticity_cooldown_frames) given to each ladder rung before it's
    # declared failed and the next rung is tried. Sized the same as
    # add_player_max_frames: roughly one main "overdue window" worth of
    # dedicated search per attempt.
    auto_exploiter_frame_budget: int = int(9e6)
    # How often (learner steps) a running exploiter checks its own
    # promotion bar. A fraction of save_interval_steps so a clear win
    # doesn't sit undetected for long; the check itself is cheap (a
    # handful of dict lookups over a tiny league).
    auto_exploiter_check_interval: int = 5_000
    # Per-attempt probability of shrink-and-perturbing the forked params
    # before training starts, instead of using the raw loaded checkpoint
    # verbatim. Adapted from AlphaStar's LeagueExploiter.checkpoint(),
    # which has a flat 25% chance of resetting to its original init rather
    # than continuing from its current (already-trained) weights — kept
    # here at the same 0.25. Addresses a real gap: without this, every
    # ladder rung (k=1, k=3, k=5) and every retry across separate
    # stagnation episodes always searches from the IDENTICAL starting
    # point (same weights, same optimizer momentum), so repeated failures
    # just repeat the same local search rather than genuinely
    # diversifying it. Reuses shrink_and_perturb_player_state — the same
    # mechanism plasticity resets already use — rather than a fresh
    # random init: the shrink keeps the perturbed policy anchored to
    # pre-perturbation behaviour via target_params/KL, exactly like a
    # plasticity event, instead of discarding everything learned so far.
    exploiter_hard_reset_prob: float = 0.25
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
    # eval strength regressed from its 56k peak. Judge by normalised
    # entropy holding in the ~0.5-0.65 band through 100k; if a static coef
    # can't hold it, the proper fix is a PI controller on the coef with a
    # target-entropy schedule (same pattern as the replay-KL controller).
    player_magnet_kl_coef: float = 0.05

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

    # Advantage estimation params
    player_gamma: float = 1.0
    player_alpha: float = 1.0
    # Value-target (td) lambda — pinned; the critic's objective and the
    # MC-anchor calibration signal never drift.
    player_lambda: float = 0.99
    # Advantage (gae) lambda — the actor's bias/variance knob; the
    # lambda controller (rl/online/controllers.py) drives it at runtime
    # when enabled, this is the starting/static value otherwise.
    player_adv_lambda: float = 0.99

    # Lambda PI controller: holds the measured bootstrap bias
    # (player_bootstrap_gap: main head vs lambda=1.0 MC-anchor value gap)
    # at lambda_ctrl_gap_target by adjusting the advantage lambda in
    # log(1-lambda) space. Gap under target -> lambda anneals down;
    # over -> backs off toward outcomes. During plasticity recovery
    # lambda is forced to lambda_ctrl_max (bootstrap untrustworthy) and
    # re-anneals afterwards. gap_target is on the +-1 value scale —
    # calibrate from the first run's player_bootstrap_gap telemetry.
    lambda_ctrl_enabled: bool = True
    lambda_ctrl_gap_target: float = 0.05
    # Gains sized from the 1329 trace: a sustained full-scale error moves
    # log(1-lambda) ~0.05/tick -> the 0.99 -> 0.95 traverse (~+2.0 in
    # log space) takes ~30-40k steps. The original ki=0.01 needed
    # 100-200k — most of a run spent mid-anneal.
    lambda_ctrl_kp: float = 0.2
    lambda_ctrl_ki: float = 0.05
    lambda_ctrl_interval: int = 500
    lambda_ctrl_min: float = 0.5
    lambda_ctrl_max: float = 1.0
    lambda_ctrl_sensor_ema: float = 0.01

    # Master switch for the adaptivity controller (the magnet KL coef
    # loop; name kept for checkpoint/config continuity).
    entropy_ctrl_enabled: bool = True
    # Adaptivity controller (rl/online/controllers.py) is floor-plus-decay
    # now. It used to also hold player_commit_cov's EMA at adapt_ctrl_
    # commit_target via a PI loop — removed after repeated bugs (an
    # unreachable target pinning pressure at the ceiling for ~50k steps
    # in 1338/1339, then a divide-by-near-zero once the target was
    # recalibrated toward 0.0 in 1341, then a second bug in how
    # exploit_ctrl scaled that same target). None of those bugs ever
    # touched the floors below, which is why they're what's left. See
    # AdaptivityController's class docstring for the full removal
    # history and for exactly why adapt_ctrl_decay_rate below can't
    # reproduce any of those three bug patterns.
    adapt_ctrl_floor_gain: float = 2.0
    # Event bumps, added to log(coef) directly (0.7 ~ a 2x step,
    # 1.4 ~ 4x) — the only source of a deliberate pressure increase.
    adapt_ctrl_event_bump: float = 0.7
    adapt_ctrl_perturb_bump: float = 1.4
    # Geometric per-step relaxation of the magnet-KL coefficient back
    # toward baseline, skipped on any step with an active floor breach.
    # Deliberately NOT sensor-driven (no commit_cov, no target, nothing
    # for exploit_ctrl to scale) — see AdaptivityController's docstring
    # for why that specifically is what makes this safe to add back.
    # Default 0.0 (off): matches the post-removal "pressure only ever
    # rises" behaviour exactly unless a run opts in. 3.47e-5 gives a
    # ~20,000-step half-life (1 - 0.5**(1/20_000)) — slow enough not to
    # undo real, still-needed pressure within a single overdue window
    # (~35k steps), fast enough to actually relieve a stale, unrelieved
    # stack of league-addition bumps over a run's lifetime instead of
    # leaving it monotonically climbing toward entropy_ctrl_max_scale
    # forever. Untested — no run has exercised this yet.
    adapt_ctrl_decay_rate: float = 3.47e-5  # ~20,000-step half-life
    # Hard floors — backstops, not the mechanism: the commitment
    # covariance is blind to actions the policy never takes, so a
    # modality going extinct (1330: switching to 0.002) must trip
    # something the loop cannot miss. Breaches always override decay in
    # the same tick — a real, ongoing collapse risk always wins.
    entropy_ctrl_floor: float = 0.40
    # Hard floor for the MACRO MODALITY entropy axis, which sits
    # structurally lower than total action entropy. 1328 gained strength
    # through a stretch at 0.18-0.26, so holding 0.40 there (as the
    # shared floor did in 1331) mandates more switching than the game
    # rewards; 0.20 still trips hard on a real collapse (1330 died at
    # 0.08). The learner rescales this axis into action-entropy units.
    entropy_ctrl_modality_floor: float = 0.20
    # Range the controller may drive the magnet coef over, as multiples
    # of the player_magnet_kl_coef baseline. Kept well above zero
    # regardless of how far decay can pull it down: c -> 0 against a
    # fixed magnet loses the stable fixed point.
    entropy_ctrl_max_scale: float = 10.0
    entropy_ctrl_min_scale: float = 0.2

    # BT-rating telemetry (rl/online/bandit.py): every bandit_window_steps
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
    # retired in favour of the lambda gap-controller and the
    # exploitability controller (both react every manage_league_interval
    # call; the rating itself needs hundreds of games per point, so it
    # stays an auditor, never a control signal — see bandit.py).
    bandit_window_steps: int = 20_000
    bandit_min_games_per_opponent: float = 20.0
    bandit_min_rated_opponents: int = 2

    # Exploitability controller (rl/online/controllers.py): PI on
    # 1 - (main's win-rate vs its worst historical league snapshot),
    # measured every manage_league_interval call from the same win-rate
    # table _should_add_new_player already reads (not the slower
    # BT-rating auditors above) — no bandit-style exploration tax, so it
    # reacts as fast as the underlying win-rate signal allows. Output is
    # a caution-scale multiplier (baseline 1.0) applied to the lambda and
    # replay controllers' targets — lambda_ctrl_gap_target and the replay
    # KL target both shrink as exploitability rises, pushing toward more
    # caution; it does not drive a runtime scalar of its own. Used to
    # also grow the adaptivity controller's commit_target, but that
    # target no longer exists (AdaptivityController is floor-only now —
    # see its class docstring). exploit_ctrl_target=0.3 mirrors the
    # existing "dominant" league-addition threshold (win-rate > 0.7).
    exploit_ctrl_enabled: bool = True
    exploit_ctrl_target: float = 0.3
    exploit_ctrl_kp: float = 0.2
    exploit_ctrl_ki: float = 0.05
    exploit_ctrl_interval: int = 1
    exploit_ctrl_sensor_ema: float = 0.3
    exploit_ctrl_min_scale: float = 0.5
    exploit_ctrl_max_scale: float = 2.0
    # Historical snapshots required before trusting the win-rate-min
    # reading — a lone freshly-added snapshot's win-rate is still
    # Bayesian-prior-dominated (League._win_rate_by_steps).
    exploit_ctrl_min_historical: int = 2
    # A snapshot must ALSO have this many effective games against main
    # before counting toward win_rates.min() — same reliability bar as
    # bandit_min_games_per_opponent, applied here for a different reason:
    # a freshly-added (or lightly-played) snapshot reads near 0.5 by
    # construction (main vs. a near-identical recent self), which looks
    # exactly like a real exploitability hole to this controller (1338:
    # two snapshots 5.5k/26.9k steps old, win-rate never left 0.48-0.54,
    # pinned the caution scale at its ceiling from a false positive).
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
    # 1.25, 2, 5, 10, 20, inf turns against a ~45-turn mean game (the
    # 20->inf gap is covered by the main head's lambda=0.99 ~ 100).
    # Fixed, independent of the advantage lambda, which the lambda
    # controller (or a bandit, historically) varies at runtime. Length
    # must match the model config's aux_v_head.num_heads.
    player_aux_lambdas: tuple = (0.2, 0.5, 0.8, 0.9, 0.95, 1.0)
    player_aux_value_coef: float = 0.5

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


def get_learner_config():
    return Porygon2LearnerConfig()
