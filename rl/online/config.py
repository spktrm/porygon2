from typing import Literal

import chex

from rl.config.common import AdamWConfig, BaseTrainingConfig

PolicyObjectiveT = Literal["spo"]


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

    # Capacity instrumentation: every N learner steps, run an encoder-only
    # forward on the current batch and log trunk representation health —
    # dormant-unit fraction (ReDo criterion) and srank@0.99 — alongside the
    # per-step fresh-vs-replayed value-error gap. Pure observer since the
    # plasticity controller was removed (2026-08-21); kept because it is
    # what caught the 1e-4 LR collapse (srank 0.27 by 13k steps while
    # actor-KL sat at 0.002 — KL headroom is not LR headroom, LESSONS.md
    # 5). 0 disables the probe (the value-error gap is always on).
    capacity_probe_interval: int = 1000

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
    # 0.2 (up from 0.1, 2026-08-19, user's call for the COMA relaunch):
    # the q-boost lineage collapsed at ~3x the baseline speed (mod-entropy
    # 0.87→0.27 by 18k, switch_ratio 0.45→0.02) with 0.1 holding nothing.
    # Known limit (docs/entropy-gradient-pressure.md §3): reverse-KL force
    # is π-weighted and cannot hold a floor once a modality is starved —
    # 0.2 buys pressure in the healthy-mass formative window only; the
    # per-state restoring channel is the COMA loss below.
    # 0.05 (2026-08-21, user's call with the epsilon explore ladder):
    # this term is entropy regularisation (uniform prior => KL = log|A|
    # - H) and its climb 0.05 -> 0.1 -> 0.2 was entirely compensation
    # for a SUPPLY problem -- rare-action coverage shrinking with the
    # collapse -- that the behaviour-side epsilon floor now owns without
    # touching pi. Back to regularisation-only: convergence of the
    # zero-sum dynamics and keeping genuinely mixed spots mixed, not
    # holding switch mass up against the critic. Watchdog escalation
    # retired with it.
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

    # Builder policy objective: ratio-based surrogate with a trust region.
    # The player has no ratio-surrogate term anymore — the single-action PG
    # and UPGO losses were removed 2026-08-21 (LESSONS.md 3).
    builder_policy_objective: PolicyObjectiveT = "spo"
    builder_ppo_clip_threshold: float = 0.3

    # Loss coefficients
    ## Player
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

    # All-action Q critic (docs/q-critic-plan.md) — STRUCTURAL since
    # 2026-08-20 (no enable flag): the two-rung hierarchical Q head is
    # part of the model, its CE always trains, and every consumer (boost,
    # COMA, diagnostics) assumes it exists. Singles only (asserted in
    # get_player_model_config).
    # CE weight — same modest scale as the aux value spectrum, and for the
    # same reason (heavy aux gradient globally clips everything).
    player_q_coef: float = 0.5
    # Retrace trace parameter; matches player_lambda's 0.8 default.
    player_q_lambda: float = 0.8
    # COMA-style all-action counterfactual policy loss (2026-08-19,
    # REPLACES the stage-2 forward KL; derivation in
    # docs/entropy-gradient-pressure.md). Per legal cell:
    # adv(a) = E[Q̄_all(a)] − Σ_a' π(a')·E[Q̄_all(a')] — the COMA
    # counterfactual baseline (swap own action, hold the world fixed,
    # marginalise under the CURRENT policy) — and the loss is
    # −Σ_a π(a)·sg(adv(a)), whose per-cell gradient π(a)·adv(a) is the
    # exact all-action policy gradient: zero sampling variance, and
    # counterfactual pressure lands on EVERY real-choice row, untaken
    # actions included (the sampled boost advantage structurally carries
    # none — on a move row, retrace−v_exp says nothing about the switch
    # not taken). Still π-weighted, so it converts rare-kick recovery
    # into steady exponential re-mixing toward the critic's per-state
    # preferences rather than a mass-independent barrier — deliberate:
    # it transmits only what the critic believes per state, never a
    # fixed prior. Q̄_all is the PRIVILEGED rung (COMA's centralised
    # critic): advantages enter as stop-gradient scalars, so the policy
    # stays bitwise invariant to opp_private_team (value-ladder tests
    # still bind).
    #
    # THE policy gradient since 2026-08-21: the single-action PG and UPGO
    # terms are gone, so this is the only loss that moves the action
    # logits toward return. -adv(b) lands on the LOGITS directly (Hennes
    # et al. 2020 eq. 10, all-action over Q_all, so no 1/pi importance
    # weight and none of the sampled-NeuRD variance). COMA's own pi
    # prefactor was dropped first (2026-08-21): NeuRD eq. 6's restoring
    # force shrinks with the starved cell's own mass, and the 157k-step
    # 2026-08-20 lineage measured absadv_ratio ~4 against prob_ratio
    # ~0.075 — the critic preferred switch cells MORE than move cells and
    # pi alone throttled the update (LESSONS.md 3).
    player_coma_enabled: bool = True
    # This coefficient is now the POLICY LEARNING RATE, not a 1% nudge
    # beside a coef-1.0 PG term. History: 0.05 was sized as ~1% relative
    # pressure against normalised PG advantages; 0.1 came from the
    # 2026-08-21 NeuRD retune (dropping pi over ~10 legal cells makes the
    # raw per-cell gradient ~10x COMA's, so a starved switch cell at
    # pi 0.02 / adv +0.15 gets ~0.015/logit, the magnet's order). With PG
    # gone this must be re-sized against the magnet (the only opposing
    # force left) — judge on player_neurd_clipped_switch and normalised
    # modality entropy. Runtime scalar: retuning never recompiles.
    player_coma_coef: float = 0.1
    # NeuRD logit-gap clip beta: no outward push on a legal cell whose
    # log-policy sits more than beta from the row's legal-mean. Bounds
    # the logit spread NeuRD can build (advantages are not zero-mean
    # per row, so unclipped logits diverge); other losses still move
    # cells outside the band. 2.0 = OpenSpiel's NeuRD default.
    player_neurd_logit_clip: float = 2.0
    # Boltzmann temperature for the p_q OBSERVER kept from stage 2
    # (softmax(E[Q̄_private]/tau) diagnostics — the "what does the critic
    # want" leading indicator; no loss reads it since the stage-2 KL's
    # removal). 0.1 = sharp but not argmax.
    player_q_observer_tau: float = 0.1
    # Loss-free critic-quality diagnostics (player_q_action_var and its
    # uniform/p90 companions, calibration r2 fresh/replay); on
    # permanently. They matter more, not less, now that NeuRD through
    # Q_all is the policy's only link to return: if the critic is
    # action-flat there is nothing for it to amplify but noise.
    player_q_diagnostic_enabled: bool = True
    # Agent57/Ape-X-style exploration ladder (replaces stage 4's
    # cross-population intake, removed 2026-08-15: it conflated another
    # agent's policy evidence with main's own action values, and its
    # frozen-between-blocks stock went stale — foreign-row Q R² 0.27 vs
    # 0.84 own). Every player actor independently makes each game it
    # plays with its own live params an explore game with this
    # probability, drawing a fresh epsilon log-uniform from
    # explore_eps_range (R2D2's geometrically-spaced epsilon ladder,
    # assigned per game like Agent57's per-episode picks rather than
    # per dedicated actor slot;
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
    # 2026-08-21: epsilon-mix with the hierarchical prior REPLACES the
    # temperature draw (mu = (1-eps).pi + eps.prior per explore game;
    # HeadParams.mix). Supply arithmetic: the prior puts ~1/2 of a
    # real-choice row on switch cells, so forced-switch rows ~
    # explore_game_prob x E[eps] x 1/2 ~ 1/3 x 0.28 x 1/2 ~ 4.7% on top
    # of the ~3% the collapsed policy supplies itself -- the pre-collapse
    # ~8-10% the critic needs to keep a switch belief. Why a ladder on a
    # MINORITY of games rather than a small eps everywhere: v-trace's
    # rho-bar truncation bites where pi/mu > 1, bounded by 1/(1-eps), so
    # large eps belongs in games whose rows are few; and both sides of
    # an explore game play mu, so every explore game is against a noisier
    # opponent. ISR ESS (player_isr_ess) is the guard: below ~0.9, lower
    # the top of the range. The magnet KL no longer has to double as the
    # exploration crutch -- it can return to its regularisation-only
    # value once coverage is confirmed.
    explore_game_prob: float = 1.0 / 3.0
    explore_eps_range: tuple[float, float] = (0.1, 0.6)
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
    host computes each call and passes into the jitted train_step as ONE
    pytree argument.
    These must never move into the static config above — config is a jit
    static_argname, and static scalars retained ~5GB of executables per
    distinct value and OOM-killed run 1326; as traced leaves they change
    freely with zero recompiles. None falls back at the use site (to the
    static coef for magnet, to 0 for coma); the live Learner path always
    fills both."""

    # config.player_magnet_kl_coef.
    magnet_coef: chex.Array | None = None
    # config.player_coma_coef — the policy gradient's coefficient.
    coma_coef: chex.Array | None = None


def get_learner_config():
    return Porygon2LearnerConfig()
