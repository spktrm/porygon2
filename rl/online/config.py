from typing import Literal

import chex

from rl.config.common import AdamWConfig, BaseTrainingConfig

PolicyObjectiveT = Literal["spo"]


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
    # Fast EMA target (IMPACT-style): supplies the clipped-target ratio in
    # the surrogate, the v-trace reference policy, and the value bootstraps,
    # so it must track the learner closely for stability under replay reuse.
    # (R-NaD likewise keeps a 1e-3 target purely for v-trace stability,
    # separate from its slow anchors.)
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
    # harder to predict — see rl/online/controllers.py's module docstring
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

    # Builder policy objective: ratio-based surrogate with a trust region.
    # The player has no ratio-surrogate term anymore — the single-action PG
    # and UPGO losses were removed 2026-08-21 (LESSONS.md 3).
    builder_policy_objective: PolicyObjectiveT = "spo"
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
    # everything (LESSONS.md 5).
    player_q_coef: float = 0.5
    # No trace parameter since 2026-08-23 (Step 3): the residual critic
    # regresses on the TD(0) label r + gamma*V_win_target(s'), and the
    # policy reads that critic directly. Retrace at q_lambda 1.0 / pi
    # lambda 0.8 (the outcome chain within the chunk) is in git history;
    # the Step-6 probe showed the categorical head fitting those labels
    # through a state-only route, which the residual form closes.
    # THE policy gradient. All-action NeuRD (Hennes et al. 2020 eq. 10):
    # per legal cell of every real-choice row,
    # adv(a) = E[Q̄_all(a)] − Σ_a' π(a')·E[Q̄_all(a')] (the COMA
    # counterfactual baseline — swap own action, hold the world fixed,
    # marginalise under the CURRENT policy), centred over legal cells,
    # applied to the RAW LOGITS with no π prefactor. Zero sampling
    # variance, and counterfactual pressure lands on untaken actions —
    # which a sampled objective structurally cannot do (on a move row,
    # retrace−v_exp says nothing about the switch not taken). Advantages
    # enter as stop-gradient scalars off the target net.
    #
    # Lineage: this was COMA proper (−π(b)·adv(b) per logit) until
    # 2026-08-21. NeuRD eq. 6's restoring force shrinks with the starved
    # cell's own mass, and the 157k-step 2026-08-20 run measured
    # absadv_ratio ~4 against prob_ratio ~0.075 — the critic preferred
    # switch cells MORE than move cells and π alone throttled the update
    # (LESSONS.md 3). The single-action PG and UPGO terms went the same
    # day, leaving this as the only loss that moves the action logits
    # toward return.
    #
    # 1.0 (up from 0.1, 2026-08-21): this is the policy learning rate
    # now, so it takes the coefficient the PG term it replaced carried.
    # Scale check — the PG advantages it stands in for were EMA-
    # normalised to ~unit std, while these are raw win units (~0.1-0.2
    # spread), so at 1.0 a starved switch cell at adv +0.15 gets
    # ~0.15/logit: ~10x its pull at 0.1, and about an order above the
    # magnet's. That asymmetry is the point (the magnet is the only
    # opposing force left), but it is also the thing to watch: if
    # normalised modality entropy cliffs or player_neurd_clipped_switch
    # pins at 1, the clip rather than the critic is bounding switch mass
    # — back this off or raise the magnet. Static config since
    # 2026-08-21, so retuning it costs one train_step recompile at the
    # next launch (nothing varies it during a run any more).
    # 0.2 (2026-08-21 eve, from 1.0): the first run at 1.0 tripped the
    # condition above inside 6k steps -- pi(switch) 0.16 -> 0.011 with the
    # critic's gap still -0.04, clipped_switch 0.99 by 10k, switch mass
    # ~0.1% per cell (the band floor) -- a near-zero early gap
    # transmitted at full volume before the critic had a belief. 0.1
    # with the old PG still present never saturated past ~0.7; 0.1-0.2
    # brackets it.
    # 0.1 (2026-08-21 night, from 0.2): 0.2 on a fresh lineage tripped
    # the same wire, only slower -- clipped_switch 0 -> 0.83 by 12.9k,
    # switch_ratio 0.47 -> 0.076, switch_push pinned at -0.035, magnet KL
    # 0.43 and climbing; no checkpoint yet (first save 20k) so relaunched
    # from scratch.
    # 0.2 (2026-08-22, from 0.1, with magnet 0.1 in place): 0.1 + magnet
    # 0.1 held switching but learnt nothing -- normalised entropy 0.9+
    # parked next to the anchor (magnet KL ~0.1, not straining), eval wr
    # vs the heuristic flat at 0.07 from 48k to 65k while the collapsed
    # 0.2/0.05 lineage was at 0.15-0.23 by 20k. NeuRD is the sole
    # improvement term now, so halving it halved the learning. Tests
    # whether magnet 0.1 holds 0.2 where 0.05 didn't; resumed @60k.
    # 2026-08-22: the magnet is gone; the opposing force is now inside
    # the advantage itself (player_ref_eta), so this coef scales both
    # the improvement and the regularisation together, as in rnad.py.
    player_neurd_coef: float = 0.2
    # Step-2 warm-up (docs/critic-weakness-analysis.md, 2026-08-23): NeuRD's
    # coefficient ramps 0 -> player_neurd_coef linearly over the lineage's
    # first N learner steps, and the reference snap is gated off (reference
    # policy frozen at the launch snapshot) until the ramp completes, so
    # player_ref_kl reads KL(pi || pi_launch) = policy drift from
    # launch. The Q routes are zero-initialised and NeuRD consumed an
    # immature Q from step 0, reshaping the behaviour distribution before
    # the critic had any action coverage — the support loss began at
    # launch, not at 13k (run 3sc7wlgq). 11k = pre-registered first
    # schedule only (the Q R2 plateau age on that run); acceptance is the
    # panel set in the plan doc plus a >=5k-step hold at full coefficient.
    # 0 disables. Traced from step_count inside train_step, so a resumed
    # lineage never re-ramps.
    player_neurd_warmup_steps: int = 20_000
    # NeuRD logit-gap clip beta: no outward push on a legal cell whose
    # log-policy sits more than beta from the row's legal-mean. Bounds
    # the logit spread NeuRD can build (advantages are not zero-mean
    # per row, so unclipped logits diverge); other losses still move
    # cells outside the band. 2.0 = OpenSpiel's NeuRD default.
    player_neurd_logit_clip: float = 2.0
    # Quadratic decay on each level's centred free logits, inside the
    # NeuRD bracket (shares neurd_coef and the warm-up ramp, so the
    # per-cell fixed point |centred logit| = |w| / decay is coef- and
    # ramp-invariant). The linear all-action loss has no inward force
    # inside the +-beta band; a persistent same-sign modality advantage
    # therefore integrates without bound — macro-head grad runaway at
    # ~64k on three lineages (2026-08-25). 0.05 puts the fixed point at
    # the band edge for |w| = 0.1 (the observed coherent macro push) and
    # is mass-independent: starved cells are pulled back toward the
    # legal-set mean, not just dominant ones pulled down.
    player_neurd_logit_decay: float = 0.05
    # Reference-policy penalty in the NeuRD advantage: the ONLY place the
    # reference enters since the reg-value stream was deleted
    # (2026-08-25). Per legal cell, analytically, -eta*(log pi - log
    # pi_reg) is added to Q before centring (targets.ref_penalised_q);
    # both critics learn the PLAIN game — no reward transform, no
    # transformed bootstrap. pi_reg is a periodic SNAP of the target
    # params (player_reg_snap_steps).
    # This is NashPG (arXiv:2510.18183, Oct 2025): own-side KL to an
    # iteratively refined reference in the POLICY objective, no reward
    # transformation, reported at or below R-NaD's exploitability — and
    # its independent sensitivity study lands on alpha = 0.2 with a
    # U-shaped curve, which is where DeepNash's eta already put us.
    # Own-side only, as NashPG's objective also is (the opponent's
    # +eta*KL_opp would need the other side's chunk — no game id in a
    # Trajectory), so the regularised game is not exactly zero-sum.
    # NOTE the analytic advantage shift is deliberate and is NOT NashPG's
    # differentiated KL: d/dz KL(pi||rho) carries a pi prefactor and so
    # vanishes on a starved cell, whereas shifting the advantage under
    # NeuRD's logit gradient makes the refill force GROW as pi -> 0. Same
    # lesson as the gradient-side magnet this replaced. 0 = plain NeuRD.
    player_ref_eta: float = 0.2
    # Snap period of the reference (2026-08-25, replacing the continuous
    # EMA — 1e-4, then 5e-5): reg_params <- target_params, in place,
    # every N steps once the NeuRD warm-up is done (rnad.py's delta_m
    # reset without the prev/prev_ crossfade pair — no 4th param set).
    # The EMA never reset, so the log(pi/pi_reg) gap compounded with
    # policy speed into the grad-norm runaways of pgaijs6l (56k) and
    # 2wvnlsz3 (79.6k, ref_kl 2.07 nats, p90 62k by 98k): the penalty is
    # unbounded in the gap by design, so the GAP is bounded structurally
    # instead. 20k = rnad.py's delta_m; also divides the warm-up so the
    # first snap lands exactly at ramp completion, and a 20k-multiple
    # resume snaps at its first step.
    player_reg_snap_steps: int = 20_000
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
