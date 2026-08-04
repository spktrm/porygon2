from collections.abc import Callable
from typing import Literal

import chex
import jax.numpy as jnp

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

    # Plasticity (shrink-and-perturb) params. Triggered when the main player
    # keeps failing to dominate its own league history: after
    # `plasticity_overdue_trigger` consecutive overdue-only league additions,
    # player params are interpolated toward a fresh init draw.
    plasticity_enabled: bool = True
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
    builder_ema_update_rate: float = 1e-3

    # Advantage estimation params
    player_gamma: float = 1.0
    player_alpha: float = 1.0
    player_lambda: float = 0.99

    # Learning-progress bandit over the main v-trace lambda (the
    # policy-target mixture meta-controller; rl/online/bandit.py). Every
    # bandit_window_steps the learner fits a Bradley-Terry rating for
    # main against the frozen league snapshots from the payoff table the
    # league already keeps, rewards the live arm with the (scale-aligned)
    # rating gain over the window, and picks the next arm by discounted
    # UCB. Pure self-play signal: mirror games carry no reward and the
    # scripted eval baselines are never consulted. Before
    # bandit_min_rated_opponents snapshots have
    # bandit_min_games_per_opponent effective games against main, no
    # reward exists — the bandit idles on bandit_default_arm (the current
    # production lambda) and re-baselines at the first valid fit. Each
    # distinct arm costs one extra jit compile of train_step (config is a
    # static argument).
    bandit_enabled: bool = True
    bandit_lambdas: tuple[float, ...] = (0.90, 0.99, 1.0)
    bandit_default_arm: int = 1
    bandit_window_steps: int = 20_000
    bandit_ucb_c: float = 0.25
    bandit_discount: float = 0.9
    bandit_min_games_per_opponent: float = 20.0
    bandit_min_rated_opponents: int = 2

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
    # Fixed, independent of config.player_lambda, which the mixture
    # bandit varies per window. Length must match the model config's
    # aux_v_head.num_heads.
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


