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
    player_magnet_kl_coef: float = 0.01

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
    # λ for the potential advantage channel ONLY (win channel keeps
    # player_lambda). Kept low so the channel's advantage stays close to
    # the one-step PBRS signal γΦ(s')−Φ(s): dense per-move credit from the
    # frozen critic. At λ→1 the channel telescopes to Φ(s_T)−Φ(s_t) ≈
    # outcome − baseline — a rescaled copy of the win channel that adds no
    # action-level information (the observed no-op shaping regime,
    # player_potential_win_adv_corr ≈ 0.6 with zero winrate effect).
    player_potential_lambda: float = 0.2

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

    # Potential-based shaping (requires offline_critic_ckpt_path): target
    # share of the combined advantage magnitude held by the potential
    # channel. The channel coefficient is solved per batch from measured
    # channel stds (coef = s/(1−s) · σ_win/σ_pot, capped below), so
    # dominance stays pinned at the target while either channel's scale
    # drifts over a long hold — a fixed coef cannot guarantee that.
    # Schedule: hold dominant (0.8) for 500k steps so early learning
    # follows the human-replay critic's value landscape, then anneal to
    # zero over 100k steps so the asymptotic objective is pure win/loss.
    # Judge via player_potential_adv_share (realised share, should track
    # the target) and player_potential_adv_coef (the solved coef).
    player_potential_target_adv_share_fn: Callable[[int], float] = (
        lambda step: 0.8 * jnp.clip((600_000 - step) / 100_000, 0.0, 1.0)
    )
    # Cap on the solved coefficient: keeps a near-silent potential channel
    # (tiny σ_pot from heavy uncertainty gating or a flat Φ) from being
    # amplified into pure noise to hit the share target. Must sit above the
    # coefficients the solver actually needs, or the share mechanism is
    # inert: at 30 the cap bound 100% of steps in the July 2026 run and
    # realised adv_share ran ~0.55 against the 0.8 target, while the
    # channel's quality signals stayed healthy (terminal_agreement 1.0,
    # win_adv_corr ≈ 0). Judge via player_potential_adv_coef staying below
    # the cap, with player_potential_adv_sign_flip and actor-KL as the
    # noise guardrails.
    player_potential_coef_max: float = 100.0

    # Loss coefficients
    ## Player
    player_policy_loss_coef: float = 1.0
    player_kl_loss_coef: float = 0.05
    player_value_head_loss_coef: float = 1.0

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

    # Standalone offline critic (rl/offline/train.py artifact) used as the
    # learned state potential in compute_player_targets. Loaded once at
    # learner startup and held OUTSIDE the train state: its params never
    # enter the optimizer or the RL network, so the RL model trains fully
    # from scratch with no frozen or warm-started subtrees. The potential
    # advantage channel is gated by player_potential_target_adv_share_fn.
    # A tuple of paths loads an ensemble (members trained with
    # rl.offline.train --ensemble-index k) for uncertainty-gated shaping.
    offline_critic_ckpt_path: str | tuple[str, ...] | None = tuple(
        f"ckpts/offline/gen9randombattle-ens{k}/ckpt_best" for k in range(4)
    )
    # Ensemble-disagreement gate: Φ = mean * exp(-scale * std). Where the
    # members disagree (off the human data distribution) shaping goes
    # quiet. 0 disables; irrelevant for single-member critics. With
    # potential_gate_scale_learned this is only the INITIAL scale.
    potential_uncertainty_scale: float = 5.0
    # Learn the gate scale online from the run's own outcomes: every
    # trajectory yields (ensemble mean, ensemble std, final result)
    # triplets on exactly the self-play state distribution the gate must
    # serve, so the scale is a 1-parameter regression — periodically solve
    # c* = argmin_c E[(mean·exp(-c·std) - outcome)^2] over a reservoir of
    # triplets and EMA the live scale toward it. Replaces the hand-picked
    # 5.0, which offline calibration suggested was over-shrinking
    # (member std ~0.085 -> gate ~0.65 at ~78% gated sign accuracy).
    # Judge via potential_gate_scale (the live value) vs
    # potential_gate_scale_solved (each fit's argmin).
    potential_gate_scale_learned: bool = True
    potential_gate_scale_min: float = 0.0
    potential_gate_scale_max: float = 20.0
    # Solve cadence (learner steps), reservoir capacity (triplets), states
    # sampled per trajectory, minimum fill before the first solve, and the
    # per-solve EMA step toward c*. The reservoir is a ring buffer, so the
    # fit always reflects recent policy/state distribution.
    potential_gate_scale_interval: int = 1000
    potential_gate_scale_buffer: int = 32768
    potential_gate_scale_samples: int = 8
    potential_gate_scale_min_samples: int = 4096
    potential_gate_scale_ema: float = 0.2
    # What Φ reads off the critic's 13-bin margin distribution (training
    # stays distributional either way): "win" = P(win) − P(loss) — pure
    # outcome belief, flat across decided positions, never prefers a wider
    # win over a likelier one; "margin" = expected margin — also grades
    # decisiveness, denser shaping inside decided positions but can
    # transiently reward margin-seeking over win-optimal lines.
    potential_readout: str = "win"
    # With an Elo-conditioned critic (auto-detected from the artifact),
    # condition Φ at this rating: shaping then reflects how games between
    # players of that strength resolve, not ladder-average conversion.
    # 0 = leave the inputs alone (self-play carries no ratings, so the
    # critic uses its unknown-rating bucket). Ignored for pre-rating
    # artifacts. Conditioning far above the data's rating support makes
    # ensemble members disagree, which the uncertainty gate then quiets.
    potential_condition_rating: int = 1800
    # Dice-excised PBRS: the potential channel's next-step bootstrap uses
    # Φ_ann(t+1) — the critic read at the announced state, where both
    # players' choices for the turn are revealed but chance is unresolved —
    # in place of the realised Φ(t+1). Φ_ann = E[Φ | announcement] (tower
    # property ⇒ same expected shaping, verified conditionally unbiased by
    # rl.offline.diagnose --martingale), so the channel stops paying the
    # agent for crits, misses and damage rolls: measured on the July 2026
    # critic, ~85% of the channel's per-turn variance was resolved chance.
    # Expect potential_adv_coef to rise ~σ-ratio at unchanged adv_share
    # (denser signal, same loudness). Requires every artifact in
    # offline_critic_ckpt_path to be announced-trained (manifest
    # announced_states — Φ_ann adds no params, so this is checked from the
    # manifests at startup and fails loudly). A/B off vs on, judged by
    # actor-KL (~0.045 reference) and strength-per-step; False is the
    # realised-PBRS comparison arm.
    potential_dice_excised: bool = True


def get_learner_config():
    return Porygon2LearnerConfig()


