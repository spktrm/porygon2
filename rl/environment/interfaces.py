from dataclasses import field

from chex import dataclass
from jaxtyping import ArrayLike


@dataclass
class PlayerEnvOutput:
    # Standard Info
    info: ArrayLike = ()
    done: ArrayLike = ()
    win_reward: ArrayLike = ()
    public_team: ArrayLike = ()
    revealed_team: ArrayLike = ()
    field: ArrayLike = ()
    opp_moveset: ArrayLike = ()

    # Private Info
    my_moveset: ArrayLike = ()
    private_team: ArrayLike = ()
    # Privileged (training self-play only): the opponent's match-start team
    # sheet, frozen at THEIR first request — all-zero at deploy time. Feeds
    # ONLY the everything-value readout; the policy/state streams never see
    # it (rl/model/encoder.py RoundBlock).
    opp_private_team: ArrayLike = ()

    action_mask: ArrayLike = ()


@dataclass
class PlayerPackedHistoryOutput:
    public_cache: ArrayLike = ()
    revealed_cache: ArrayLike = ()
    edge_cache: ArrayLike = ()


@dataclass
class PlayerHistoryOutput:
    field: ArrayLike = ()


@dataclass
class PlayerActorInput:
    env: PlayerEnvOutput = field(default_factory=PlayerEnvOutput)
    packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)


@dataclass
class RegressionValueHeadOutput:
    logits: ArrayLike = ()


@dataclass
class CategoricalValueHeadOutput:
    logits: ArrayLike = ()
    log_probs: ArrayLike = ()
    entropy: ArrayLike = ()
    expectation: ArrayLike = ()
    l2_norm: ArrayLike = ()


@dataclass
class PolicyHeadOutput:
    action_index: ArrayLike = ()
    log_prob: ArrayLike = ()
    entropy: ArrayLike = ()
    normalized_entropy: ArrayLike = ()
    log_policy: ArrayLike = ()
    magnet_kl: ArrayLike = ()


@dataclass
class PlayerPolicyHeadOutput(PolicyHeadOutput):
    src_index: ArrayLike = ()
    tgt_index: ArrayLike = ()
    normalized_modality_entropy: ArrayLike = ()
    # Raw composed action-grid logits (masked cells at the fill value);
    # learner only, like log_policy. The NeuRD loss needs the LOGITS --
    # written against log_policy the logit-gap clip reintroduces a
    # pi(b).sum_a w(a) cross-term once clipped cells break zero-sum.
    logits: ArrayLike = ()
    # KL(mu || pi) of the optimistic behaviour tilt at this step (0 when
    # HeadParams.ucb_c is 0). Stored by actors: it is the per-state cost
    # of optimism and the dial's natural unit.
    ucb_kl: ArrayLike = ()


@dataclass
class PlayerActorOutput:
    value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    action_head: PlayerPolicyHeadOutput = field(default_factory=PlayerPolicyHeadOutput)
    # Learner-only (cfg.train; the Q critic is structural): (T, A, n_bins)
    # categorical logits of the all-action Q critic over the flat src x tgt
    # action grid (docs/q-critic-plan.md). q_logits is the privileged rung
    # (conditioned on value_all — drives the Retrace recursion);
    # private_q_logits shares every head param but is conditioned on the
    # deployable information set, and trains by CE against the same Retrace
    # labels. Actors leave both empty so replay transitions stay small.
    q_logits: ArrayLike = ()
    private_q_logits: ArrayLike = ()
    # Learner-only (cfg.train): (T, n_bins) categorical logits of the
    # counterfactual value ladder — `private` sees the deployable information
    # set (no opponent team sheet), `public` sees the history context only.
    # The main value_head reads the privileged everything-stream. Gaps
    # between the three expectations are per-state value-of-information
    # readouts.
    private_value_logits: ArrayLike = ()
    public_value_logits: ArrayLike = ()
    # Learner-only (cfg.train), private rung: (T, K, n_bins) logits of the
    # bootstrapped value ensemble whose spread std_k E[V_k] is the
    # intrinsic reward, and (T,) scalar V_int — the critic of that
    # reward's discounted return. Actors leave both empty.
    ens_value_logits: ArrayLike = ()
    int_value: ArrayLike = ()
    # Learner-only (cfg.train), private rung: (T, A, K, n_bins) logits of
    # the bootstrapped Q ensemble (+ frozen prior) whose epistemic /
    # aleatoric spread drives the optimistic behaviour policy. Computed
    # at act time too (the tilt needs it) but never stored.
    q_ens_logits: ArrayLike = ()


@dataclass
class PlayerAgentOutput:
    actor_output: PlayerActorOutput = field(default_factory=PlayerActorOutput)


@dataclass
class PlayerTransition:
    env_output: PlayerEnvOutput = field(default_factory=PlayerEnvOutput)
    agent_output: PlayerAgentOutput = field(default_factory=PlayerAgentOutput)


@dataclass
class BuilderEnvOutput:
    species_mask: ArrayLike = ()
    item_mask: ArrayLike = ()
    ability_mask: ArrayLike = ()
    move_mask: ArrayLike = ()
    hp_ev_mask: ArrayLike = ()
    atk_ev_mask: ArrayLike = ()
    def_ev_mask: ArrayLike = ()
    spa_ev_mask: ArrayLike = ()
    spd_ev_mask: ArrayLike = ()
    spe_ev_mask: ArrayLike = ()
    teratype_mask: ArrayLike = ()
    nature_mask: ArrayLike = ()
    gender_mask: ArrayLike = ()

    species_usage: ArrayLike = ()
    item_usage: ArrayLike = ()
    ability_usage: ArrayLike = ()
    move_usage: ArrayLike = ()
    hp_ev_usage: ArrayLike = ()
    atk_ev_usage: ArrayLike = ()
    def_ev_usage: ArrayLike = ()
    spa_ev_usage: ArrayLike = ()
    spd_ev_usage: ArrayLike = ()
    spe_ev_usage: ArrayLike = ()
    teratype_usage: ArrayLike = ()
    nature_usage: ArrayLike = ()
    gender_usage: ArrayLike = ()

    done: ArrayLike = ()
    ts: ArrayLike = ()
    ev_reward: ArrayLike = ()
    curr_order: ArrayLike = ()
    curr_attribute: ArrayLike = ()
    curr_position: ArrayLike = ()
    validator_reward: ArrayLike = ()


@dataclass
class BuilderHistoryOutput:
    packed_team_member_tokens: ArrayLike = ()
    order: ArrayLike = ()
    member_position: ArrayLike = ()
    member_attribute: ArrayLike = ()


@dataclass
class BuilderActorInput:
    env: BuilderEnvOutput = field(default_factory=BuilderEnvOutput)
    history: BuilderHistoryOutput = field(default_factory=BuilderHistoryOutput)


@dataclass
class BuilderActorOutput:
    action_head: PolicyHeadOutput = field(default_factory=PolicyHeadOutput)
    conditional_entropy_head: RegressionValueHeadOutput = field(
        default_factory=RegressionValueHeadOutput
    )
    value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )


@dataclass
class BuilderAgentOutput:
    actor_output: BuilderActorOutput = field(default_factory=BuilderActorOutput)


@dataclass
class BuilderTransition:
    env_output: BuilderEnvOutput = field(default_factory=BuilderEnvOutput)
    agent_output: BuilderAgentOutput = field(default_factory=BuilderAgentOutput)


@dataclass
class PlayerTargets:
    win_returns: ArrayLike = ()
    policy_mask: ArrayLike = ()
    value_mask: ArrayLike = ()


@dataclass
class IntrinsicTargets:
    """Learner-side intrinsic-reward channel (targets.compute_intrinsic_targets).

    int_reward: (T, B) RMS-normalised r_int_t = std_k E[V_k(s_{t+1})]
    (0 on terminal / past-done / bootstrap-only rows); int_returns: (T, B)
    v-trace targets for V_int under player_int_gamma; int_adv: (T, B)
    rho-weighted one-step advantage r + gamma * int_returns_{t+1} - V_int
    along the taken action; ens_std: (T, B) the raw spread at s_t;
    int_rms_new: () the running mean-square after this batch.
    """

    int_reward: ArrayLike = ()
    int_returns: ArrayLike = ()
    int_adv: ArrayLike = ()
    ens_std: ArrayLike = ()
    int_rms_new: ArrayLike = ()


@dataclass
class BuilderTargets:
    win_returns: ArrayLike = ()
    win_advantages: ArrayLike = ()
    ent_advantages: ArrayLike = ()
    ent_returns: ArrayLike = ()


@dataclass
class Trajectory:
    builder_transitions: BuilderTransition = field(default_factory=BuilderTransition)
    builder_history: BuilderHistoryOutput = field(default_factory=BuilderHistoryOutput)

    player_transitions: PlayerTransition = field(default_factory=PlayerTransition)
    player_packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    player_history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)

    # How many times this trajectory had been sampled BEFORE this one, shape
    # (1,) per trajectory — (1, B) once batched. Attached at sample time by
    # PlayerTrajectoryStore (0 = first visit) and consumed by the
    # fresh-vs-replayed value-error plasticity diagnostic. () outside the
    # learner's sampling path.
    reuse_count: ArrayLike = ()

    # Exploration-ladder flag, shape (1,) per trajectory — (1, B) once
    # batched. True = this game was played under the epsilon-mixed
    # behaviour policy mu = (1-eps).pi + eps.prior (config.
    # explore_game_prob / explore_eps_range). These rows train EVERY
    # player loss since 2026-08-17 — their ISRs are exact, so the
    # off-policy correction handles them; the flag now gates only the
    # signals a deliberately noisier policy WOULD bias: league cadence,
    # builder losses, the replay controller. Set at construction
    # by PlayerActor; normalised at batch-assembly time (all-False when
    # unset) so the shared train_step jit sees ONE pytree structure. ()
    # outside the online path.
    explore: ArrayLike = ()


@dataclass
class Batch(Trajectory):
    rng_key: ArrayLike = ()
