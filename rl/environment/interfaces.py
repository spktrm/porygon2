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
    # The two levels' FREE logits, learner only like log_policy: macro
    # (M,) over modalities, micro (A,) over the flat src x tgt grid,
    # f32. The NeuRD loss is written against these
    # (loss.hierarchical_neurd): the composed pi_logits are already a
    # normalised log-policy, and a loss on them picks up a
    # -pi(.).sum_a w(a) cross-term the moment the logit-gap clip breaks
    # zero-sum (2026-08-24).
    macro_logits: ArrayLike = ()
    micro_logits: ArrayLike = ()


@dataclass
class PlayerActorOutput:
    value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    action_head: PlayerPolicyHeadOutput = field(default_factory=PlayerPolicyHeadOutput)
    # Learner-only (cfg.train; the advantage head is structural): (T, A)
    # scalar RAW advantage logits over the flat src x tgt action grid —
    # Q(s, a) = sg(V(s)) + A(s, a) - E_pi[A(s, .)], composed learner-side
    # (targets.residual_q). One rung since 2026-08-25, on the same
    # information set the policy acts from. Actors leave it empty so
    # replay transitions stay small.
    q_adv: ArrayLike = ()


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

    # Completed-game side data (2026-08-23), shape (1,) per chunk — (1, B)
    # batched. game_outcome: the game's terminal reward in CAT_VF_SUPPORT
    # units on EVERY chunk of a completed game (NaN for a truncated game);
    # game_length: decision rows in the game; game_step_offset: index of
    # this chunk's row 0 within the game. Telemetry and the offline harness
    # ONLY — never a target recursion (win_reward keeps that role) and not
    # part of the chunk contract. () where the actor did not attach them.
    game_outcome: ArrayLike = ()
    game_length: ArrayLike = ()
    game_step_offset: ArrayLike = ()


@dataclass
class Batch(Trajectory):
    rng_key: ArrayLike = ()
