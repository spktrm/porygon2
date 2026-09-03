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
    # The OPPONENT's request, same row schema as private_team (2026-09-01).
    # Learner-only truth: the encoder routes it into the leak-masked trunk
    # partition; empty on old shards and at deploy, decoded as zeros.
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
class HistoryCarry:
    """The history encoder's state after a window (2026-09-02), so an actor
    can feed the NEXT request's new steps alone and resume the recursion
    instead of re-running the whole window. Optional by construction: every
    leaf defaults to () and the encoder then starts from its learned h0
    exactly as it always has (the learner, the offline tools and any caller
    sending a full window never build one); with leaves present, `valid`
    selects between the carried state and h0 per call, so a False carry is
    also the from-scratch function. slot_states (12, D) and field_states
    (3, D) are f32 -- the scan's own recursion state, taken before the
    compute-dtype cast; node_snapshots (12, D) is the compute dtype.
    """

    slot_states: ArrayLike = ()
    field_states: ArrayLike = ()
    node_snapshots: ArrayLike = ()
    valid: ArrayLike = ()


@dataclass
class PlayerActorInput:
    env: PlayerEnvOutput = field(default_factory=PlayerEnvOutput)
    packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)
    # Actor-only: the state after the window the PREVIOUS request was
    # answered from, when `history` holds only the steps since. () = none.
    history_carry: HistoryCarry = field(default_factory=HistoryCarry)


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
    # `src_index`/`tgt_index` lived here until 2026-08-31: coordinates into
    # the 41x41 scoring grid the wire Action used to carry. `action_index`
    # IS the wire action now -- an index into the block space.
    normalized_modality_entropy: ArrayLike = ()


@dataclass
class PlayerActorOutput:
    value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    action_head: PlayerPolicyHeadOutput = field(default_factory=PlayerPolicyHeadOutput)
    # Learner-only (cfg.train), like log_policy: the privileged critic over
    # the VALUE_CLS row, and the opponent discrete-code one-hot (T, 6, G, K)
    # that is the belief head's label. Actors ship the () defaults.
    priv_value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    opp_code: ArrayLike = ()
    # The belief head: (T, 6, G, K) logits predicting opp_code from the
    # matched PUBLIC rows, and the per-mon alignment mask.
    belief_logits: ArrayLike = ()
    # The species-only matched control: the same (T, 6, G, K) logits from
    # a table keyed on the matched public row's species token alone.
    species_belief_logits: ArrayLike = ()
    belief_matched: ArrayLike = ()
    # The dynamics head (2026-09-03): (T, NUM_DYNAMICS_ROWS, D) pre-trunk
    # content of the target rows (the EMA forward's copy is the label) and
    # the online head's prediction of each row's NEXT-step content from the
    # post-trunk row and the taken cell's readout rows. Learner-only.
    dynamics_target: ArrayLike = ()
    dynamics_pred: ArrayLike = ()
    # Trunk row homogeneity per step (rl/model/trunk.py row_homogeneity):
    # mean off-diagonal cosine and participation ratio over the valid rows
    # of the trunk's output. The over-smoothing instrument; learner-only.
    trunk_row_cosine: ArrayLike = ()
    trunk_row_participation: ArrayLike = ()
    # History-encoder telemetry (history_encoder.history_step_stats), one
    # per-trajectory scalar broadcast over T: the step GAT's normalised
    # attention entropy, the mass non-source rows place on source rows
    # beside its uniform baseline, and the backbone's mean write gate.
    history_step_attn_entropy: ArrayLike = ()
    history_step_attn_to_src: ArrayLike = ()
    history_step_attn_to_src_uniform: ArrayLike = ()
    history_gate_mean: ArrayLike = ()
    # The history state after this forward's window (`valid` always True
    # here), for the actor to hand back as the next request's
    # `PlayerActorInput.history_carry`. Stripped before a transition is
    # stored (`without_history_carry`): chunks never carry (12, D) tensors,
    # and the learner's forward drops the computation as unread.
    history_carry: HistoryCarry = field(default_factory=HistoryCarry)

    def without_history_carry(self) -> "PlayerActorOutput":
        return self.replace(history_carry=HistoryCarry())

    # `advantage` and `q` lived here until 2026-08-29: the learner-only
    # Q = V + A decomposition over the flat src x tgt grid, composed in the
    # model by heads.compose_q. The policy stopped reading it at the NashPG
    # switch, which left it a matched-control observer for an architecture
    # that no longer exists; its last readings are banked in the ledger.


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
    pg_advantages: ArrayLike = ()
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
