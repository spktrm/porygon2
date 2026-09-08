from dataclasses import field, fields

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
class SearchOutput:
    """Per-step diagnostics of the search eval arm (rl/model/search.py),
    populated only when `cfg.search.enabled`; every leaf `()` otherwise.
    `root_kl` is KL(pi_search || pi) over legal cells -- the operator's
    size; `root_value_gap` the search policy's expected Q minus V at the
    root; `legal_truncated` whether the root had more legal cells than
    `cfg.search.max_cells` (the base policy is returned untouched there,
    counted). Depth 2 adds `deep_gain` (the mean over root cells of the
    depth-2 backup minus the depth-1 value), `deep_continue` (the mean
    predicted continuation at the depth-1 nodes), `candidate_retained_mass`
    and `candidate_occupied` (the generator's support at those nodes)."""

    root_kl: ArrayLike = ()
    search_value: ArrayLike = ()
    root_value_gap: ArrayLike = ()
    num_legal: ArrayLike = ()
    legal_truncated: ArrayLike = ()
    deep_gain: ArrayLike = ()
    deep_continue: ArrayLike = ()
    candidate_retained_mass: ArrayLike = ()
    candidate_occupied: ArrayLike = ()
    mcts_visits: ArrayLike = ()
    # Accepted cached expansions; vmap can also execute masked model work.
    mcts_model_calls: ArrayLike = ()
    mcts_depth_reached: ArrayLike = ()


@dataclass
class PlayerActorOutput:
    value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    action_head: PlayerPolicyHeadOutput = field(default_factory=PlayerPolicyHeadOutput)
    # Learner-only (cfg.train), like log_policy: the privileged critic over
    # the VALUE_CLS row, the opponent discrete-code one-hot (T, 6, G, K)
    # the secret rows are built from, and the belief head's LABEL: the
    # same code over each mon's HIDDEN tokens only (2026-09-05,
    # encoder.OppCodeLabels). Actors ship the () defaults.
    priv_value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    opp_code: ArrayLike = ()
    hidden_code: ArrayLike = ()
    # The belief head: (T, 6, G, K) logits predicting hidden_code from the
    # matched PUBLIC rows, the per-mon alignment mask, and whether the mon
    # has any hidden token left to predict.
    belief_logits: ArrayLike = ()
    # The species-only matched control: the same (T, 6, G, K) logits from
    # a table keyed on the matched public row's species token alone.
    species_belief_logits: ArrayLike = ()
    # The revealed-row matched control (2026-09-04): the same logits from
    # an MLP over the matched mon's own PRE-trunk public row alone.
    revealed_belief_logits: ArrayLike = ()
    belief_matched: ArrayLike = ()
    belief_hidden_any: ArrayLike = ()
    # The transition model's grounding label (2026-09-03, dynamics head;
    # 2026-09-05 relabelled): (T, NUM_DYNAMICS_ROWS, D) pre-trunk content
    # of the target rows. Step t's grounding head is scored against step
    # t+1's copy, aligned by dynamics_alignment. Learner-only.
    dynamics_target: ArrayLike = ()
    # The latent transition model (2026-09-05; latent actions and the K-step
    # unroll 2026-09-07, rl/model/transition.py). Learner-only. Three
    # leading-axis families, every one followed by the trajectory axis T:
    # - per START step (T, ...): cons_err / cons_scale (T, 73) f32 -- the
    #   first imagined step's squared distance from the real next row, and
    #   the real next row's from the current one (the copy predictor's
    #   error, the normaliser); ground (T, NUM_DYNAMICS_ROWS, D) the
    #   grounding head on the first posterior-decoded step and
    #   ground_prior the same read of a no-gradient prior-mode decode;
    #   value_head_prior the frozen critic on that prior-mode decode;
    #   pred_rms (T,) rms(pred) / rms(rows) over the rows valid at t;
    #   newly_valid (T,) whether a row zeroed at t exists at t+1 (a panel
    #   mask); action_logits (T, max_cells, C) the action encoder over the
    #   enumerated legal cells with action_cells / action_cell_valid /
    #   action_taken_index / action_overflow the enumeration;
    # - per TRANSITION k = 0 .. K-1 (K, T, ...), paired with the real step
    #   t+k+1: prior / post logits and post_one_hot (K, T, G, Kc);
    #   action_one_hot (K, T, C) the latent action drawn for the step;
    #   value_head the shared critic on the imagined CLS row (LIVE under
    #   `transition.value_trains_v_head`, else a frozen clone);
    #   kind_logits (K, T, 4), done_logit (K, T) and terminal_logits
    #   (K, T, 3) the imagined CLS row's request kind, done and
    #   conditional terminal outcome;
    # - per NODE k = 0 .. K (K+1, T, ...), hhat_0 the real state: the
    #   generator teacher-forced on the real step t+k's teacher order
    #   (generator_logits (K+1, T, J, C), teacher_codes (K+1, T, J),
    #   generator_target (K+1, T, C) = sg p(u | h_{t+k}), support_mask
    #   (K+1, T, C), node_overflow (K+1, T)), and the encoder on the
    #   recorded cell at the node (align_logits (K+1, T, C)) against its
    #   real-state distribution (align_target (K+1, T, C), sg).
    transition_cons_err: ArrayLike = ()
    transition_cons_scale: ArrayLike = ()
    # Target-label mask (T, rows): present at either endpoint, never an
    # input to imagination. Batched forwards insert B after T.
    transition_cons_valid: ArrayLike = ()
    transition_prior_logits: ArrayLike = ()
    transition_post_logits: ArrayLike = ()
    transition_post_one_hot: ArrayLike = ()
    transition_ground: ArrayLike = ()
    transition_ground_prior: ArrayLike = ()
    transition_value_head: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    transition_value_head_prior: CategoricalValueHeadOutput = field(
        default_factory=CategoricalValueHeadOutput
    )
    transition_pred_rms: ArrayLike = ()
    transition_newly_valid: ArrayLike = ()
    transition_kind_logits: ArrayLike = ()
    transition_done_logit: ArrayLike = ()
    transition_terminal_logits: ArrayLike = ()
    transition_action_logits: ArrayLike = ()
    transition_action_cells: ArrayLike = ()
    transition_action_cell_valid: ArrayLike = ()
    transition_action_taken_index: ArrayLike = ()
    transition_action_overflow: ArrayLike = ()
    transition_action_one_hot: ArrayLike = ()
    transition_generator_logits: ArrayLike = ()
    transition_teacher_codes: ArrayLike = ()
    transition_generator_target: ArrayLike = ()
    transition_support_mask: ArrayLike = ()
    transition_node_overflow: ArrayLike = ()
    transition_align_logits: ArrayLike = ()
    transition_align_target: ArrayLike = ()
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
    # The search eval arm's per-step read (empty on every other path).
    search: SearchOutput = field(default_factory=SearchOutput)

    def without_history_carry(self) -> "PlayerActorOutput":
        return self.replace(history_carry=HistoryCarry())

    # The transition leaves whose FIRST axis is the unroll offset (K
    # transitions or K+1 nodes) ahead of the trajectory axis T: the
    # learner's batch vmap must place B after T on these, at axis 2.
    OFFSET_LEADING_LEAVES = (
        "transition_prior_logits",
        "transition_post_logits",
        "transition_post_one_hot",
        "transition_value_head",
        "transition_kind_logits",
        "transition_done_logit",
        "transition_terminal_logits",
        "transition_action_one_hot",
        "transition_generator_logits",
        "transition_teacher_codes",
        "transition_generator_target",
        "transition_support_mask",
        "transition_node_overflow",
        "transition_align_logits",
        "transition_align_target",
    )

    @classmethod
    def batch_out_axes(cls) -> "PlayerActorOutput":
        """`out_axes` for a vmap of the player forward over the batch: B at
        axis 1 (after T) on every leaf, at axis 2 on the offset-leading
        transition leaves (after K and T). ONE definition, so a leaf added
        with a leading offset axis is registered here and nowhere else."""
        axes = {leaf.name: 1 for leaf in fields(cls)}
        for name in cls.OFFSET_LEADING_LEAVES:
            axes[name] = 2
        return cls(**axes)

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
    # Replay feedback identity, (1,) per chunk / (1, B) in the learner.
    # IDs never repeat within a store, including across clear().
    replay_slot: ArrayLike = ()
    replay_id: ArrayLike = ()

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
