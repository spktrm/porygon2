"""The jitted learner update.

One function by design: it is jitted with static_argnames=["config"] and
donates the train states, so the whole update has to sit inside a single
traced call. Helpers it calls (targets, losses, telemetry) live beside it.
"""

import logging

import jax
import jax.numpy as jnp
import optax

from rl.environment.data import (
    CAT_VF_SUPPORT,
    PackedSetFeature,
)
from rl.environment.interfaces import Batch, BuilderActorInput, PlayerActorInput
from rl.environment.protos.features_pb2 import (
    EntityPublicNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.model.constants import (
    DYNAMICS_GROUP_SLICES,
    LEARNER_ONLY_GROUPS,
    POLICY_READABLE_ROWS,
    SEQUENCE_GROUP_IDS,
    SequenceGroup,
)
from rl.model.heads import HeadParams
from rl.model.history_encoder import major_arg_step_mask
from rl.model.player_model import dynamics_alignment
from rl.model.state_features import REVEALED_ID_COLUMNS, hp_input_rows
from rl.model.transition import unimix_probs
from rl.model.utils import Params
from rl.online.artifact import Porygon2BuilderTrainState, Porygon2PlayerTrainState
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.loss import (
    backward_kl_loss,
    clip_fraction,
    factorised_entropies,
    forward_kl_loss,
    mse_value_loss,
    policy_gradient_loss,
    uniform_kl_modalities,
)
from rl.online.training.targets import (
    compute_builder_targets,
    compute_player_targets,
    reference_kl,
)
from rl.online.training.telemetry import (
    action_axis_masks,
    belief_accuracy_logs,
    calculate_r2,
    code_usage_logs,
    collect_batch_telemetry_data,
    critic_outcome_telemetry,
    head_param_telemetry,
    promote_map,
)
from rl.utils import average

logger = logging.getLogger(__name__)


# Floor on a group's per-batch normaliser (mean squared delta, row units).
# The field rows move on few steps, so on a small batch their mean squared
# delta is often ~0.03 and sometimes EXACTLY 0 (nothing on the field
# changed in any chunk); under a 1e-6 eps that batch scored loss ~400 and a
# 2.5e4 head gradient norm, which the global clip (10) turned into a wasted
# step for every other loss (irqeetfg @633-637k, ~1 logged step in 5). The
# floor is ~1/50 of the public scale (0.53) and under the field scale on
# every normal batch read, so it only binds where the ratio is degenerate;
# the gradient amplification of the normaliser is bounded at 1/floor.
DYNAMICS_SCALE_FLOOR = 1e-2


def dynamics_hp_basis(params: Params) -> jax.Array:
    """(D, r): an orthonormal basis of the subspace the public row's hp
    tokens write into -- QR of the EMA `public_persistent_linear` kernel's
    hp input rows. The instrument for the delta loss's known shortcoming:
    the target's per-feature scale is learnable (the state linears train
    under every loss and the EMA copies them), so a normalised MSE can
    shrink the unpredictable directions' share of the normaliser instead
    of predicting them. `player_dynamics_hp_share` reads the public delta's
    variance in this subspace; falling while the public gain rises is the
    gaming shape."""
    kernel = params["params"]["encoder"]["public_persistent_linear"]["kernel"]
    rows = kernel[hp_input_rows("public_persistent_linear")].astype(jnp.float32)
    basis, _ = jnp.linalg.qr(rows.T)
    return jax.lax.stop_gradient(basis)


# The transition-split instruments (2026-09-05, stochastic-transition plan
# step 1). A learner row is a REQUEST, and the edges the sim runs between
# request t and t+1 -- the opponent's decisions, the rolls, the reveals --
# are the unobserved branches the delta head's mean sits between. The
# service stamps every edge with the count of the request it PRECEDES
# (`_preprocessEdge` reads `player.requestCount` after the choose-time
# increment, `runner.ts`), so the steps a transition spans are exactly the
# window steps whose REQUEST_COUNT equals the info count at t+1.
TRANSITION_SHORT_EDGES = 2
TRANSITION_LONG_EDGES = 4


def transition_edges(env_output, history_field: jax.Array) -> jax.Array:
    """(T-1, B) int32: how many valid history-window steps transition
    t -> t+1 spans. 0 where the window no longer holds them."""
    request_counts = env_output.info[..., InfoFeature.INFO_FEATURE__REQUEST_COUNT]
    field_valid = history_field[..., FieldFeature.FIELD_FEATURE__VALID] > 0
    field_requests = history_field[..., FieldFeature.FIELD_FEATURE__REQUEST_COUNT]
    spanned = field_valid[None] & (field_requests[None] == request_counts[1:, None])
    return spanned.sum(axis=1).astype(jnp.int32)


def transition_reveals(
    env_output, matched: jax.Array, next_index: jax.Array
) -> jax.Array:
    """(T-1, B) bool: some OPPONENT public row, matched across the step,
    changed an identity token -- a reveal (or an Illusion rewrite), the
    hidden-information branch of the transition."""
    public = DYNAMICS_GROUP_SLICES["public"]
    ids = env_output.revealed_team[..., REVEALED_ID_COLUMNS]
    ids_next = jnp.take_along_axis(ids[1:], next_index[..., public, None], axis=2)
    changed = (ids[:-1] != ids_next).any(-1)
    theirs = (
        env_output.public_team[
            :-1, ..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ]
        == 0
    )
    return (changed & matched[..., public] & theirs).any(-1)


def masked_percentile(values: jax.Array, mask: jax.Array, fraction: float) -> jax.Array:
    """The `fraction` quantile of `values` where `mask` (nearest rank), by one sort (no
    data-derived shapes); -1 when nothing is masked in."""
    flat = jnp.sort(jnp.where(mask, values, -1).ravel())
    count = mask.sum()
    index = flat.shape[0] - count + jnp.round(fraction * (count - 1)).astype(jnp.int32)
    return flat[jnp.clip(index, 0, flat.shape[0] - 1)]


def dynamics_losses(
    pred: jax.Array,
    pred_prior: jax.Array,
    target: jax.Array,
    env_output,
    acted_mask: jax.Array,
    value_mask: jax.Array,
    hp_basis: jax.Array | None = None,
    history_field: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array], dict[str, jax.Array]]:
    """The transition model's GROUNDING loss and the transition-split
    instruments (2026-09-04 delta head; 2026-09-05 relabelled).

    `pred` (T, B, R, D) is the grounding head's read of the imagined next
    sequence at step t -- row j of it is the NEXT step's target row j,
    the imagined rows keep the next step's layout -- and `target` is the
    pre-trunk content of the target rows (`dynamics_target`, computed
    outside the loss fn so it carries no gradient). Row j at t is matched
    to its next-step row through `dynamics_alignment` (public rows re-sort
    every step; private rows follow the request order) and the prediction
    is gathered to it, so the label is `target_next` and the error is
    normalised by the COPY predictor's, `|target_next - target_now|^2`:
    the same static-token cancellation the delta form had, on the same
    scale, with the copy scoring exactly 1 per group. A row counts when it
    is matched, an action was taken at t (`acted_mask`, which drops the
    done row) and t+1 is a real state (`value_mask` -- the bootstrap-only
    final row is a valid TARGET). Forced rows stay in: the transition is
    real even when the choice was not.

    Per group g of DYNAMICS_GROUP_SLICES, f32:

        num_g   = average(|pred_next - target_next|^2, mask_g)
        scale_g = sg(average(|target_next - target_now|^2, mask_g))
        loss_g  = num_g / max(scale_g, floor)

    mean over groups, so the field rows' small movements neither vanish
    under the public scale nor dominate it; `gain_g = 1 - loss_g` is the
    R^2 of the change (0 = copy, negative = worse than copy). An
    all-masked group averages to 0 on both sides and contributes 0; a
    group whose rows did not move on the batch is normalised at
    DYNAMICS_SCALE_FLOOR instead of dividing by ~0.

    `pred_prior` is the same read of a no-gradient decode from the prior
    MODE -- the rollout-side number. It is EXPECTED BELOW the posterior
    gain (a sample from a two-branch law is further from the truth in MSE
    than the mean the old head fitted); the panel exists so nobody reads
    it as a regression.

    `..._gain_hp_moved` is the public gain on the rows whose wire HP_RATIO
    changed across the step, scaled on that subset: the counters (turn,
    toxic, sleep) inflate R^2 on every row, and this subset is where the
    branching lives. With `history_field` the public gain is also split by
    what the transition spans (`transition_edges`: short <= 2, long >= 4
    window steps) and by whether an opponent row revealed a token
    (`transition_reveals`); the splits are returned for the KL panels.
    """
    matched, next_index = jax.vmap(jax.vmap(dynamics_alignment))(
        jax.tree.map(lambda leaf: leaf[:-1], env_output),
        jax.tree.map(lambda leaf: leaf[1:], env_output),
    )
    target = target.astype(jnp.float32)

    def gather_next(rows):
        return jnp.take_along_axis(
            rows.astype(jnp.float32), next_index[..., None], axis=2
        )

    target_next = gather_next(target[1:])
    delta = target_next - target[:-1]
    valid_step = acted_mask[:-1] & value_mask[1:]
    mask = matched & valid_step[..., None]
    sq_err = jnp.square(gather_next(pred[:-1]) - target_next).sum(-1)
    sq_err_prior = jnp.square(gather_next(pred_prior[:-1]) - target_next).sum(-1)
    sq_delta = jnp.square(delta).sum(-1)

    def normalised(rows_mask, err=sq_err):
        num = average(err, rows_mask)
        scale = jax.lax.stop_gradient(average(sq_delta, rows_mask))
        return num / jnp.maximum(scale, DYNAMICS_SCALE_FLOOR), scale

    logs = dict(
        player_transition_ground_rows_frac=average(
            mask.astype(jnp.float32).mean(-1), valid_step
        ),
    )
    group_losses = []
    for group, rows in DYNAMICS_GROUP_SLICES.items():
        group_mask = jnp.zeros_like(mask).at[..., rows].set(mask[..., rows])
        group_loss, group_scale = normalised(group_mask)
        group_losses.append(group_loss)
        logs[f"player_transition_gain_{group}"] = 1.0 - group_loss
        logs[f"player_transition_ground_scale_{group}"] = group_scale
    loss = jnp.mean(jnp.stack(group_losses))
    logs["player_loss_transition_ground"] = loss

    public = DYNAMICS_GROUP_SLICES["public"]
    public_mask = jnp.zeros_like(mask).at[..., public].set(mask[..., public])
    prior_loss, _ = normalised(public_mask, sq_err_prior)
    logs["player_transition_gain_public_prior"] = 1.0 - prior_loss
    hp_ratio = env_output.public_team[
        ..., EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
    ]
    hp_next = jnp.take_along_axis(hp_ratio[1:], next_index[..., public], axis=2)
    hp_moved = jnp.zeros_like(mask).at[..., public].set(hp_ratio[:-1] != hp_next)
    moved_loss, _ = normalised(public_mask & hp_moved)
    logs["player_transition_gain_hp_moved"] = 1.0 - moved_loss
    moved_prior_loss, _ = normalised(public_mask & hp_moved, sq_err_prior)
    logs["player_transition_gain_hp_moved_prior"] = 1.0 - moved_prior_loss
    logs["player_transition_hp_moved_frac"] = average(
        hp_moved[..., public].astype(jnp.float32).mean(-1), valid_step
    )
    if hp_basis is not None:
        hp_energy = jnp.square(delta @ hp_basis).sum(-1)
        logs["player_transition_hp_share"] = average(hp_energy, public_mask) / (
            jnp.maximum(average(sq_delta, public_mask), DYNAMICS_SCALE_FLOOR)
        )
    splits = {}
    if history_field is not None:
        edges = transition_edges(env_output, history_field)
        logs["player_transition_edges_mean"] = average(
            edges.astype(jnp.float32), valid_step
        )
        logs["player_transition_edges_p90"] = masked_percentile(
            edges, valid_step, 0.9
        ).astype(jnp.float32)
        reveal = transition_reveals(env_output, matched, next_index)
        logs["player_transition_reveal_frac"] = average(
            reveal.astype(jnp.float32), valid_step
        )
        splits = dict(
            short=edges <= TRANSITION_SHORT_EDGES,
            long=edges >= TRANSITION_LONG_EDGES,
            reveal=reveal,
            no_reveal=~reveal,
        )
        for name, rows in splits.items():
            split_loss, _ = normalised(public_mask & rows[..., None])
            logs[f"player_transition_gain_public_{name}"] = 1.0 - split_loss
    return loss, logs, splits


def _code_perplexity(probs: jax.Array, mask: jax.Array, prefix: str) -> dict:
    """Usage perplexity of a (T-1, B, G, K) code over the masked
    transitions: exp(H) of each group's batch marginal, mean and min over
    groups (min -> 1 is a dead group)."""
    weights = mask.astype(jnp.float32)[..., None, None]
    marginal = (probs * weights).sum(axis=(0, 1)) / jnp.maximum(weights.sum(), 1.0)
    marginal = marginal / jnp.maximum(marginal.sum(-1, keepdims=True), 1e-8)
    entropy = -(marginal * jnp.log(jnp.maximum(marginal, 1e-8))).sum(-1)
    perplexity = jnp.exp(entropy)
    return {
        f"{prefix}_perplexity_mean": perplexity.mean(),
        f"{prefix}_perplexity_min": perplexity.min(),
    }


def transition_losses(
    pred,
    env_output,
    acted_mask: jax.Array,
    value_mask: jax.Array,
    policy_mask: jax.Array,
    flat_action_mask: jax.Array,
    win_returns: jax.Array,
    v_target: jax.Array,
    target_log_policy: jax.Array,
    cat_vf_support: jax.Array,
    splits: dict[str, jax.Array],
    config: Porygon2LearnerConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Every transition-model loss except grounding (2026-09-05,
    rl/model/transition.py), each on an OBSERVED label at t+1, over the
    transitions with an action taken at t and a real state at t+1:

    - consistency: per sequence group, the imagined rows' squared error
      against the real next post-trunk rows normalised by the copy
      predictor's (`|h_{t+1} - h_t|^2`, per-group scale, the grounding
      form) -- the copy scores 1, and the zero-init `out_proj` starts
      there;
    - the KL halves (DreamerV3): prior <- sg(posterior) at dyn_coef,
      posterior <- sg(prior) at rep_coef, each clipped below at free_nats
      per transition (summed over groups);
    - value: the shared critic on the imagined CLS row, CE to the t+1
      win_returns; `value_r2` beside `player_value_head_r2` and
      `value_gap` (|V(imagined) - V_target(real t+1)|, support units --
      the step-1 probe's number, 0.031 for the mean head);
    - policy: the shared readout on the imagined rows, forward KL from
      the sg'd target policy at t+1 over the real next legal set;
    - the next action mask (per-cell BCE) with its request kind (CE) and
      done (BCE).

    `player_transition_kl` is the unclipped prior<-posterior KL per
    transition, THE chance-node number, split by spanned edges and by
    reveals: a code that carries the unobserved branches reads higher
    where more happened.
    """
    valid_step = acted_mask[:-1] & value_mask[1:]
    logs = {}

    err = pred.transition_cons_err[:-1].astype(jnp.float32)
    scale = pred.transition_cons_scale[:-1].astype(jnp.float32)
    group_ids = SEQUENCE_GROUP_IDS[POLICY_READABLE_ROWS]
    group_losses = []
    for group in SequenceGroup:
        if group in LEARNER_ONLY_GROUPS:
            continue
        group_mask = valid_step[..., None] & jnp.asarray(group_ids == group)
        num = average(err, group_mask)
        group_scale = jax.lax.stop_gradient(average(scale, group_mask))
        group_loss = num / jnp.maximum(group_scale, DYNAMICS_SCALE_FLOOR)
        group_losses.append(group_loss)
        logs[f"player_transition_cons_gain_{group.name.lower()}"] = 1.0 - group_loss
    loss_cons = jnp.mean(jnp.stack(group_losses))

    prior_logits = pred.transition_prior_logits[:-1]
    post_logits = pred.transition_post_logits[:-1]
    if prior_logits.shape[-2] > 0:
        prior = unimix_probs(prior_logits)
        post = unimix_probs(post_logits)

        def kl(from_probs, to_probs):
            return (from_probs * (jnp.log(from_probs) - jnp.log(to_probs))).sum(
                axis=(-2, -1)
            )

        kl_dyn = kl(jax.lax.stop_gradient(post), prior)
        kl_rep = kl(post, jax.lax.stop_gradient(prior))
        free = config.player_transition_free_nats
        loss_kl = config.player_transition_dyn_coef * average(
            jnp.maximum(kl_dyn, free), valid_step
        ) + config.player_transition_rep_coef * average(
            jnp.maximum(kl_rep, free), valid_step
        )
        kl_value = jax.lax.stop_gradient(kl_dyn)
        logs["player_transition_kl"] = average(kl_value, valid_step)
        logs["player_transition_kl_free_frac"] = average(
            (kl_value < free).astype(jnp.float32), valid_step
        )
        for name, rows in splits.items():
            logs[f"player_transition_kl_{name}"] = average(kl_value, valid_step & rows)
        logs.update(_code_perplexity(post, valid_step, "player_transition_post"))
        logs.update(_code_perplexity(prior, valid_step, "player_transition_prior"))
        logs["player_transition_prior_post_agree"] = average(
            (prior.argmax(-1) == post.argmax(-1)).all(-1).astype(jnp.float32),
            valid_step,
        )
    else:
        loss_kl = jnp.zeros((), jnp.float32)

    value_head = pred.transition_value_head
    next_returns = win_returns[1:].astype(jnp.float32)
    loss_value = average(
        optax.softmax_cross_entropy(
            logits=value_head.logits[:-1].astype(jnp.float32), labels=next_returns
        ),
        valid_step,
    )
    expectation = value_head.expectation[:-1].astype(jnp.float32)
    logs["player_transition_value_r2"] = calculate_r2(
        value_prediction=expectation,
        value_target=next_returns @ cat_vf_support.astype(jnp.float32),
        mask=valid_step,
    )
    logs["player_transition_value_gap"] = average(
        jnp.abs(expectation - v_target[1:].astype(jnp.float32)), valid_step
    )

    next_mask = flat_action_mask[1:]
    target_policy = masked_policy(target_log_policy[1:], next_mask)
    target_log = jnp.where(next_mask, target_log_policy[1:].astype(jnp.float32), 0.0)
    imagined_log = pred.transition_log_policy[:-1].astype(jnp.float32)
    policy_rows = acted_mask[:-1] & policy_mask[1:]
    loss_policy = average(
        (target_policy * (target_log - imagined_log)).sum(-1), policy_rows
    )

    mask_logits = pred.transition_mask_logits[:-1].astype(jnp.float32)
    mask_labels = next_mask.astype(jnp.float32)
    loss_mask = average(
        optax.sigmoid_binary_cross_entropy(mask_logits, mask_labels).mean(-1),
        valid_step,
    )
    mask_hit = (mask_logits > 0) == next_mask
    logs["player_transition_mask_acc"] = average(mask_hit.mean(-1), valid_step)
    logs["player_transition_mask_exact_frac"] = average(
        mask_hit.all(-1).astype(jnp.float32), valid_step
    )
    # Of the legal cells, the share predicted legal: the number "all off"
    # cannot score on (legal cells are ~3% of the block space).
    logs["player_transition_mask_recall"] = average(
        (mask_hit & next_mask).sum(-1) / jnp.maximum(next_mask.sum(-1), 1),
        valid_step,
    )

    kind_labels = env_output.info[1:, ..., InfoFeature.INFO_FEATURE__REQUEST_TYPE]
    kind_logits = pred.transition_kind_logits[:-1].astype(jnp.float32)
    loss_kind = average(
        optax.softmax_cross_entropy_with_integer_labels(kind_logits, kind_labels),
        valid_step,
    )
    logs["player_transition_kind_acc"] = average(
        (kind_logits.argmax(-1) == kind_labels).astype(jnp.float32), valid_step
    )

    done_labels = env_output.done[1:].astype(jnp.float32)
    done_logit = pred.transition_done_logit[:-1].astype(jnp.float32)
    loss_done = average(
        optax.sigmoid_binary_cross_entropy(done_logit, done_labels), valid_step
    )
    logs["player_transition_done_acc"] = average(
        ((done_logit > 0) == (done_labels > 0)).astype(jnp.float32), valid_step
    )
    logs["player_transition_done_frac"] = average(done_labels, valid_step)

    loss = (
        loss_cons
        + loss_kl
        + loss_value
        + loss_policy
        + loss_mask
        + loss_kind
        + loss_done
    )
    logs.update(
        player_loss_transition_cons=loss_cons,
        player_loss_transition_kl=loss_kl,
        player_loss_transition_value=loss_value,
        player_loss_transition_policy=loss_policy,
        player_loss_transition_mask=loss_mask,
        player_loss_transition_kind=loss_kind,
        player_loss_transition_done=loss_done,
        player_transition_rows_frac=average(
            valid_step.astype(jnp.float32), jnp.ones_like(valid_step)
        ),
    )
    return loss, logs


def masked_policy(log_policy: jax.Array, legal_mask: jax.Array) -> jax.Array:
    """f32 probabilities over legal cells: exp(log_policy), illegal cells
    zeroed, renormalised so the legal mass sums to 1."""
    policy = jnp.exp(log_policy.astype(jnp.float32)) * legal_mask
    return policy / jnp.maximum(policy.sum(axis=-1, keepdims=True), 1e-8)


def train_step(
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    batch: Batch,
    config: Porygon2LearnerConfig,
):
    """Train for a single step.

    Every loss coefficient is a static config field. There used to be a
    RuntimeScalars pytree carrying host-varied coefficients as TRACED leaves
    so a host-side controller could vary them without recompiling — the
    hazard being that config is a jit static_argname, and static scalars
    retained ~5GB of executables per distinct value and OOM-killed run
    1326. Nothing varies them any more (the ramps, the exploiter zeroing
    and every controller that actuated them are gone), so they moved back
    into config. Reintroducing ANY host-varied coefficient — the magnet PI
    controller is the documented candidate — means reintroducing a traced
    pytree for it; never widen the static config with a value that changes
    during a run.
    """
    player_transitions = batch.player_transitions
    player_history = batch.player_history
    player_packed_history = batch.player_packed_history
    builder_transitions = batch.builder_transitions
    builder_history = batch.builder_history

    player_actor_input = PlayerActorInput(
        env=player_transitions.env_output,
        packed_history=player_packed_history,
        history=player_history,
    )

    player_target_pred = player_state.apply_fn(
        player_state.target_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    )

    # NashPG reference policy: the FROZEN reg_params' full-support
    # log-policy on the same batch (one extra forward; stop-gradient by
    # construction — reg_params are not the differentiated leaf).
    reg_log_policy = player_state.apply_fn(
        player_state.reg_params,
        player_actor_input,
        player_transitions.agent_output.actor_output,
        HeadParams(),
    ).action_head.log_policy

    player_actor_action_head = player_transitions.agent_output.actor_output.action_head
    player_actor_log_prob = player_actor_action_head.log_prob
    player_target_log_prob = player_target_pred.action_head.log_prob

    float_dtype = player_actor_log_prob.dtype

    cat_vf_support = jnp.asarray(CAT_VF_SUPPORT, dtype=float_dtype)

    player_valid = jnp.bitwise_not(player_transitions.env_output.done)

    training_logs = {}

    # Directed-message sanity (2026-09-01): fraction of valid history steps
    # carrying an identified SOURCE row (a real major arg). Expect >> 0.5;
    # a collapse here says the src identification broke, not the game.
    history_field = player_history.field
    step_is_valid = history_field[..., FieldFeature.FIELD_FEATURE__VALID] > 0
    step_has_src = jax.vmap(major_arg_step_mask, in_axes=(1, 1), out_axes=1)(
        history_field, player_packed_history.edge_cache
    )
    training_logs["player_history_src_frac"] = jnp.where(
        step_is_valid.sum() > 0,
        (step_has_src & step_is_valid).sum() / step_is_valid.sum().clip(min=1),
        0.0,
    )

    target_actor_log_ratio = player_target_log_prob - player_actor_log_prob
    target_actor_ratio = jnp.exp(target_actor_log_ratio)
    # mu/pi_target clipped at 2, telemetry only (player_impact_clip_frac):
    # the IMPACT surrogate it once recentred is gone; the panel still
    # reads how far behaviour has drifted from the fast target.
    actor_target_clipped_ratio = jnp.exp(-target_actor_log_ratio).clip(min=0.0, max=2.0)

    # IMPACT-style targets: the fast target network supplies the Retrace
    # reference policy and value/kl bootstraps. Under
    # player_privileged_targets the bootstraps -- and therefore
    # pg_advantages -- come from the PRIVILEGED head (asymmetric
    # actor-critic, 2026-09-01); False is bit-for-bit the old estimator.
    if config.player_privileged_targets:
        target_value_log_probs = player_target_pred.priv_value_head.log_probs
    else:
        target_value_log_probs = player_target_pred.value_head.log_probs
    player_targets, channel_logs = compute_player_targets(
        batch,
        value_log_probs=target_value_log_probs,
        isr=target_actor_ratio,
        config=config,
    )
    training_logs.update(channel_logs)
    policy_mask = player_targets.policy_mask
    value_mask = player_targets.value_mask
    # NashPG reference SNAP: reg_params <- target_params every
    # player_reg_snap_steps — their outer-loop rho reset, in place,
    # still three param sets, FROZEN between snaps. (The continuous EMA
    # this replaced never reset, so the KL gap compounded with policy
    # speed: 2wvnlsz3 hit ref_kl 2.07 nats by 98k.) Step 0 snaps
    # trivially (reg = target = init), and a resume at a snap multiple
    # snaps on its first step, repairing any accumulated gap at restart.
    reg_snap = player_state.step_count % config.player_reg_snap_steps == 0
    training_logs["player_reg_snapped"] = reg_snap.astype(jnp.float32)

    # Fraction of steps where the IMPACT clipped-target correction is
    # saturated at its cap — a second staleness signal alongside the actor
    # KL and ESS diagnostics.
    training_logs["player_impact_clip_frac"] = (
        (actor_target_clipped_ratio >= 2.0).astype(jnp.float32).mean(where=policy_mask)
    )

    # Fresh-vs-replayed value error: the memorisation gap. A network with
    # healthy plasticity fits fresh and replayed trajectories about equally;
    # replayed error falling while fresh error rises means the buffer is
    # being memorised — the plasticity signature that should gate any
    # shrink-and-perturb, unlike league stagnation which has many causes.
    # Per-group means are NaN in batches with no fresh (or no replayed)
    # member; wandb line plots skip them.
    if not isinstance(batch.reuse_count, tuple):
        target_value = player_target_pred.value_head.expectation
        return_value = player_targets.win_returns @ cat_vf_support
        value_sq_err = jnp.square(target_value - return_value)
        vm = value_mask.astype(value_sq_err.dtype)
        per_traj_err = (value_sq_err * vm).sum(axis=0) / (vm.sum(axis=0) + 1e-8)
        fresh = batch.reuse_count[0] == 0
        replayed = ~fresh
        fresh_err = per_traj_err.mean(where=fresh)
        replay_err = per_traj_err.mean(where=replayed)
        training_logs.update(
            {
                "plasticity_fresh_value_err": fresh_err,
                "plasticity_replay_value_err": replay_err,
                "plasticity_value_err_reuse_gap": fresh_err - replay_err,
            }
        )
    # Already flat: the env mask IS the block-cell vector since 2026-08-31.
    flat_action_mask = player_transitions.env_output.action_mask

    # NashPG advantage: the plain v-trace pass from targets.py, batch-
    # normalised over the surrogate's own rows with masked mean/std (the
    # reference's update_agent does exactly this per minibatch). f32
    # before promote_map bf16-casts the rest of the targets. The count
    # guard covers an all-masked batch; there is no running statistic
    # here to poison (LESSONS 2 applies to EMAs, not batch stats).
    pg_advantages = player_targets.pg_advantages.astype(jnp.float32)
    pg_adv_count = jnp.maximum(policy_mask.sum().astype(jnp.float32), 1.0)
    pg_adv_mean = jnp.where(policy_mask, pg_advantages, 0.0).sum() / pg_adv_count
    pg_adv_var = (
        jnp.where(policy_mask, jnp.square(pg_advantages - pg_adv_mean), 0.0).sum()
        / pg_adv_count
    )
    pg_adv_std = jnp.sqrt(pg_adv_var)
    pg_adv_norm = jnp.where(
        policy_mask, (pg_advantages - pg_adv_mean) / (pg_adv_std + 1e-8), 0.0
    )
    training_logs["player_pg_adv_mean"] = pg_adv_mean
    training_logs["player_pg_adv_std"] = pg_adv_std

    player_targets = promote_map(player_targets, float_dtype)

    v_target = player_target_pred.value_head.expectation.astype(jnp.float32)
    # The critics learn the PLAIN game: NashPG carries its reference KL
    # in the POLICY objective and uses no reward transform
    # (arXiv:2510.18183), so no penalty stream enters the labels or
    # bootstraps.
    # An action was actually taken here — including on forced single-option
    # steps, which policy_mask excludes — but not on terminal rows.
    acted_mask = value_mask & jnp.logical_not(player_transitions.env_output.done)
    # One derivation of the switch/move predicates for the whole step —
    # the panels below, critic_outcome_telemetry and the policy-loss
    # telemetry all read THESE, so they cannot drift apart again
    # (telemetry.ActionAxisMasks).
    axis = action_axis_masks(flat_action_mask, player_actor_action_head.action_index)
    taken_switch = axis.taken_switch
    has_move = axis.has_move
    voluntary_switch_mask = acted_mask & taken_switch & has_move
    forced_switch_mask = acted_mask & taken_switch & jnp.logical_not(has_move)
    move_mask = acted_mask & jnp.logical_not(taken_switch)
    # Realised behaviour frequency on the axis every collapse formed in
    # (RENAMED off the player_q_* prefix 2026-08-30 with the last of the Q
    # machinery — same quantities, fresh wandb continuity by design).
    training_logs["player_taken_switch_frac"] = average(
        taken_switch.astype(jnp.float32), acted_mask
    )
    training_logs["player_taken_voluntary_switch_frac"] = average(
        (taken_switch & has_move).astype(jnp.float32), acted_mask
    )

    # Off-policy attenuation audit, split by the TAKEN modality. isr =
    # pi_target/mu_actor is what v-trace multiplies its TD errors by
    # (targets.py: rho_t = c_t = min(1, isr) — Retrace went 2026-08-23
    # and the alpha blend 2026-08-21). As pi(switch) decays, isr on
    # switch-taken rows falls
    # below 1 and those rows contribute proportionally less.
    #
    # NOTE this is the CORRECT importance correction, not a bug: the
    # readout is effective sample size on switch rows, not bias. A
    # widening gap between the two means the learner is hearing the
    # switch evidence ever more faintly as the collapse deepens, which
    # is a self-reinforcing loop even though every individual update is
    # properly weighted. below1_frac is the cleaner signal than the
    # mean (isr is heavy-tailed on the upside).
    isr_f32 = target_actor_ratio.astype(jnp.float32)
    training_logs["player_isr_switch_voluntary"] = average(
        isr_f32, voluntary_switch_mask
    )
    training_logs["player_isr_switch_forced"] = average(isr_f32, forced_switch_mask)
    training_logs["player_isr_move"] = average(isr_f32, move_mask)
    training_logs["player_isr_below1_switch_voluntary"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), voluntary_switch_mask
    )
    training_logs["player_isr_below1_move"] = average(
        (isr_f32 < 1.0).astype(jnp.float32), move_mask
    )

    if not isinstance(batch.game_outcome, tuple):
        # Step-1 panels (docs/critic-weakness-analysis.md): the realised-
        # outcome instruments, a property of the games, not of any critic.
        training_logs.update(
            critic_outcome_telemetry(
                game_outcome=batch.game_outcome,
                game_length=batch.game_length,
                game_step_offset=batch.game_step_offset,
                v_target=v_target,
                flat_action_mask=flat_action_mask,
                masks=axis,
                acted_mask=acted_mask,
                value_mask=value_mask,
            )
        )
    if not isinstance(batch.reuse_count, tuple):
        fresh_cols = batch.reuse_count[0] == 0
        ~fresh_cols
        vm_fresh = value_mask & fresh_cols[None, :]
        training_logs["player_value_r2_fresh"] = jnp.where(
            vm_fresh.any(),
            calculate_r2(
                value_prediction=player_target_pred.value_head.expectation.astype(
                    jnp.float32
                ),
                value_target=(player_targets.win_returns @ cat_vf_support).astype(
                    jnp.float32
                ),
                mask=vm_fresh,
            ),
            0.0,
        )

    def player_loss_fn(params: Params):

        learner_player_pred = player_state.apply_fn(
            params,
            player_actor_input,
            player_transitions.agent_output.actor_output,
            HeadParams(),
        )

        learner_value_head = learner_player_pred.value_head
        learner_action_head = learner_player_pred.action_head
        learner_log_prob = learner_action_head.log_prob

        learner_actor_log_ratio = learner_log_prob - player_actor_log_prob
        learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

        learner_target_log_ratio = learner_log_prob - player_target_log_prob
        learner_target_ratio = jnp.exp(learner_target_log_ratio)

        loss_v_win = average(
            optax.softmax_cross_entropy(
                logits=learner_value_head.logits.astype(jnp.float32),
                labels=player_targets.win_returns.astype(jnp.float32),
            ),
            value_mask,
        )
        # The privileged critic: SAME labels, SAME mask -- the deployable
        # head above stays trained unchanged as the matched control, and
        # the priv-vs-deploy R2 pair is the discriminator the 2026-08-25
        # falsification never had.
        learner_priv_value_head = learner_player_pred.priv_value_head
        loss_v_win_priv = average(
            optax.softmax_cross_entropy(
                logits=learner_priv_value_head.logits.astype(jnp.float32),
                labels=player_targets.win_returns.astype(jnp.float32),
            ),
            value_mask,
        )
        # Belief-state shaping: CE from each matched public row's belief
        # logits to the STOPPED hidden-token code (the code net trains
        # through the privileged value CE, never through its own
        # prediction; the label is its reading of the tokens the public
        # row does NOT show, so the CE cannot be paid by re-reading the
        # row -- 2026-09-05, after the revealed-row control caught the
        # full-sheet label: margin 0.20 -> 0.017). Mean over groups so the
        # scale is K-independent; masked to rows where the alignment
        # holds, the mon still has a hidden token, AND the step counts
        # for value.
        belief_labels = jax.lax.stop_gradient(
            learner_player_pred.hidden_code.astype(jnp.float32)
        )
        belief_ce = optax.softmax_cross_entropy(
            logits=learner_player_pred.belief_logits.astype(jnp.float32),
            labels=belief_labels,
        ).mean(axis=-1)
        belief_mask = (
            learner_player_pred.belief_matched
            & learner_player_pred.belief_hidden_any
            & value_mask[..., None]
        )
        loss_belief = average(belief_ce, belief_mask)
        belief_logs = belief_accuracy_logs(
            learner_player_pred.belief_logits, belief_labels, belief_mask
        )
        # The species-only matched control: the same CE on the same labels
        # and rows from a table keyed on the public row's species token.
        # `player_belief_gain_over_species` is what the belief head reads
        # from the public row BEYOND its species; ~0 says it is a lookup.
        species_ce = optax.softmax_cross_entropy(
            logits=learner_player_pred.species_belief_logits.astype(jnp.float32),
            labels=belief_labels,
        ).mean(axis=-1)
        loss_species_belief = average(species_ce, belief_mask)
        species_logs = belief_accuracy_logs(
            learner_player_pred.species_belief_logits,
            belief_labels,
            belief_mask,
            prefix="player_species_belief",
        )
        # The revealed-row matched control: the same CE again from an MLP
        # over the matched mon's own pre-trunk public row (stop-gradient),
        # so it scores everything that row says in isolation.
        # `player_belief_context_margin` = belief minus this: what the head
        # infers from CONTEXT (history, the other rows); ~0 says the head
        # only reads the mon's own revealed tokens.
        revealed_ce = optax.softmax_cross_entropy(
            logits=learner_player_pred.revealed_belief_logits.astype(jnp.float32),
            labels=belief_labels,
        ).mean(axis=-1)
        loss_revealed_belief = average(revealed_ce, belief_mask)
        revealed_logs = belief_accuracy_logs(
            learner_player_pred.revealed_belief_logits,
            belief_labels,
            belief_mask,
            prefix="player_revealed_belief",
        )

        # The latent transition model (rl/model/transition.py): grounding
        # against the pre-trunk label -- `player_target_pred`'s copy,
        # computed outside this fn, so it carries no gradient -- and every
        # other head against its observed t+1 label. The target and reg
        # forwards' own transition leaves are never read, so XLA drops
        # those two passes.
        loss_ground, ground_logs, transition_splits = dynamics_losses(
            learner_player_pred.transition_ground,
            learner_player_pred.transition_ground_prior,
            player_target_pred.dynamics_target,
            player_transitions.env_output,
            acted_mask,
            value_mask,
            hp_basis=dynamics_hp_basis(player_state.target_params),
            history_field=history_field,
        )
        loss_transition_rest, transition_logs = transition_losses(
            learner_player_pred,
            player_transitions.env_output,
            acted_mask,
            value_mask,
            policy_mask,
            flat_action_mask,
            player_targets.win_returns,
            v_target,
            player_target_pred.action_head.log_policy,
            cat_vf_support,
            transition_splits,
            config,
        )
        loss_transition = loss_ground + loss_transition_rest
        transition_logs["player_loss_transition"] = loss_transition

        action_head_entropy = average(learner_action_head.entropy, policy_mask)
        action_head_normalized_entropy = average(
            learner_action_head.normalized_entropy, policy_mask
        )

        loss_actor_forward_kl = forward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=policy_mask,
        )
        loss_actor_backward_kl = backward_kl_loss(
            policy_ratio=learner_actor_ratio,
            log_policy_ratio=learner_actor_log_ratio,
            valid=policy_mask,
        )
        loss_target_forward_kl = forward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=policy_mask,
        )
        loss_target_backward_kl = backward_kl_loss(
            policy_ratio=learner_target_ratio,
            log_policy_ratio=learner_target_log_ratio,
            valid=policy_mask,
        )

        normalized_modality_entropy = average(
            learner_action_head.normalized_modality_entropy, policy_mask
        )
        # Real-choice rows on the stay/switch axis: both a switch and a
        # a MOVE are legal. This is the slice the collapse forms in, and
        # every policy-modality readout below is scoped to it.
        #
        # STRICT since 2026-08-25. This previously required a switch and any
        # legal NON-switch, which also admitted WILDCARD / OTHER / TARGET
        # cells — so a row offering {switch, pass} counted as a stay/switch
        # decision here while the identically-described `has_both` in the Q
        # panels excluded it, and CLAUDE.md 3's rule reads the two families
        # against each other. A stay/switch decision only means something
        # when staying and attacking is actually on offer.
        switch_actions = axis.switch_cells
        switch_choice_mask = policy_mask & axis.has_both

        # JOINT surrogate: one pi/mu ratio on the taken action (the
        # 2026-08-27 per-level split is reverted — 2026-08-28 ledger).
        learner_log_policy = learner_action_head.log_policy
        pi_learner = masked_policy(learner_log_policy, flat_action_mask)

        macro_valid = policy_mask & (axis.num_legal_modalities >= 2)
        micro_valid = policy_mask & (axis.taken_modality_count >= 2)

        loss_pg = policy_gradient_loss(
            policy_ratios=learner_actor_ratio,
            advantages=pg_adv_norm,
            valid=policy_mask,
            threshold=config.player_ppo_clip,
            objective=config.player_pg_objective,
        )
        # Per-level entropies (macro = modality marginal, micro = within
        # the taken modality) are OBSERVERS only — the collapse instruments
        # the acceptance gates read. The regulariser itself is NashPG's:
        # the plain joint entropy bonus at the static player_ent_coef
        # (2026-08-30; the per-axis dual temperatures are removed).
        h_macro_rows, h_micro_rows = factorised_entropies(
            learner_log_policy, axis.taken_modality, flat_action_mask
        )
        entropy_macro = average(h_macro_rows, macro_valid)
        entropy_micro_taken = average(h_micro_rows, micro_valid)
        loss_entropy = -average(
            learner_action_head.entropy.astype(jnp.float32), policy_mask
        )
        # Magnet: full-distribution KL(pi || pi_reg) per row —
        # differentiated through the learner side (reg_log_policy comes
        # off the frozen reg_params, a constant).
        magnet_kl_rows = reference_kl(
            learner_log_policy, reg_log_policy, flat_action_mask
        )
        loss_mag = average(magnet_kl_rows, policy_mask)

        # The zero-avoiding term, on the MODALITY MARGINAL (2026-08-31):
        # forward KL from uniform over live modalities, modality-level
        # gradient exactly pi_m - 1/M. It is the only force in this bracket
        # that is not pi-prefactored, so it is the only one still acting on
        # a modality the policy has abandoned -- and unlike the row form it
        # replaced (the sp75c lesson) the loss is IDENTICALLY invariant to
        # within-modality redistribution, so it restores WHETHER-to-switch
        # mass without flattening WHICH-move-to-pick. The constant reference
        # still cannot be ratcheted flat or invert on the Q^pi sign.
        loss_modality_kl = average(
            uniform_kl_modalities(learner_log_policy, flat_action_mask),
            policy_mask,
        )

        # Modality decomposition of the two factors any taken-action
        # update is throttled by: pi mass and the observer critic's |A|,
        # over legal switch vs non-switch cells of real-choice rows.
        # Loss-agnostic, kept across the 2026-08-26 policy-loss transition:
        # prob_ratio falling with absadv_ratio ~ 1 is still the
        # starvation signature to watch, now with nothing but the magnet
        # cycle and entropy to oppose it.
        pg_row = switch_choice_mask[..., None]
        pg_switch_cells = flat_action_mask & switch_actions & pg_row
        pg_move_cells = flat_action_mask & jnp.logical_not(switch_actions) & pg_row

        def modality_ratio(numerator, denominator):
            return numerator / jnp.maximum(denominator, 1e-8)

        policy_prob_switch = average(pi_learner, pg_switch_cells)
        policy_prob_move = average(pi_learner, pg_move_cells)

        pg_logs = dict(
            player_loss_pg=loss_pg,
            player_loss_entropy=loss_entropy,
            player_entropy_macro=entropy_macro,
            player_entropy_micro_taken=entropy_micro_taken,
            player_ppo_clip_frac=clip_fraction(
                policy_ratios=learner_actor_ratio,
                valid=policy_mask,
                clip_ppo=config.player_ppo_clip,
            ),
            player_policy_prob_switch=policy_prob_switch,
            player_policy_prob_move=policy_prob_move,
            player_policy_prob_ratio=modality_ratio(
                policy_prob_switch, policy_prob_move
            ),
            # KL(pi_learner || pi_reg) per state — the magnet loss's own
            # value (name kept across the transition). Sawtooth: drifts
            # up against the FROZEN reference, drops to ~0 at each snap;
            # a level climbing ACROSS snaps is a policy running away
            # faster than the snap period can repair.
            player_ref_kl=loss_mag,
            player_loss_modality_kl=loss_modality_kl,
        )
        # pg bracket + v + kl.
        loss = (
            # pg: the NashPG bracket — surrogate + ent_coef * (-H) +
            # mag_coef * KL(pi || pi_reg), one coefficient scaling
            # improvement and regularisation together — plus the
            # zero-avoiding KL, which is the one deliberate divergence from
            # the reference actor loss (the reference carries no
            # mass-independent restorer in either mag_divergence mode).
            config.player_pg_coef
            * (
                loss_pg
                + config.player_ent_coef * loss_entropy
                + config.player_mag_coef * loss_mag
                + config.player_uniform_kl_coef * loss_modality_kl
            )
            # v: one critic, on the deploy-time information set. The
            # all-action advantage that sat beside it retired 2026-08-29 --
            # the policy stopped reading it at the NashPG switch, which left
            # it a matched control for an architecture that is now gone.
            + config.player_value_head_loss_coef * loss_v_win
            + config.player_priv_value_head_loss_coef * loss_v_win_priv
            + config.player_belief_coef * loss_belief
            + config.player_dynamics_coef * loss_transition
            # The species control, unscaled: its only param is the table
            # (an integer input has no gradient), and its gradient norm is
            # O(0.05) against a total of ~10, so the global clip is unmoved.
            + loss_species_belief
            # The revealed-row control, likewise unscaled: its input is under
            # stop_gradient, so only its own MLP receives the gradient.
            + loss_revealed_belief
            # kl: trust region against the behaviour policy — the
            # replay-staleness guard alongside the PPO clip.
            + config.player_kl_loss_coef * loss_actor_backward_kl
        )

        return loss, dict(
            **pg_logs,
            player_loss_v_win=loss_v_win,
            player_loss_v_win_priv=loss_v_win_priv,
            player_loss_belief=loss_belief,
            **belief_logs,
            player_loss_species_belief=loss_species_belief,
            **species_logs,
            player_belief_gain_over_species=belief_logs["player_belief_accuracy"]
            - species_logs["player_species_belief_accuracy"],
            player_loss_revealed_belief=loss_revealed_belief,
            **revealed_logs,
            player_belief_context_margin=belief_logs["player_belief_accuracy"]
            - revealed_logs["player_revealed_belief_accuracy"],
            player_belief_matched_frac=average(
                learner_player_pred.belief_matched.astype(jnp.float32).mean(-1),
                value_mask,
            ),
            # Of the matched mons, the share still carrying a hidden token
            # (the belief loss's population); 1 - this is fully-revealed.
            player_belief_hidden_frac=average(
                learner_player_pred.belief_hidden_any.astype(jnp.float32),
                learner_player_pred.belief_matched & value_mask[..., None],
            ),
            **ground_logs,
            **transition_logs,
            # Trunk over-smoothing (cosine up / participation down = rows
            # converging); the offline per-block twin is
            # rl/offline/trunk_homogeneity.py.
            player_trunk_row_cosine=average(
                learner_player_pred.trunk_row_cosine, value_mask
            ),
            player_trunk_row_participation=average(
                learner_player_pred.trunk_row_participation, value_mask
            ),
            # The history encoder's step GAT and write gate
            # (history_encoder.history_step_stats): per-trajectory scalars
            # broadcast over T, so this is the valid-step-weighted batch mean.
            player_history_step_attn_entropy=average(
                learner_player_pred.history_step_attn_entropy, value_mask
            ),
            player_history_step_attn_to_src=average(
                learner_player_pred.history_step_attn_to_src, value_mask
            ),
            player_history_step_attn_to_src_uniform=average(
                learner_player_pred.history_step_attn_to_src_uniform, value_mask
            ),
            player_history_gate_mean=average(
                learner_player_pred.history_gate_mean, value_mask
            ),
            player_loss_kl=loss_actor_backward_kl,
            # Per head entropies (diagnostics only — no longer regularized)
            player_action_entropy=action_head_entropy,
            player_action_normalized_entropy=action_head_normalized_entropy,
            player_normalized_modality_entropy=normalized_modality_entropy,
            player_learner_actor_ratio=average(learner_actor_ratio, policy_mask),
            player_learner_target_ratio=average(learner_target_ratio, policy_mask),
            player_learner_actor_forward_kl=loss_actor_forward_kl,
            # Modality-resolved split of the SAME k3 estimator. The
            # global mean is an expectation over the policy, so drift
            # confined to a modality carrying ~0.2 mass is diluted to
            # near nothing — which is why the replay reuse controller
            # (whose set-point is the global mean) cannot fire on a
            # collapsing switch modality even in principle. These two
            # de-average it by the taken modality.
            #
            # LIMITATION: this is still a mu-SAMPLED estimator over
            # taken actions, not a full-support KL, because actors do
            # not persist their full log_policy (player_model.py gates
            # log_policy on cfg.train). An exact full-support modality
            # KL against the actor would need that field stored.
            player_learner_actor_forward_kl_switch=forward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=policy_mask & taken_switch,
            ),
            player_learner_actor_forward_kl_move=forward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=policy_mask & jnp.logical_not(taken_switch),
            ),
            player_learner_actor_backward_kl=loss_actor_backward_kl,
            player_learner_target_forward_kl=loss_target_forward_kl,
            player_learner_target_backward_kl=loss_target_backward_kl,
            player_value_head_r2=calculate_r2(
                value_prediction=learner_value_head.expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            # THE discriminator for the privileged premise: this pair on one
            # panel. The 2026-08-25 rung read WORSE than the deployable head
            # and was deleted for it; priv < deploy sustained past 30k is
            # this pass's pre-registered abort.
            player_priv_value_head_r2=calculate_r2(
                value_prediction=learner_priv_value_head.expectation,
                value_target=player_targets.win_returns @ cat_vf_support,
                mask=value_mask,
            ),
            # Mean absolute priv-minus-deploy expectation gap: the "worth
            # 0.005 value units" number, re-measured live.
            player_priv_value_gap=average(
                jnp.abs(
                    learner_priv_value_head.expectation - learner_value_head.expectation
                ),
                value_mask,
            ),
            player_nll_sum=(
                batch.player_transitions.agent_output.actor_output.action_head.log_prob
                * policy_mask
            )
            .sum(axis=0)
            .mean(),
            **code_usage_logs(
                learner_player_pred.opp_code,
                batch.player_transitions.env_output.opp_private_team,
                value_mask,
            ),
            # The label's own usage over the rows the belief loss scores:
            # a hidden code pinned at perplexity 1 is a dead label, not a
            # solved belief.
            **code_usage_logs(
                learner_player_pred.hidden_code,
                batch.player_transitions.env_output.opp_private_team,
                value_mask,
                row_mask=belief_mask,
                prefix="player_hidden_code",
            ),
        )

    player_grad_fn = jax.value_and_grad(player_loss_fn, has_aux=True)
    (player_loss_val, player_logs), player_grads = player_grad_fn(player_state.params)

    prev_player_state = player_state
    player_state = player_state.apply_gradients(grads=player_grads)
    player_state = player_state.replace(
        step_count=player_state.step_count + 1,
        frame_count=player_state.frame_count + player_valid.sum(),
        # Hard-snapped to the target net every player_reg_snap_steps —
        # NashPG's outer-loop reference reset (see reg_snap above).
        reg_params=jax.tree.map(
            lambda t, r: jnp.where(reg_snap, t, r),
            player_state.target_params,
            player_state.reg_params,
        ),
        target_params=optax.incremental_update(
            player_state.params,
            player_state.target_params,
            config.player_ema_update_rate,
        ),
    )
    # A non-finite loss or gradient must never reach the params or the EMA
    # scalars: one poisoned update is permanent, and the next periodic save
    # then overwrites the last good checkpoint with it. Keep the previous
    # state wholesale and log the skip.
    player_update_finite = jnp.isfinite(player_loss_val) & jnp.isfinite(
        optax.global_norm(player_grads)
    )
    player_state = jax.tree.map(
        lambda new, old: jnp.where(player_update_finite, new, old),
        player_state,
        prev_player_state,
    )

    training_logs.update(player_logs)
    training_logs.update(
        dict(
            player_loss=player_loss_val,
            player_param_norm=optax.global_norm(player_state.params),
            player_gradient_norm=optax.global_norm(player_grads),
            # Q-head learning readouts: the three-scalar micro gate, the
            # drift-from-init of the zero-init out layers and the pointer
            # kernels, and per-subtree grad norms (pre-clip). A micro
            # kernel rms sitting at its lecun init (0.0625 at fan-in 256)
            # with a flat gate = the within-modality route never trained.
            **head_param_telemetry(prev_player_state.params, player_grads),
            player_win_returns_sum=average(
                player_targets.win_returns.sum(axis=-1), value_mask
            ),
            player_win_returns_min=jnp.min(
                jnp.where(value_mask[..., None], player_targets.win_returns, 1000.0)
            ),
            player_policy_mask_sum=policy_mask.sum(),
            player_value_mask_sum=value_mask.sum(),
            # Which lattice combo this variant was compiled for (static
            # per executable) — the retuning readout for
            # config.player_shape_lattice.
            player_shape_T=float(batch.player_transitions.env_output.done.shape[0]),
            player_shape_H=float(batch.player_history.field.shape[0]),
            player_policy_value_mask_ratio=policy_mask.sum()
            / (value_mask.sum() + 1e-8),
            player_update_skipped=1.0 - player_update_finite.astype(jnp.float32),
        )
    )
    training_logs.update(
        {
            f"player_{k}_gradient_norm": optax.global_norm(player_grads["params"][k])
            for k in player_grads["params"]
        }
    )
    training_logs.update(
        {
            f"player_{k}_gradient_norm": optax.global_norm(
                player_grads["params"]["encoder"][k]
            )
            for k in player_grads["params"]["encoder"]
            if any(substring in k for substring in ("decoder", "encoder"))
        }
    )

    # --- Builder ---
    if config.smogon_format != "randombattle":
        builder_actor_input = BuilderActorInput(
            env=builder_transitions.env_output,
            history=builder_history,
        )

        builder_target_pred = builder_state.apply_fn(
            builder_state.target_params,
            builder_actor_input,
            builder_transitions.agent_output.actor_output,
            HeadParams(),
        )

        builder_actor_action_head = (
            builder_transitions.agent_output.actor_output.action_head
        )

        # Calculate importance sampling ratios for off-policy correction.
        builder_actor_log_prob = builder_actor_action_head.log_prob
        builder_target_log_prob = builder_target_pred.action_head.log_prob
        builder_actor_target_log_ratio = (
            builder_actor_log_prob - builder_target_log_prob
        )
        builder_actor_target_ratio = jnp.exp(builder_actor_target_log_ratio)
        builder_target_actor_ratio = jnp.exp(-builder_actor_target_log_ratio)
        builder_actor_target_clipped_ratio = jnp.clip(
            builder_actor_target_ratio, min=0.0, max=2.0
        )

        builder_valid = jnp.bitwise_not(builder_transitions.env_output.done)
        # Chunked unrolls: compute_builder_targets reads the game outcome
        # off player win_reward[-1], which only the game's TERMINAL chunk
        # carries — a mid-game chunk's builder rows would grade the team
        # against a zero payoff. average()-based builder losses are
        # empty-mask-safe (0, not NaN) if a batch happens to hold no
        # terminal chunk.
        builder_valid = builder_valid & player_transitions.env_output.done.any(axis=0)[
            None, :
        ].astype(jnp.bool_)
        # Compute builder targets inside train_step (JAX/JIT compatible).
        builder_targets = compute_builder_targets(
            batch,
            builder_target_pred,
            builder_target_actor_ratio,
            lambda_=config.builder_lambda,
            entropy_normalising_constant=config.builder_entropy_prediction_normalising_constant,
        )
        builder_returns = promote_map(builder_targets, float_dtype)

        builder_advantages = (
            builder_targets.win_advantages
            + config.builder_entropy_advantage_scale * builder_targets.ent_advantages
        )

        builder_win_return_correction = builder_targets.win_returns.sum(
            axis=-1, keepdims=True
        )
        builder_win_returns = (
            builder_targets.win_returns / builder_win_return_correction
        )

        def builder_loss_fn(params: Params):

            pred = builder_state.apply_fn(
                params,
                builder_actor_input,
                builder_transitions.agent_output.actor_output,
                HeadParams(),
            )

            learner_value_head = pred.value_head
            learner_action_head = pred.action_head
            learner_conditional_entropy_head = pred.conditional_entropy_head
            learner_log_prob = learner_action_head.log_prob

            learner_actor_log_ratio = learner_log_prob - builder_actor_log_prob
            learner_actor_ratio = jnp.exp(learner_actor_log_ratio)

            learner_target_log_ratio = learner_log_prob - builder_target_log_prob
            learner_target_ratio = jnp.exp(learner_target_log_ratio)

            builder_policy_ratio = (
                builder_actor_target_clipped_ratio * learner_actor_ratio
            )

            # Calculate the losses. threshold is REQUIRED (keyword-only,
            # no default): omitting it was a latent TypeError on the first
            # builder train step, unreachable only because randombattle
            # skips this branch entirely.
            loss_pg = policy_gradient_loss(
                policy_ratios=builder_policy_ratio,
                advantages=builder_advantages,
                valid=builder_valid,
                threshold=config.builder_ppo_clip_threshold,
            )

            loss_v = average(
                optax.softmax_cross_entropy(
                    logits=learner_value_head.logits, labels=builder_win_returns
                ),
                builder_valid,
            )

            loss_builder_entropy = -average(learner_action_head.entropy, builder_valid)
            loss_builder_conditional_entropy = mse_value_loss(
                pred=learner_conditional_entropy_head.logits,
                target=builder_targets.ent_returns,
                valid=builder_valid,
            )

            loss_forward_kl = forward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=builder_valid,
            )
            loss_backward_kl = backward_kl_loss(
                policy_ratio=learner_actor_ratio,
                log_policy_ratio=learner_actor_log_ratio,
                valid=builder_valid,
            )

            human_valid_mask = (
                builder_transitions.env_output.curr_attribute
                != PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE
            ) & (
                builder_transitions.env_output.curr_attribute
                != PackedSetFeature.PACKED_SET_FEATURE__GENDER
            )

            loss_human = average(
                learner_action_head.magnet_kl, valid=builder_valid & human_valid_mask
            )

            loss = (
                config.builder_policy_loss_coef * loss_pg
                + config.builder_value_loss_coef * loss_v
                + config.builder_kl_loss_coef * loss_backward_kl
                + config.builder_human_loss_coef * loss_human
                + config.builder_conditional_entropy_loss_coef
                * loss_builder_conditional_entropy
                + config.builder_entropy_coef * loss_builder_entropy
            )

            return loss, dict(
                builder_loss_pg=loss_pg,
                builder_loss_v=loss_v,
                builder_loss_kl_rl=loss_backward_kl,
                builder_loss_entropy=loss_builder_entropy,
                builder_loss_conditional_entropy=loss_builder_conditional_entropy,
                builder_loss_human=loss_human,
                # Ratios
                builder_learner_actor_ratio=average(learner_actor_ratio, builder_valid),
                builder_learner_target_ratio=average(
                    learner_target_ratio, builder_valid
                ),
                # Approx KL values
                builder_learner_actor_approx_kl=loss_forward_kl,
                builder_learner_condtional_entropy_head_mean=average(
                    learner_conditional_entropy_head.logits, builder_valid
                ),
                builder_learner_condtional_entropy_head_std=jnp.std(
                    learner_conditional_entropy_head.logits, where=builder_valid
                ),
                # Extra stats
                builder_value_function_r2=calculate_r2(
                    value_prediction=learner_value_head.expectation,
                    value_target=builder_returns.win_returns @ cat_vf_support,
                    mask=builder_valid,
                ),
            )

        builder_grad_fn = jax.value_and_grad(builder_loss_fn, has_aux=True)
        (builder_loss_val, builder_logs), builder_grads = builder_grad_fn(
            builder_state.params
        )
        training_logs.update(builder_logs)
        training_logs.update(
            dict(
                builder_loss=builder_loss_val,
                builder_win_return_correction=average(
                    builder_win_return_correction.reshape(builder_valid.shape),
                    builder_valid,
                ),
                builder_nll_sum=(
                    batch.builder_transitions.agent_output.actor_output.action_head.log_prob
                    * builder_valid
                )
                .sum(axis=0)
                .mean(),
                builder_param_norm=optax.global_norm(builder_state.params),
                builder_gradient_norm=optax.global_norm(builder_grads),
                builder_norm_adv_mean=average(builder_advantages, builder_valid),
                builder_norm_adv_std=builder_advantages.std(where=builder_valid),
            )
        )
        prev_builder_state = builder_state
        builder_state = builder_state.apply_gradients(grads=builder_grads)
        builder_state = builder_state.replace(
            step_count=builder_state.step_count + 1,
            frame_count=builder_state.frame_count + builder_valid.sum(),
            target_params=optax.incremental_update(
                builder_state.params,
                builder_state.target_params,
                config.builder_ema_update_rate,
            ),
        )
        # Same non-finite gate as the player update above.
        builder_update_finite = jnp.isfinite(builder_loss_val) & jnp.isfinite(
            optax.global_norm(builder_grads)
        )
        builder_state = jax.tree.map(
            lambda new, old: jnp.where(builder_update_finite, new, old),
            builder_state,
            prev_builder_state,
        )
        training_logs["builder_update_skipped"] = 1.0 - builder_update_finite.astype(
            jnp.float32
        )

    training_logs.update(collect_batch_telemetry_data(batch, config))
    training_logs.update(
        dict(
            player_frame_count=player_state.frame_count,
            builder_frame_count=builder_state.frame_count,
            training_step=player_state.step_count,
        )
    )

    return player_state, builder_state, training_logs


# Module-level, not per-Learner: exactly ONE Learner exists for the life of
# the process, so this is only ever compiled once either way. Kept
# module-level so the compiled fn survives a Learner rebuild in tests and
# scripts rather than paying the compile again.
TRAIN_STEP_JIT = jax.jit(
    train_step,
    static_argnames=["config"],
    donate_argnames=["player_state", "builder_state"],
)
