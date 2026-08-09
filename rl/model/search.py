"""Stage A root search: 1-ply announced-state evaluation + QRE solve.

No world model: a candidate joint action (mine, opponent's) is rendered as
two OUTCOME-MASKED announced edges — the same three-integer rows
(MAJOR_ARG, MOVE_TOKEN, ENTITY_IDX) that mask_outcome_features leaves of a
real log edge — and the REAL history recurrence is advanced one hypothetical
step to the announced state, where the integrated history critic reads the
value. The k_me x k_opp value matrix is then solved as a uniform-anchored
QRE (mirror descent — the same regularised-equilibrium family as the magnet
KL) and the root action is sampled from the solved own-side policy.

Everything here is per-timestep pure math; the model-side orchestration
lives in Porygon2PlayerModel.act_search.
"""

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import ALLY_SWITCH_INDICES, MOVE_INDICES, NUM_ACTION_FEATURES
from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
from rl.environment.protos.features_pb2 import EntityEdgeFeature, MovesetFeature

# --- Static src-slot -> announcement tables (singles; ally 1 acts) -------
# my_moveset row r feeds action src slot MOVE_INDICES[r] (encoder builds
# the action tokens exactly this way), so the inverse map recovers the
# moveset row — and with it MOVE_ID and the actor's public slot — from a
# flat action index.
SRC_TO_MOVESET_ROW = np.full(NUM_ACTION_FEATURES, -1, dtype=np.int32)
for _row, _src in enumerate(MOVE_INDICES.reshape(-1)):
    SRC_TO_MOVESET_ROW[_src] = _row
SRC_IS_SWITCH = np.zeros(NUM_ACTION_FEATURES, dtype=bool)
SRC_IS_SWITCH[ALLY_SWITCH_INDICES.reshape(-1)] = True

MAJOR_MOVE = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE
MAJOR_SWITCH = BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__SWITCH


def announced_edge_row(
    num_edge_features: int,
    major_arg: jax.Array,
    move_token: jax.Array,
    entity_idx: jax.Array,
) -> jax.Array:
    """A hypothetical announced edge: identical to what
    mask_outcome_features leaves of a real edge — the three announcement
    integers, everything else zero. Exact train/search consistency follows
    because the critic's announced training inputs are masked the same
    way."""
    row = jnp.zeros((num_edge_features,), dtype=jnp.int32)
    row = row.at[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG].set(major_arg)
    row = row.at[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN].set(move_token)
    row = row.at[EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX].set(entity_idx)
    return row


def my_action_announcement(
    action_index: jax.Array, my_moveset: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(major, move_token, entity_idx) for one of MY flat action indices.

    Moves resolve through the src-slot -> moveset-row table; switches are
    announced as (SWITCH, no move, outgoing active slot) — the incoming
    mon's identity is deliberately omitted (v1: under-informative
    announcements are safe under the masking asymmetry rule; leaking is
    not)."""
    src = action_index // NUM_ACTION_FEATURES
    row = jnp.take(jnp.asarray(SRC_TO_MOVESET_ROW), src)
    is_switch = jnp.take(jnp.asarray(SRC_IS_SWITCH), src)
    safe_row = jnp.maximum(row, 0)
    move_token = my_moveset[safe_row, MovesetFeature.MOVESET_FEATURE__MOVE_ID]
    move_entity = my_moveset[safe_row, MovesetFeature.MOVESET_FEATURE__ENTITY_IDX]
    # Active slot: the acting entity of the first move row.
    active_entity = my_moveset[0, MovesetFeature.MOVESET_FEATURE__ENTITY_IDX]
    major = jnp.where(is_switch, MAJOR_SWITCH, MAJOR_MOVE)
    token = jnp.where(is_switch, 0, move_token)
    entity = jnp.where(is_switch, active_entity, move_entity)
    return major, token, entity


def opp_candidate_announcements(
    opp_moveset: jax.Array, num_moves: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(k_opp,) announcement triples for the opponent: their first
    ``num_moves`` moveset rows (unrevealed moves carry the UNK token the
    state already uses) plus one generic switch. No learned opponent model
    — uniform-anchored solving handles the weighting (interim until latent
    intents)."""
    rows = jnp.arange(num_moves)
    move_tokens = opp_moveset[rows, MovesetFeature.MOVESET_FEATURE__MOVE_ID]
    entities = opp_moveset[rows, MovesetFeature.MOVESET_FEATURE__ENTITY_IDX]
    active = opp_moveset[0, MovesetFeature.MOVESET_FEATURE__ENTITY_IDX]
    majors = jnp.concatenate(
        [jnp.full((num_moves,), MAJOR_MOVE), jnp.array([MAJOR_SWITCH])]
    )
    tokens = jnp.concatenate([move_tokens, jnp.array([0])])
    entity_idx = jnp.concatenate([entities, jnp.array([active])])
    return majors, tokens, entity_idx


def qre_solve(
    values: jax.Array, steps: int, temp: float
) -> tuple[jax.Array, jax.Array]:
    """Uniform-anchored QRE of the zero-sum matrix game ``values`` (my
    payoff, k_me x k_opp) via simultaneous mirror descent. Returns
    (p_me, p_opp). temp is the anchor temperature: higher = closer to
    uniform (mirrors the magnet KL's role in training)."""
    k_me, k_opp = values.shape
    log_p = jnp.zeros((k_me,))
    log_q = jnp.zeros((k_opp,))
    eta = 1.0 / max(temp, 1e-6)

    def body(carry, _):
        log_p, log_q = carry
        p = jax.nn.softmax(log_p)
        q = jax.nn.softmax(log_q)
        # Best-response gradients against the opponent's current mixture,
        # anchored to uniform by averaging in logit space (QRE fixed point).
        log_p = 0.5 * (log_p + eta * (values @ q))
        log_q = 0.5 * (log_q + eta * (-values.T @ p))
        return (log_p, log_q), None

    (log_p, log_q), _ = jax.lax.scan(body, (log_p, log_q), None, length=steps)
    return jax.nn.softmax(log_p), jax.nn.softmax(log_q)
