"""Depth-1 sampled event rollouts on the event world model (plan Step 4).

Free functions over an `EventSearchFns` bundle of callables so the operator
runs on stubs in tests. From a root public state and one own declared
action, an event is sampled from the decoder one grammar position at a
time (KIND, ACTOR, MOVE, TARGET, TOUCHED), our own execution event is
forced to the declaration, the flow imagines the next state on the
touched rows, and the rollout stops at the next own decision point: a new
turn, an own-side faint with reserves, or the end of the game. The value
of the state reached is the leaf; Q(a) averages the leaves over rollouts.
"""

import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from rl.environment.data import (
    MOVE_CELL_OFFSET,
    MOVE_INDICES,
    NUM_SWITCH_CELLS,
    OTHER_CELL_OFFSET,
    TARGET_SLOT_INDICES,
)
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    InfoFeature,
    MovesetFeature,
)
from rl.model import world_model as wm
from rl.model.constants import NUM_PUBLIC_SLOTS
from rl.model.identity import FIRST_ACTIVE_POSITION, SECOND_ACTIVE_POSITION
from rl.offline.event_labels import NO_SLOT, DeclaredKind, EventKind

SIDE_MINE = 1
SIDE_OPPONENT = 0


class EventSearchFns(NamedTuple):
    # (rows, declared_kind, declared_arg, tokens, actor_is_mine) -> DecoderLogits
    decode_fn: Callable
    # (rows, tokens, row_mask, rng) -> rows
    imagine_fn: Callable
    # (rows) -> scalar value in [-1, 1]
    value_fn: Callable
    # (rows) -> scalar expected terminal outcome in [-1, 1]
    terminal_fn: Callable


class RolloutBudget(NamedTuple):
    max_events: int = 8
    temp: float = 1.0
    value_blind: bool = False


class RootInfo(NamedTuple):
    """What the root knows about the slots: sides, the revealed count, the
    active slots the target rows follow (-1 = none), and whether we hold a
    reserve for a forced switch."""

    slot_sides: jax.Array  # (12,)
    num_revealed: jax.Array  # ()
    ally_active_slots: jax.Array  # (2,)
    enemy_active_slots: jax.Array  # (2,)
    has_reserve: jax.Array  # ()


def root_info(env_step) -> RootInfo:
    """What the root request tells the rollout: each public slot's side
    (PUBLIC_ORDER maps public rows to slots), the revealed count, the
    active slots the target rows follow, and whether a reserve exists for
    a forced switch (an alive, unfainted bench mon on my sheet)."""
    order = env_step.info[
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1
    ]
    order_valid = (order >= 0) & (order < NUM_PUBLIC_SLOTS)
    sides = env_step.public_team[
        :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
    ]
    positions = env_step.public_team[
        :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
    ]
    slot_sides = (
        jnp.zeros(NUM_PUBLIC_SLOTS, jnp.int32)
        .at[order.clip(0, NUM_PUBLIC_SLOTS - 1)]
        .set(jnp.where(order_valid, sides, 0))
    )
    num_revealed = jnp.where(order_valid, order + 1, 0).max()

    def active(side, position):
        match = order_valid & (sides == side) & (positions == position)
        return jnp.where(match.any(), order[match.argmax()], -1)

    ally = jnp.stack(
        [
            active(SIDE_MINE, FIRST_ACTIVE_POSITION),
            active(SIDE_MINE, SECOND_ACTIVE_POSITION),
        ]
    )
    enemy = jnp.stack(
        [
            active(SIDE_OPPONENT, FIRST_ACTIVE_POSITION),
            active(SIDE_OPPONENT, SECOND_ACTIVE_POSITION),
        ]
    )
    private = env_step.private_team
    known = (
        private[:, EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX] > 0
    )
    fainted = (
        private[:, EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__FAINTED] > 0
    )
    has_reserve = (
        env_step.action_mask[:MOVE_CELL_OFFSET].any() | (known & ~fainted).sum() > 1
    )
    return RootInfo(
        slot_sides=slot_sides,
        num_revealed=num_revealed,
        ally_active_slots=ally,
        enemy_active_slots=enemy,
        has_reserve=has_reserve,
    )


def declared_from_cell(
    cell: jax.Array, env_step, root: RootInfo
) -> tuple[jax.Array, jax.Array]:
    """A block cell as the declaration the world model reads: a switch
    cell names the private row's public slot (an unrevealed mon is the
    next slot to be revealed), a move cell its move id, the standalone
    cells are UNKNOWN."""
    is_switch = cell < MOVE_CELL_OFFSET
    is_move = (cell >= MOVE_CELL_OFFSET) & (cell < OTHER_CELL_OFFSET)
    private_row = cell.clip(0, NUM_SWITCH_CELLS - 1)
    entity = env_step.private_team[
        private_row, EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    ]
    switch_slot = jnp.where(entity > 0, entity - 1, root.num_revealed)
    move_slot = ((cell - MOVE_CELL_OFFSET) // len(TARGET_SLOT_INDICES)).clip(
        0, len(MOVE_INDICES) - 1
    )
    move_id = env_step.my_moveset[move_slot, MovesetFeature.MOVESET_FEATURE__MOVE_ID]
    kind = jnp.where(
        is_switch,
        DeclaredKind.SWITCH,
        jnp.where(is_move, DeclaredKind.MOVE, DeclaredKind.UNKNOWN),
    ).astype(jnp.int32)
    arg = jnp.where(is_switch, switch_slot, jnp.where(is_move, move_id, 0)).astype(
        jnp.int32
    )
    return kind, arg


class RolloutState(NamedTuple):
    rows: jax.Array
    num_revealed: jax.Array
    own_executed: jax.Array  # our declared action has been played out
    stopped: jax.Array
    ended: jax.Array  # stopped at END: the terminal head is the leaf
    num_events: jax.Array


def blank_tokens() -> wm.EventTokens:
    return wm.EventTokens(
        kind=jnp.asarray(0, jnp.int32),
        actor=jnp.asarray(NO_SLOT, jnp.int32),
        move=jnp.asarray(0, jnp.int32),
        target=jnp.asarray(NO_SLOT, jnp.int32),
    )


def _sample(
    logits: jax.Array, legal: jax.Array, rng: jax.Array, temp: float
) -> jax.Array:
    masked = jnp.where(legal, logits / temp, -1e9)
    return jax.random.categorical(rng, masked)


def sample_event(
    rows: jax.Array,
    state: RolloutState,
    declared_kind: jax.Array,
    declared_arg: jax.Array,
    root: RootInfo,
    fns: EventSearchFns,
    rng: jax.Array,
    temp: float,
) -> tuple[wm.EventTokens, jax.Array, jax.Array, jax.Array]:
    """One event from the decoder, position by position (five decoder
    passes with a growing teacher-forced prefix). Returns the tokens, the
    touched bits, the new-turn bit and whether the actor is our side."""
    keys = jax.random.split(rng, 6)
    blank = blank_tokens()
    # KIND: the decoder's own prior; the declared token is readable here.
    logits = fns.decode_fn(rows, declared_kind, declared_arg, blank, jnp.asarray(True))
    kind = _sample(logits.kind, jnp.ones(wm.NUM_EVENT_KINDS, jnp.bool_), keys[0], temp)
    new_turn = jax.random.bernoulli(keys[1], jax.nn.sigmoid(logits.new_turn))
    tokens = blank.replace(kind=kind)
    logits = fns.decode_fn(rows, declared_kind, declared_arg, tokens, jnp.asarray(True))
    actor = _sample(
        logits.actor, wm.actor_mask(kind, state.num_revealed), keys[2], temp
    )
    # A newly revealed slot has no side on file: it is ours only when our
    # declaration is a switch we have not yet played.
    own_new_switch = (
        (declared_kind == DeclaredKind.SWITCH)
        & ~state.own_executed
        & (actor == state.num_revealed)
    )
    known_side = root.slot_sides[actor.clip(0, wm.NUM_PUBLIC_SLOTS - 1)] == SIDE_MINE
    actor_is_mine = jnp.where(actor == state.num_revealed, own_new_switch, known_side)
    actor_is_mine = actor_is_mine & (actor != NO_SLOT)
    tokens = tokens.replace(actor=actor)
    logits = fns.decode_fn(rows, declared_kind, declared_arg, tokens, actor_is_mine)
    move = _sample(logits.move, wm.move_mask(kind), keys[3], temp)
    # Our own execution event carries the declared move; a sampled CANT stands.
    forced_move = (
        actor_is_mine
        & ~state.own_executed
        & (kind == EventKind.MOVE)
        & (declared_kind == DeclaredKind.MOVE)
    )
    move = jnp.where(forced_move, declared_arg, move)
    move = jnp.where(wm.move_position_valid(kind), move, 0)
    tokens = tokens.replace(move=move)
    logits = fns.decode_fn(rows, declared_kind, declared_arg, tokens, actor_is_mine)
    target = _sample(logits.target, wm.target_mask(state.num_revealed), keys[4], temp)
    target = jnp.where(wm.target_position_valid(kind), target, NO_SLOT)
    tokens = tokens.replace(target=target)
    logits = fns.decode_fn(rows, declared_kind, declared_arg, tokens, actor_is_mine)
    touched = jax.random.bernoulli(keys[5], jax.nn.sigmoid(logits.touched))
    return tokens, touched, new_turn, actor_is_mine


def initial_state(rows: jax.Array, root: RootInfo) -> RolloutState:
    return RolloutState(
        rows=rows,
        num_revealed=root.num_revealed,
        own_executed=jnp.asarray(False),
        stopped=jnp.asarray(False),
        ended=jnp.asarray(False),
        num_events=jnp.asarray(0, jnp.int32),
    )


def rollout_step(
    state: RolloutState,
    key: jax.Array,
    declared_kind: jax.Array,
    declared_arg: jax.Array,
    root: RootInfo,
    fns: EventSearchFns,
    budget: RolloutBudget,
) -> RolloutState:
    """One sampled event applied to the state, carried unchanged once the
    rollout has stopped."""
    event_key, imagine_key = jax.random.split(key)
    tokens, touched, new_turn, actor_is_mine = sample_event(
        state.rows,
        state,
        declared_kind,
        declared_arg,
        root,
        fns,
        event_key,
        budget.temp,
    )
    is_end = tokens.kind == EventKind.END
    row_mask = wm.update_rows(
        touched[: wm.NUM_PUBLIC_SLOTS],
        touched[wm.FIELD_TOUCHED_BIT],
        root.ally_active_slots,
        root.enemy_active_slots,
    )
    next_rows = fns.imagine_fn(state.rows, tokens, row_mask, imagine_key)
    executed = state.own_executed | (
        actor_is_mine
        & (
            (tokens.kind == EventKind.MOVE)
            | (tokens.kind == EventKind.SWITCH)
            | (tokens.kind == EventKind.CANT)
        )
    )
    own_faint = (tokens.kind == EventKind.FAINT) & actor_is_mine & root.has_reserve
    stop_after = new_turn | own_faint | is_end
    live = ~state.stopped
    return RolloutState(
        rows=jnp.where(live, next_rows, state.rows),
        num_revealed=jnp.where(
            live & (tokens.actor == state.num_revealed),
            state.num_revealed + 1,
            state.num_revealed,
        ),
        own_executed=jnp.where(live, executed, state.own_executed),
        stopped=state.stopped | stop_after,
        ended=state.ended | (live & is_end),
        num_events=state.num_events + live.astype(jnp.int32),
    )


def rollout(
    rows: jax.Array,
    declared_kind: jax.Array,
    declared_arg: jax.Array,
    root: RootInfo,
    fns: EventSearchFns,
    budget: RolloutBudget,
    rng: jax.Array,
    scan: Callable | None = None,
) -> RolloutState:
    """Sample events until the next own decision point or `max_events`.
    `scan(step, init, keys)` runs `step(fns, state, key)`; the default is
    `lax.scan` over these `fns`, and a flax module passes a lifted scan
    that rebuilds the readers from the transformed module, because they
    carry parameters."""

    def step(step_fns, state, key):
        return (
            rollout_step(
                state, key, declared_kind, declared_arg, root, step_fns, budget
            ),
            None,
        )

    if scan is None:

        def scan(step_fn, init, keys):
            return jax.lax.scan(functools.partial(step_fn, fns), init, keys)

    final, _ = scan(
        step, initial_state(rows, root), jax.random.split(rng, budget.max_events)
    )
    return final


def leaf_value(state: RolloutState, fns: EventSearchFns) -> jax.Array:
    return jnp.where(state.ended, fns.terminal_fn(state.rows), fns.value_fn(state.rows))


def q_values(
    rows: jax.Array,
    declared_kinds: jax.Array,
    declared_args: jax.Array,
    cell_valid: jax.Array,
    root: RootInfo,
    fns: EventSearchFns,
    budget: RolloutBudget,
    num_samples: int,
    rng: jax.Array,
    scan: Callable | None = None,
) -> tuple[jax.Array, RolloutState]:
    """(C,) the mean leaf value over `num_samples` rollouts per declared
    action (C static candidates, `cell_valid` masking the padding), plus
    the rollouts' final states (C, S) for diagnostics."""
    num_cells = declared_kinds.shape[0]
    keys = jax.random.split(rng, num_cells * num_samples).reshape(
        num_cells, num_samples
    )

    def per_cell(declared_kind, declared_arg, cell_keys):
        def per_sample(key):
            final = rollout(
                rows, declared_kind, declared_arg, root, fns, budget, key, scan=scan
            )
            return leaf_value(final, fns), final

        values, finals = jax.vmap(per_sample)(cell_keys)
        return values.mean(), finals

    values, finals = jax.vmap(per_cell)(declared_kinds, declared_args, keys)
    return jnp.where(cell_valid, values, 0.0), finals


def search_bonus(
    q: jax.Array, cell_valid: jax.Array, temp: float, value_blind: bool
) -> jax.Array:
    """The additive logit bonus Q / temp on the valid cells; the value-blind
    arm consumed the same rollouts and adds nothing."""
    bonus = jnp.where(cell_valid, q / temp, 0.0)
    if value_blind:
        return jnp.zeros_like(bonus)
    return bonus


def search_diagnostics(
    base_logits: jax.Array, bonus: jax.Array, cell_valid: jax.Array
) -> dict[str, jax.Array]:
    """Root KL between the searched and the base policy over the valid
    cells, and the gap between the best and the policy-mean bonus."""
    base = jax.nn.log_softmax(jnp.where(cell_valid, base_logits, -1e9))
    searched = jax.nn.log_softmax(jnp.where(cell_valid, base_logits + bonus, -1e9))
    probs = jnp.exp(searched)
    kl = jnp.sum(jnp.where(cell_valid, probs * (searched - base), 0.0))
    expected_bonus = jnp.sum(jnp.where(cell_valid, jnp.exp(base) * bonus, 0.0))
    best_bonus = jnp.max(jnp.where(cell_valid, bonus, -jnp.inf))
    return dict(search_root_kl=kl, search_bonus_gap=best_bonus - expected_bonus)
