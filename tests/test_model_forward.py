"""Real player-model init + forward on the bundled example trajectory.

Marked gpu: runs wherever JAX puts it (the training box GPU, with
preallocation disabled by conftest so it coexists with a live learner).
Marked slow (~1 min): deselect with `-m "not slow"` for the quick suite.
"""

import dataclasses
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util

from rl.environment.data import MOVE_CELL_OFFSET, NUM_TARGET_SLOTS, OTHER_CELL_OFFSET
from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
)
from rl.environment.protos.features_pb2 import InfoFeature
from rl.model.constants import (
    MOVE_ROWS,
    NUM_FIELD_ROWS,
    NUM_HISTORY_REGISTERS,
    NUM_PUBLIC_SLOTS,
    PRIVATE_ROWS,
    SEQUENCE_SLICES,
    TARGET_ROWS,
    SequenceGroup,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_init_produces_finite_params(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
    real_model_apply: Callable,
) -> None:
    _, params, _, _ = real_model_and_trajectory
    leaves = jax.tree.leaves(params)
    assert leaves
    for leaf in leaves:
        assert np.isfinite(np.asarray(leaf, dtype=np.float32)).all()


def test_forward_outputs_finite_and_shaped(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
    real_model_apply: Callable,
) -> None:
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    out = real_model_apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]

    log_probs = np.asarray(out.value_head.log_probs, dtype=np.float32)
    assert log_probs.shape[0] == T
    assert np.isfinite(log_probs).all()
    # log_probs is a categorical distribution over the value support.
    np.testing.assert_allclose(np.exp(log_probs).sum(-1), 1.0, atol=1e-3)

    pi_lp = np.asarray(out.action_head.log_prob, dtype=np.float32)
    assert np.isfinite(pi_lp).all()


def test_forward_is_deterministic(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
    real_model_apply: Callable,
) -> None:
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    a = real_model_apply(params, actor_input, actor_output, HeadParams())
    b = real_model_apply(params, actor_input, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(a.value_head.log_probs, dtype=np.float32),
        np.asarray(b.value_head.log_probs, dtype=np.float32),
    )


def _assembled_rows(
    network: nn.Module, params: dict, actor_input: PlayerActorInput
) -> np.ndarray:
    """`_assembled_step_rows` at the trajectory's first timestep."""
    env_step = jax.tree.map(lambda x: x[0], actor_input.env)
    return _assembled_step_rows(network, params, env_step)


def _assembled_step_rows(
    network: nn.Module, params: dict, env_step: PlayerEnvOutput
) -> np.ndarray:
    """The trunk's input sequence for one timestep, BEFORE any attention.

    Every identity a row carries -- side, group, position -- is additive and
    applied here, so this is where an identity bug is visible. After the
    trunk every row has mixed with every other and the reading would be
    behavioural rather than structural.
    """

    def call(module: nn.Module, env_step: PlayerEnvOutput) -> jax.Array:
        encoder = module.encoder
        width = encoder.cfg.entity_size
        zero_slots = jnp.zeros((NUM_PUBLIC_SLOTS, width), env_step.field.dtype)
        sequence, _ = encoder._assemble_sequence(
            env_step,
            zero_slots.astype(encoder.cfg.dtype),
            jnp.zeros(NUM_PUBLIC_SLOTS, jnp.bool_),
            jnp.zeros((NUM_FIELD_ROWS, width), encoder.cfg.dtype),
            jnp.zeros((NUM_HISTORY_REGISTERS, width), encoder.cfg.dtype),
        )
        return sequence

    return np.asarray(
        jax.jit(lambda p, e: network.apply(p, e, method=call))(params, env_step),
        dtype=np.float32,
    )


def _zeroed(params: dict, *names: str) -> dict:
    """`params` with every leaf whose own name is in `names` set to zero."""
    flat = traverse_util.flatten_dict(params)
    return traverse_util.unflatten_dict(
        {
            path: jnp.zeros_like(leaf) if path[-1] in names else leaf
            for path, leaf in flat.items()
        }
    )


def test_prev_action_rows_are_the_rows_the_cell_names(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
) -> None:
    """The previous action is described by the rows its cell's logit is read
    from (heads.chosen_bank_rows) -- for a move cell, its move row and its
    target row -- not by a slot id.

    With the three additive identities zeroed (the src/tgt tags and the group
    bias) and every group's norm scale at its init of 1, each prev-action row
    must equal the row it names exactly: same content, same dtype, same norm.
    """
    network, params, actor_input, _ = real_model_and_trajectory
    move_block = np.asarray(actor_input.env.action_mask)[
        :, MOVE_CELL_OFFSET:OTHER_CELL_OFFSET
    ]
    step = int(np.flatnonzero(move_block.any(-1))[0])
    relative = int(np.flatnonzero(move_block[step])[0])
    move_slot, target_slot = divmod(relative, NUM_TARGET_SLOTS)

    env_step = jax.tree.map(lambda x: x[step], actor_input.env)
    info = (
        jnp.asarray(env_step.info).at[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION].set(1)
    )
    info = info.at[InfoFeature.INFO_FEATURE__PREV_ACTION_CELL].set(
        MOVE_CELL_OFFSET + relative
    )
    named = dataclasses.replace(env_step, info=info)
    prev_rows = SEQUENCE_SLICES[SequenceGroup.PREV_ACTION]

    bare = _zeroed(
        params, "prev_action_src_bias", "prev_action_tgt_bias", "sequence_group_bias"
    )
    rows = _assembled_step_rows(network, bare, named)
    assert np.abs(rows[prev_rows]).max() > 0
    np.testing.assert_array_equal(rows[prev_rows][0], rows[MOVE_ROWS][move_slot])
    np.testing.assert_array_equal(rows[prev_rows][1], rows[TARGET_ROWS][target_slot])

    # Controls: the src/tgt tags are live, so with them in place the rows
    # differ; and with no previous action both rows are exactly zero.
    tagged = _assembled_step_rows(
        network, _zeroed(params, "sequence_group_bias"), named
    )
    assert not np.array_equal(tagged[prev_rows][0], tagged[MOVE_ROWS][move_slot])
    absent = dataclasses.replace(
        named, info=info.at[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION].set(0)
    )
    np.testing.assert_array_equal(
        _assembled_step_rows(network, params, absent)[prev_rows], 0.0
    )


@pytest.mark.parametrize("side", [0, 1])
def test_private_sheet_uses_shared_side_after_normalisation(
    real_model_and_trajectory, side: int
) -> None:
    network, params, actor_input, _ = real_model_and_trajectory
    base = _assembled_rows(network, params, actor_input)[PRIVATE_ROWS]
    changed = jax.tree.map(lambda value: value, params)
    side_table = changed["params"]["encoder"]["side_bias"]["embedding"]
    changed["params"]["encoder"]["side_bias"]["embedding"] = side_table.at[side].add(1)
    moved = _assembled_rows(network, changed, actor_input)[PRIVATE_ROWS]
    valid = np.any(base != 0, axis=-1)
    assert valid.any()
    np.testing.assert_allclose(moved[valid] - base[valid], side, atol=0.03)
    np.testing.assert_array_equal(moved[~valid], 0)


def _with_group_bias(params: dict, value: float) -> dict:
    tree = jax.tree.map(lambda x: x, params)
    encoder_params = tree["params"]["encoder"]
    encoder_params["sequence_group_bias"] = jnp.full_like(
        encoder_params["sequence_group_bias"], value
    )
    return tree


def test_row_identity_is_added_after_the_input_norm(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
) -> None:
    """The content is normalised to RMS 1 and the group identity goes on
    AFTER (2026-09-10): the assembled rows with a group bias of all-ones minus
    the rows with no bias equal that bias exactly, and the unbiased rows
    that carry content sit at RMS 1. A bias added BEFORE the norm is divided
    by the content's RMS along with it, so the difference would be
    normalise(content + 1) - normalise(content), not 1 -- the rows with
    content are the discriminator, which is why the test requires some (the
    harness's zero history and field rows are bias-only either way)."""
    network, params, actor_input, _ = real_model_and_trajectory
    content_params = jax.tree.map(lambda value: value, params)
    encoder_params = content_params["params"]["encoder"]
    for name in ("side_bias", "position_bias"):
        encoder_params[name]["embedding"] = jnp.zeros_like(
            encoder_params[name]["embedding"]
        )
    encoder_params["target_slot_embeddings"] = jnp.zeros_like(
        encoder_params["target_slot_embeddings"]
    )
    base = _assembled_rows(network, _with_group_bias(content_params, 0.0), actor_input)
    biased = _assembled_rows(
        network, _with_group_bias(content_params, 1.0), actor_input
    )
    valid = np.any(biased != 0, axis=-1)
    with_content = np.any(base != 0, axis=-1)
    assert with_content.sum() > 1
    assert np.array_equal(with_content & valid, with_content)
    np.testing.assert_allclose(biased[valid] - base[valid], 1, atol=0.03)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(base[with_content]), axis=-1)), 1, atol=0.03
    )
    np.testing.assert_array_equal(biased[~valid], 0)


def _trunk_rows(
    network: nn.Module, params: dict, actor_input: PlayerActorInput
) -> np.ndarray:
    """The encoder's output rows for the trajectory, (T, rows, width): what
    every head reads, i.e. AFTER the output norm."""

    def call(
        module: nn.Module, env: PlayerEnvOutput, packed: object, history: object
    ) -> jax.Array:
        return module.encoder(env, packed, history)[0]

    apply = jax.jit(
        lambda p, env, packed, history: network.apply(
            p, env, packed, history, method=call
        )
    )
    return np.asarray(
        apply(params, actor_input.env, actor_input.packed_history, actor_input.history),
        np.float32,
    )


def test_every_row_leaves_the_trunk_at_unit_rms(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
) -> None:
    """The output norm (2026-09-11): every valid row leaves the trunk at RMS
    1 at init and invalid rows at exactly 0, whatever the blocks wrote.
    Positive controls: +1 on every group of the OUTPUT bank doubles every
    valid row; +1 on the INPUT bank changes the trunk's content but leaves
    the output RMS at 1 -- it is the output norm doing the sizing."""
    network, params, actor_input, _ = real_model_and_trajectory
    rows = _trunk_rows(network, params, actor_input)
    valid = np.any(rows != 0, axis=-1)
    assert valid.sum() > 1

    def row_rms(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.square(values), axis=-1))

    np.testing.assert_allclose(row_rms(rows)[valid], 1, atol=0.03)
    doubled = _trunk_rows(
        network,
        _perturbed(params, ("output_normalisation", "group_scale")),
        actor_input,
    )
    np.testing.assert_allclose(row_rms(doubled)[valid], 2, atol=0.06)
    np.testing.assert_array_equal(doubled[~valid], 0)
    rescaled_in = _trunk_rows(
        network, _perturbed(params, ("input_normalisation", "group_scale")), actor_input
    )
    assert np.max(np.abs(rescaled_in[valid] - rows[valid])) > 0.1
    np.testing.assert_allclose(row_rms(rescaled_in)[valid], 1, atol=0.03)


def _perturbed(params: dict, path: tuple[str, ...], delta: float = 1.0) -> dict:
    tree = jax.tree.map(lambda x: x, params)
    node = tree["params"]["encoder"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]] + delta
    return tree


def test_current_and_remembered_field_share_side_only(
    real_model_and_trajectory,
) -> None:
    network, params, actor_input, _ = real_model_and_trajectory
    base = _assembled_rows(network, params, actor_input)
    moved_pos = _assembled_rows(
        network, _perturbed(params, ("position_bias", "embedding")), actor_input
    )
    moved_side = _assembled_rows(
        network, _perturbed(params, ("side_bias", "embedding")), actor_input
    )
    for group in (SequenceGroup.FIELD, SequenceGroup.HISTORY_FIELD):
        rows = SEQUENCE_SLICES[group]
        np.testing.assert_array_equal(base[rows], moved_pos[rows])
        np.testing.assert_allclose((moved_side - base)[rows][1:], 1, atol=0.03)
        np.testing.assert_array_equal(base[rows][0], moved_side[rows][0])


def _with_private_column(
    actor_input: PlayerActorInput, row: int, column: int, value: int
) -> PlayerActorInput:
    """actor_input with private_team[:, row, column] set to `value`."""
    import dataclasses

    env = actor_input.env
    team = jnp.asarray(env.private_team).at[:, row, column].set(value)
    return dataclasses.replace(
        actor_input, env=dataclasses.replace(env, private_team=team)
    )


def test_private_condition_reaches_only_its_own_sheet_row(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
) -> None:
    """The truth channel is wired: a candidate's CURRENT hp on the wire moves
    its own assembled sheet row and no other -- the input-level half of what
    probe C measures behaviourally after training."""
    from rl.environment.protos.features_pb2 import EntityPrivateNodeFeature

    network, params, actor_input, _ = real_model_and_trajectory
    base = _assembled_rows(network, params, actor_input)[PRIVATE_ROWS]

    halved = _with_private_column(
        actor_input,
        2,
        EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO,
        4096,
    )
    moved = _assembled_rows(network, params, halved)[PRIVATE_ROWS]

    changed_rows = ~np.all(np.isclose(base, moved, atol=1e-6), axis=-1)
    assert changed_rows[2], "the perturbed candidate's own row must move"
    assert not changed_rows[
        [0, 1, 3, 4, 5]
    ].any(), "condition must be entity-local at assembly time"


def test_private_position_follows_public_key(real_model_and_trajectory) -> None:
    from rl.environment.protos.features_pb2 import (
        EntityPrivateNodeFeature,
        EntityPublicNodeFeature,
    )
    from rl.model.constants import PUBLIC_ROWS

    network, params, actor_input, _ = real_model_and_trajectory
    env_step = jax.tree.map(lambda value: value[0], actor_input.env)
    public = jnp.asarray(env_step.public_team)
    public = public.at[
        0, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
    ].set(2)
    public = public.at[0, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE].set(
        1
    )
    info = (
        jnp.asarray(env_step.info).at[InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0].set(0)
    )
    private_column = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    private = jnp.asarray(env_step.private_team).at[:, private_column].set(0)
    benched = dataclasses.replace(
        env_step, public_team=public, info=info, private_team=private
    )
    active = dataclasses.replace(
        benched, private_team=private.at[0, private_column].set(1)
    )
    base = _assembled_step_rows(network, params, benched)
    moved = _assembled_step_rows(network, params, active)
    position_table = np.asarray(
        params["params"]["encoder"]["position_bias"]["embedding"], np.float32
    )
    assert np.any(base[PRIVATE_ROWS][0])
    expected = position_table[2] - position_table[0]
    assert np.max(np.abs(expected)) > 0.01
    np.testing.assert_allclose((moved - base)[PRIVATE_ROWS][0], expected, atol=0.03)
    np.testing.assert_array_equal(base[PRIVATE_ROWS][1:], moved[PRIVATE_ROWS][1:])
    np.testing.assert_array_equal(base[PUBLIC_ROWS], moved[PUBLIC_ROWS])


def test_history_rows_are_normalised_memory_plus_side_position_and_group(
    real_model_and_trajectory,
):
    from rl.environment.protos.features_pb2 import EntityPublicNodeFeature
    from rl.model.constants import HISTORY_ENTITY_ROWS

    network, params, actor_input, _ = real_model_and_trajectory
    env_step = jax.tree.map(lambda value: value[0], actor_input.env)
    width = params["params"]["encoder"]["sequence_group_bias"].shape[-1]
    memory = (
        jnp.arange(NUM_PUBLIC_SLOTS * width, dtype=jnp.float32).reshape(
            NUM_PUBLIC_SLOTS, width
        )
        + 1
    )
    valid = jnp.ones(NUM_PUBLIC_SLOTS, jnp.bool_).at[-1].set(False)

    def assemble(module, env, row_states):
        encoder = module.encoder
        return encoder._assemble_sequence(
            env,
            row_states.astype(encoder.cfg.dtype),
            valid,
            jnp.zeros((NUM_FIELD_ROWS, width), encoder.cfg.dtype),
            jnp.zeros((NUM_HISTORY_REGISTERS, width), encoder.cfg.dtype),
        )[0]

    rows = np.asarray(
        jax.jit(
            lambda tree, env, states: network.apply(tree, env, states, method=assemble)
        )(params, env_step, memory),
        np.float32,
    )[HISTORY_ENTITY_ROWS]
    encoder_params = params["params"]["encoder"]
    # Match the forward's input rounding before comparing its normalisation.
    compute_dtype = network.cfg.encoder.dtype
    content = np.asarray(memory.astype(compute_dtype), np.float32)
    normalised = content / np.sqrt(np.square(content).mean(-1, keepdims=True) + 1e-6)
    sides = np.asarray(
        env_step.public_team[
            :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
        ]
    )
    expected = (
        normalised
        + np.asarray(encoder_params["side_bias"]["embedding"], np.float32)[sides]
    )
    positions = np.asarray(
        env_step.public_team[
            :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
        ]
    )
    expected += np.asarray(encoder_params["position_bias"]["embedding"], np.float32)[
        positions
    ]
    expected += np.asarray(encoder_params["sequence_group_bias"], np.float32)[
        SequenceGroup.HISTORY_ENTITY
    ]
    np.testing.assert_allclose(rows[:-1], expected[:-1], atol=0.03)
    np.testing.assert_array_equal(rows[-1], 0)
