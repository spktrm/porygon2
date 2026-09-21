"""Fixed observations, action-feature reach and resumable optimiser contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from constants import MAX_RATIO_TOKEN
from rl.environment.consequence_labels import (
    NUM_OBSERVABLE_LOGITS,
    observed_consequences,
)
from rl.environment.data import MOVE_CELL_OFFSET
from rl.environment.interfaces import (
    PlayerEnvOutput,
    PlayerHistoryOutput,
    PlayerPackedHistoryOutput,
)
from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.model.constants import NUM_PUBLIC_SLOTS, RELEVANT_ENTITY_FEATURES
from rl.online.artifact import merge_opt_state, merge_params, player_optimiser
from rl.online.config import get_learner_config
from rl.online.training.observable_consequence import observable_terms, scale_features


def fixture():
    steps = 3
    info = np.zeros((steps, 1, max(InfoFeature.values()) + 1), np.int32)
    order_start = InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0
    info[..., order_start : order_start + NUM_PUBLIC_SLOTS] = -1
    info[0, 0, order_start : order_start + 2] = [0, 1]
    info[1:, 0, order_start : order_start + 3] = [1, 0, 2]
    info[:, 0, InfoFeature.INFO_FEATURE__REQUEST_COUNT] = [10, 11, 12]
    public = np.zeros(
        (steps, 1, NUM_PUBLIC_SLOTS, max(EntityPublicNodeFeature.values()) + 1),
        np.int32,
    )
    hp = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
    public[0, 0, :2, hp] = MAX_RATIO_TOKEN
    public[1:, 0, 1, hp] = MAX_RATIO_TOKEN // 2
    public[1:, 0, 0, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED] = 1
    field = np.zeros((3, 1, max(FieldFeature.values()) + 1), np.int32)
    field[:, 0, FieldFeature.FIELD_FEATURE__VALID] = 1
    field[:, 0, FieldFeature.FIELD_FEATURE__NUM_RELEVANT] = 1
    field[:, 0, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] = [11, 11, 12]
    field[:, 0, RELEVANT_ENTITY_FEATURES[0]] = [0, 1, 2]
    edge = np.zeros((3, 1, max(EntityEdgeFeature.values()) + 1), np.int32)
    edge[:, 0, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG] = [
        BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__CANT,
        BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE,
        BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE,
    ]
    cache = np.zeros((3, 1, public.shape[-1]), np.int32)
    cache[:, 0, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE] = [1, 2, 1]
    env = PlayerEnvOutput(
        info=jnp.asarray(info),
        public_team=jnp.asarray(public),
        done=jnp.array([[False], [False], [True]]),
    )
    history = PlayerHistoryOutput(field=jnp.asarray(field))
    packed = PlayerPackedHistoryOutput(
        edge_cache=jnp.asarray(edge), public_cache=jnp.asarray(cache)
    )
    actions = jnp.full((steps, 1), MOVE_CELL_OFFSET)
    acted = jnp.array([[True], [True], [False]])
    return env, history, packed, actions, acted


def test_targets_follow_identity_and_keep_terminal_transition():
    targets = jax.jit(observed_consequences)(*fixture())
    assert targets.execution_valid[:, 0].tolist() == [True, True, False]
    assert targets.executed[:, 0].tolist() == [False, True, True]
    np.testing.assert_allclose(
        targets.hp_change[0, 0, :2], [-0.5, -1], atol=1 / MAX_RATIO_TOKEN
    )
    assert targets.fainted[0, 0, :2].tolist() == [False, True]
    assert not targets.hp_valid[0, 0, 2]
    assert targets.hp_valid[1, 0, 0]
    assert not targets.faint_valid[1, 0, 0]
    assert not targets.hp_valid[-1].any()


def test_missing_execution_and_duplicate_identity_are_unknown():
    env, history, packed, actions, acted = fixture()
    history = history.replace(field=jnp.zeros_like(history.field))
    start = InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0
    env = env.replace(info=env.info.at[1, 0, start + 1].set(1))
    targets = jax.jit(observed_consequences)(env, history, packed, actions, acted)
    assert not targets.execution_valid.any()
    assert not targets.hp_valid[0, 0, :2].any()


def test_losses_ignore_unknown_labels_and_have_live_gradients():
    targets = jax.jit(observed_consequences)(*fixture())
    logits = jnp.zeros((3, 1, NUM_OBSERVABLE_LOGITS))
    loss, gradient = jax.jit(
        jax.value_and_grad(lambda output: observable_terms(output, targets)[0])
    )(logits)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(gradient).all()
    assert jnp.abs(gradient[:-1]).max() > 0
    np.testing.assert_array_equal(gradient[-1], 0)
    good_loss = observable_terms(logits - 10 * gradient, targets)[0]
    assert good_loss < loss
    for scale in (0.0, 1.0):
        reached = jax.grad(
            lambda features: observable_terms(scale_features(features, scale), targets)[
                0
            ]
        )(logits)
        np.testing.assert_allclose(reached, scale * gradient)


def test_new_decoder_preserves_existing_partitioned_adam_state():
    config = get_learner_config()
    old = {
        "params": {
            "encoder": {"kernel": jnp.ones((2, 2))},
            "consequence": {"pair": {"kernel": jnp.ones((2, 2))}},
        }
    }
    new = {
        "params": {
            **old["params"],
            "consequence": {
                **old["params"]["consequence"],
                "observable_outcomes": {
                    "kernel": jnp.zeros((2, NUM_OBSERVABLE_LOGITS)),
                    "bias": jnp.zeros(NUM_OBSERVABLE_LOGITS),
                },
            },
        }
    }
    optimiser = player_optimiser(config)
    old_state = optimiser.init(old)
    _, old_state = optimiser.update(jax.tree.map(jnp.ones_like, old), old_state, old)
    merged, fresh, dropped, _ = merge_params(new, old)
    assert fresh == ["/params/consequence/observable_outcomes"]
    assert not dropped
    state = merge_opt_state(optimiser.init(new), old_state)
    for partition in ("model", "consequence"):
        old_adam = old_state.inner_states[partition].inner_state[1][0]
        new_adam = state.inner_states[partition].inner_state[1][0]
        assert int(new_adam.count) == int(old_adam.count)
        for moment in ("mu", "nu"):
            old_moment = getattr(old_adam, moment)
            new_moment = getattr(new_adam, moment)
            for name in ("encoder", "consequence"):
                for leaf_name, leaf in old_moment["params"][name].items():
                    jax.tree.map(
                        np.testing.assert_array_equal,
                        leaf,
                        new_moment["params"][name][leaf_name],
                    )
    optimiser.update(jax.tree.map(jnp.ones_like, merged), state, merged)


def test_gradient_ramp_preserves_bf16_features():
    features = jnp.ones((8,), jnp.bfloat16)
    scaled = jax.jit(scale_features)(features, jnp.asarray(0.25, jnp.float32))
    assert scaled.dtype == jnp.bfloat16
    np.testing.assert_array_equal(scaled, features)
    reached = jax.jit(
        jax.grad(
            lambda rows: scale_features(rows, jnp.asarray(0.25, jnp.float32)).sum()
        )
    )(features)
    np.testing.assert_array_equal(reached, jnp.full_like(features, 0.25))
