"""The distribution an actor draws from (rl/model/utils.py
sampling_log_policy): the policy as trained, or under `HeadParams.greedy`
the point mass on its most likely legal cell -- the `argmax` eval slot."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.interfaces import PlayerActorInput, PlayerActorOutput
from rl.model.utils import legal_log_policy, sampling_log_policy

DTYPE_MIN = jnp.finfo(jnp.float32).min


def _rows(logits: list[list[float]], legal: list[list[bool]]):
    legal = jnp.asarray(legal)
    return legal_log_policy(jnp.asarray(logits, jnp.float32), legal), legal


def test_not_greedy_is_the_policy_bit_identical() -> None:
    log_policy, legal = _rows([[1.0, 3.0, 9.0, 2.0]], [[True, True, False, True]])
    sampled = sampling_log_policy(log_policy, legal, False)
    np.testing.assert_array_equal(
        np.asarray(sampled), np.where(legal, log_policy, DTYPE_MIN)
    )


def test_greedy_is_the_point_mass_on_the_best_legal_cell() -> None:
    # Row 0's largest logit is ILLEGAL: an argmax over the raw logits
    # would pick cell 2, so this row fails if the legal mask is skipped.
    # Row 1 ties cells 0 and 3; the lowest index wins.
    log_policy, legal = _rows(
        [[1.0, 3.0, 9.0, 2.0], [4.0, 1.0, 0.0, 4.0]],
        [[True, True, False, True], [True, True, True, True]],
    )
    sampled = np.asarray(sampling_log_policy(log_policy, legal, True))
    expected = np.full((2, 4), DTYPE_MIN, np.float32)
    expected[0, 1] = 0.0
    expected[1, 0] = 0.0
    np.testing.assert_array_equal(sampled, expected)


def test_the_sampler_draws_the_greedy_cell_under_every_key() -> None:
    log_policy, legal = _rows([[0.0, 0.1, 0.0, 0.0]], [[True, True, True, True]])
    keys = jax.random.split(jax.random.key(0), 256)

    def draw(key: jax.Array, greedy: bool) -> jax.Array:
        return jax.random.categorical(
            key, sampling_log_policy(log_policy, legal, greedy), axis=-1
        )

    greedy_draws = np.asarray(jax.vmap(lambda key: draw(key, True))(keys))
    assert (greedy_draws == 1).all()
    # Positive control: the same keys over the near-uniform policy itself
    # land elsewhere.
    plain_draws = np.asarray(jax.vmap(lambda key: draw(key, False))(keys))
    assert (plain_draws != 1).any()


@pytest.mark.gpu
@pytest.mark.slow
def test_real_actor_greedy_changes_the_play_and_no_metric(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
) -> None:
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model
    from rl.model.utils import open_zero_init_paths
    from tests.conftest import session_player_model_config

    _, variables, actor_input, actor_output = real_model_and_trajectory
    variables = open_zero_init_paths(variables, ["action_head"])
    actor_input = actor_input.replace(
        env=jax.tree.map(lambda leaf: leaf[:4], actor_input.env)
    )
    actor_output = jax.tree.map(lambda leaf: leaf[:4], actor_output)
    apply_model = jax.jit(
        get_player_model(session_player_model_config(train=False)).apply
    )
    rngs = {"sampling": jax.random.key(9)}
    plain = apply_model(variables, actor_input, actor_output, HeadParams(), rngs=rngs)
    explicit = apply_model(
        variables, actor_input, actor_output, HeadParams(greedy=False), rngs=rngs
    )
    for expected, actual in zip(
        jax.tree.leaves(plain), jax.tree.leaves(explicit), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)
    # Positive control: greedy replaces the sampled distribution (the
    # stored log_prob is mu's, a point mass) but no policy metric.
    greedy = apply_model(
        variables, actor_input, actor_output, HeadParams(greedy=True), rngs=rngs
    )
    assert (np.asarray(plain.action_head.log_prob) < 0).any()
    np.testing.assert_array_equal(np.asarray(greedy.action_head.log_prob), 0.0)
    np.testing.assert_array_equal(
        np.asarray(greedy.action_head.entropy),
        np.asarray(plain.action_head.entropy),
    )
    other_key = apply_model(
        variables,
        actor_input,
        actor_output,
        HeadParams(greedy=True),
        rngs={"sampling": jax.random.key(10)},
    )
    np.testing.assert_array_equal(
        np.asarray(greedy.action_head.action_index),
        np.asarray(other_key.action_head.action_index),
    )
