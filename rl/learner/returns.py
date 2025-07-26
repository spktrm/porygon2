import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from rl.environment.interfaces import Transition
from rl.learner.config import Porygon2LearnerConfig


class VTraceOutput(NamedTuple):
    returns: jax.Array
    pg_advantage: jax.Array
    q_estimate: jax.Array


class Targets(NamedTuple):
    vtrace: VTraceOutput
    target_log_pi: jax.Array


def vtrace(
    v_tm1: jax.Array,
    v_t: jax.Array,
    r_t: jax.Array,
    discount_t: jax.Array,
    rho_tm1: jax.Array,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> jax.Array:
    """Calculates V-Trace errors from importance weights.

    V-trace computes TD-errors from multistep trajectories by applying
    off-policy corrections based on clipped importance sampling ratios.

    See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
    Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561).

    Args:
      v_tm1: values at time t-1.
      v_t: values at time t.
      r_t: reward at time t.
      discount_t: discount at time t.
      rho_tm1: importance sampling ratios at time t-1.
      lambda_: mixing parameter; a scalar or a vector for timesteps t.
      clip_rho_threshold: clip threshold for importance weights.
      stop_target_gradients: whether or not to apply stop gradient to targets.

    Returns:
      V-Trace error.
    """
    chex.assert_rank(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_], [1, 1, 1, 1, 1, {0, 1}]
    )
    chex.assert_type(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
        [float, float, float, float, float, float],
    )
    chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

    # Clip importance sampling ratios.
    c_tm1 = jnp.minimum(1.0, rho_tm1) * lambda_
    clipped_rhos_tm1 = jnp.minimum(clip_rho_threshold, rho_tm1)

    # Compute the temporal difference errors.
    td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

    # Work backwards computing the td-errors.
    def _body(acc, xs):
        td_error, discount, c = xs
        acc = td_error + discount * c * acc
        return acc, acc

    _, errors = jax.lax.scan(_body, 0.0, (td_errors, discount_t, c_tm1), reverse=True)

    # Return errors, maybe disabling gradient flow through bootstrap targets.
    return jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(errors + v_tm1) - v_tm1, errors
    )


def vtrace_td_error_and_advantage(
    v_tm1: jax.Array,
    v_t: jax.Array,
    r_t: jax.Array,
    discount_t: jax.Array,
    rho_tm1: jax.Array,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> VTraceOutput:
    """Calculates V-Trace errors and PG advantage from importance weights.

    This functions computes the TD-errors and policy gradient Advantage terms
    as used by the IMPALA distributed actor-critic agent.

    See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
    Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561)

    Args:
      v_tm1: values at time t-1.
      v_t: values at time t.
      r_t: reward at time t.
      discount_t: discount at time t.
      rho_tm1: importance weights at time t-1.
      lambda_: mixing parameter; a scalar or a vector for timesteps t.
      clip_rho_threshold: clip threshold for importance ratios.
      clip_pg_rho_threshold: clip threshold for policy gradient importance ratios.
      stop_target_gradients: whether or not to apply stop gradient to targets.

    Returns:
      a tuple of V-Trace error, policy gradient advantage, and estimated Q-values.
    """
    chex.assert_rank(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_], [1, 1, 1, 1, 1, {0, 1}]
    )
    chex.assert_type(
        [v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
        [float, float, float, float, float, float],
    )
    chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

    # If scalar make into vector.
    lambda_ = jnp.ones_like(discount_t) * lambda_

    errors = vtrace(
        v_tm1,
        v_t,
        r_t,
        discount_t,
        rho_tm1,
        lambda_,
        clip_rho_threshold,
        stop_target_gradients,
    )
    targets_tm1 = errors + v_tm1
    q_bootstrap = jnp.concatenate(
        [
            lambda_[:-1] * targets_tm1[1:] + (1 - lambda_[:-1]) * v_tm1[1:],
            v_t[-1:],
        ],
        axis=0,
    )
    q_estimate = r_t + discount_t * q_bootstrap
    clipped_pg_rho_tm1 = jnp.minimum(clip_pg_rho_threshold, rho_tm1)
    pg_advantages = clipped_pg_rho_tm1 * (q_estimate - v_tm1)
    return VTraceOutput(
        returns=targets_tm1, pg_advantage=pg_advantages, q_estimate=q_estimate
    )


def compute_returns(
    v_tm1: chex.Array,
    rho_tm1: chex.Array,
    batch: Transition,
    config: Porygon2LearnerConfig,
):
    """Train for a single step."""

    valid = jnp.bitwise_not(batch.timestep.env.done)
    rewards = batch.timestep.env.win_reward

    rewards = jnp.concatenate((rewards[1:], rewards[-1:]))
    v_t = jnp.concatenate((v_tm1[1:], v_tm1[-1:]))
    valids = jnp.concatenate((valid[1:], jnp.zeros_like(valid[-1:])))

    discount_t = valids * config.gamma

    with jax.default_device(jax.devices("cpu")[0]):
        return jax.vmap(
            functools.partial(
                vtrace_td_error_and_advantage,
                lambda_=config.lambda_,
                clip_rho_threshold=config.clip_rho_threshold,
                clip_pg_rho_threshold=config.clip_pg_rho_threshold,
            ),
            in_axes=1,
            out_axes=1,
        )(v_tm1, v_t, rewards, discount_t, rho_tm1)
