from typing import Sequence

import chex
import jax
from jax import lax
from jax import numpy as jnp
from jax import tree


def cosine_similarity(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:

    arr1 = arr1 / jnp.linalg.norm(arr1, axis=-1, keepdims=True)
    arr2 = arr2 / jnp.linalg.norm(arr2, axis=-1, keepdims=True)

    return arr1 @ arr2.T


def legal_policy(
    logits: chex.Array, legal_actions: chex.Array, temp: chex.Array = 1
) -> chex.Array:
    """A soft-max policy that respects legal_actions."""
    chex.assert_equal_shape((logits, legal_actions))
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(axis=-1, keepdims=True)
    logits = jnp.where(legal_actions, logits, l_min)
    logits -= logits.max(axis=-1, keepdims=True)
    logits *= legal_actions
    exp_logits = jnp.where(
        legal_actions, jnp.exp(logits / temp), 0
    )  # Illegal actions become 0.
    exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / exp_logits_sum


def legal_log_policy(
    logits: chex.Array, legal_actions: chex.Array, temp: chex.Array = 1
) -> chex.Array:
    """Return the log of the policy on legal action, 0 on illegal action."""
    chex.assert_equal_shape((logits, legal_actions))
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + jnp.log(legal_actions)
    max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = jnp.exp(logits_masked / temp)

    baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = jnp.multiply(legal_actions, (logits - max_legal_logit - baseline))
    return log_policy


def _prenorm_softmax(
    logits: chex.Array, mask: chex.Array, axis: int = -1, eps: float = 1e-5
):
    mask = mask + (mask.sum(axis=axis) == 0)
    mean = jnp.mean(logits, where=mask, axis=axis, keepdims=True)
    variance = jnp.var(logits, where=mask, axis=axis, keepdims=True)
    eps = jax.lax.convert_element_type(eps, variance.dtype)
    inv = jax.lax.rsqrt(variance + eps)
    return inv * (logits - mean)


def _player_others(
    player_ids: chex.Array, valid: chex.Array, player: int
) -> chex.Array:
    """A vector of 1 for the current player and -1 for others.

    Args:
      player_ids: Tensor [...] containing player ids (0 <= player_id < N).
      valid: Tensor [...] containing whether these states are valid.
      player: The player id as int.

    Returns:
      player_other: is 1 for the current player and -1 for others [..., 1].
    """
    chex.assert_equal_shape((player_ids, valid))
    current_player_tensor = (player_ids == player).astype(
        jnp.int32
    )  # pytype: disable=attribute-error  # numpy-scalars

    res = 2 * current_player_tensor - 1
    res = res * valid
    return jnp.expand_dims(res, axis=-1)


def _policy_ratio(
    pi: chex.Array, mu: chex.Array, actions_oh: chex.Array, valid: chex.Array
) -> chex.Array:
    """Returns a ratio of policy pi/mu when selecting action a.

    By convention, this ratio is 1 on non valid states
    Args:
      pi: the policy of shape [..., A].
      mu: the sampling policy of shape [..., A].
      actions_oh: a one-hot encoding of the current actions of shape [..., A].
      valid: 0 if the state is not valid and else 1 of shape [...].

    Returns:
      pi/mu on valid states and 1 otherwise. The shape is the same
      as pi, mu or actions_oh but without the last dimension A.
    """
    chex.assert_equal_shape((pi, mu, actions_oh))
    chex.assert_shape((valid,), actions_oh.shape[:-1])

    def _select_action_prob(pi):
        return jnp.sum(actions_oh * pi, axis=-1, keepdims=False) * valid + (1 - valid)

    pi_actions_prob = _select_action_prob(pi)
    mu_actions_prob = _select_action_prob(mu)
    return pi_actions_prob / mu_actions_prob


def _where(
    pred: chex.Array, true_data: chex.ArrayTree, false_data: chex.ArrayTree
) -> chex.ArrayTree:
    """Similar to jax.where but treats `pred` as a broadcastable prefix."""

    def _where_one(t, f):
        chex.assert_equal_rank((t, f))
        # Expand the dimensions of pred if true_data and false_data are higher rank.
        p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
        return jnp.where(p, t, f)

    return tree.map(_where_one, true_data, false_data)


def _has_played(valid: chex.Array, player_id: chex.Array, player: int) -> chex.Array:
    """Compute a mask of states which have a next state in the sequence."""
    chex.assert_equal_shape((valid, player_id))

    def _loop_has_played(carry, x):
        valid, player_id = x
        chex.assert_equal_shape((valid, player_id))

        our_res = jnp.ones_like(player_id)
        opp_res = carry
        reset_res = jnp.zeros_like(carry)

        our_carry = carry
        opp_carry = carry
        reset_carry = jnp.zeros_like(player_id)

        # pyformat: disable
        return _where(
            valid,
            _where((player_id == player), (our_carry, our_res), (opp_carry, opp_res)),
            (reset_carry, reset_res),
        )
        # pyformat: enable

    _, result = lax.scan(
        f=_loop_has_played,
        init=jnp.zeros_like(player_id[-1]),
        xs=(valid, player_id),
        reverse=True,
    )
    return result


def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def get_loss_pg(
    log_pi_list: Sequence[chex.Array],
    policy_list: Sequence[chex.Array],
    q_vr_list: Sequence[chex.Array],
    valid: chex.Array,
    player_ids: Sequence[chex.Array],
    legal_actions: Sequence[chex.Array],
    actions_oh: chex.Array,
) -> chex.Array:
    """Define the nerd loss."""
    loss_pi_list = []

    for k, (log_pi, pi, q_vr) in enumerate(zip(log_pi_list, policy_list, q_vr_list)):

        adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True)
        adv_pi = adv_pi  # importance sampling correction
        adv_pi = jnp.clip(adv_pi, min=-100, max=100)
        adv_pi = lax.stop_gradient(adv_pi)

        pg_loss = -(actions_oh * log_pi * adv_pi).mean(axis=-1, where=legal_actions)
        pg_loss = renormalize(pg_loss, valid * (player_ids == k))

        loss_pi_list.append(pg_loss)
    return sum(loss_pi_list)


def get_loss_entropy(policy: chex.Array, valid: chex.Array) -> chex.Array:
    policy_sum = policy.sum(axis=-1, keepdims=True)
    policy_sum = policy_sum + (policy_sum == 0)
    policy = policy / policy_sum
    log_policy = jnp.where(policy > 0, jnp.log(policy), 1)
    loss_entropy = (policy * log_policy).sum(-1)
    return renormalize(loss_entropy, valid)


def get_average_logit_value(logit: chex.Array, legal: chex.Array, valid: chex.Array):
    logit = logit - logit.mean(keepdims=True, axis=-1, where=legal)
    return renormalize(jnp.abs(logit).mean(axis=-1, where=legal), valid)
