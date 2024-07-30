import jax
import chex
import jax.scipy.special as jsp

from typing import Any, Sequence, Tuple

from jax import lax
from jax import numpy as jnp
from jax import tree_util as tree


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

    return tree.tree_map(_where_one, true_data, false_data)


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


# V-Trace
#
# Custom implementation of VTrace to handle trajectories having a mix of
# different player steps. The standard rlax.vtrace can't be applied here
# out of the box because a trajectory could look like '121211221122'.


def v_trace(
    v: chex.Array,
    valid: chex.Array,
    player_id: chex.Array,
    acting_policy: chex.Array,
    merged_policy: chex.Array,
    merged_log_policy: chex.Array,
    player_others: chex.Array,
    actions_oh: chex.Array,
    reward: chex.Array,
    player: int,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
    gamma: float,
) -> Tuple[Any, Any, Any]:
    """Custom VTrace for trajectories with a mix of different player steps."""

    has_played = _has_played(valid, player_id, player)

    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
    inv_mu = _policy_ratio(
        jnp.ones_like(merged_policy), acting_policy, actions_oh, valid
    )
    num_legal_actions = (acting_policy > 0).sum(axis=-1).clip(min=1)
    eta_reg_entropy = (
        -eta
        * jnp.sum(merged_policy * merged_log_policy, axis=-1)
        * jax.lax.rsqrt(num_legal_actions.astype(float))
        * jnp.squeeze(player_others, axis=-1)
    )
    eta_log_policy = -eta * merged_log_policy * player_others

    @chex.dataclass(frozen=True)
    class LoopVTraceCarry:
        """The carry of the v-trace scan loop."""

        reward: chex.Array
        # The cumulated reward until the end of the episode. Uncorrected (v-trace).
        # Gamma discounted and includes eta_reg_entropy.
        reward_uncorrected: chex.Array
        next_value: chex.Array
        next_v_target: chex.Array
        importance_sampling: chex.Array

    init_state_v_trace = LoopVTraceCarry(
        reward=jnp.zeros_like(reward[-1]),
        reward_uncorrected=jnp.zeros_like(reward[-1]),
        next_value=jnp.zeros_like(v[-1]),
        next_v_target=jnp.zeros_like(v[-1]),
        importance_sampling=jnp.ones_like(policy_ratio[-1]),
    )

    def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
        (
            cs,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mu,
            actions_oh,
            eta_log_policy,
        ) = x

        reward_uncorrected = reward + gamma * carry.reward_uncorrected + eta_reg_entropy
        discounted_reward = reward + gamma * carry.reward

        # V-target:
        our_v_target = (
            v
            + jnp.expand_dims(jnp.minimum(rho, cs * carry.importance_sampling), axis=-1)
            * (
                jnp.expand_dims(reward_uncorrected, axis=-1)
                + gamma * carry.next_value
                - v
            )
            + lambda_
            * jnp.expand_dims(jnp.minimum(c, cs * carry.importance_sampling), axis=-1)
            * gamma
            * (carry.next_v_target - carry.next_value)
        )

        opp_v_target = jnp.zeros_like(our_v_target)
        reset_v_target = jnp.zeros_like(our_v_target)

        # Learning output:
        our_learning_output = (
            v
            + eta_log_policy  # value
            + actions_oh  # regularisation
            * jnp.expand_dims(inv_mu, axis=-1)
            * (
                jnp.expand_dims(discounted_reward, axis=-1)
                + gamma
                * jnp.expand_dims(carry.importance_sampling, axis=-1)
                * carry.next_v_target
                - v
            )
        )

        opp_learning_output = jnp.zeros_like(our_learning_output)
        reset_learning_output = jnp.zeros_like(our_learning_output)

        # State carry:
        our_carry = LoopVTraceCarry(
            reward=jnp.zeros_like(carry.reward),
            next_value=v,
            next_v_target=our_v_target,
            reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
            importance_sampling=jnp.ones_like(carry.importance_sampling),
        )
        opp_carry = LoopVTraceCarry(
            reward=eta_reg_entropy + cs * discounted_reward,
            reward_uncorrected=reward_uncorrected,
            next_value=gamma * carry.next_value,
            next_v_target=gamma * carry.next_v_target,
            importance_sampling=cs * carry.importance_sampling,
        )
        reset_carry = init_state_v_trace

        # Invalid turn: init_state_v_trace and (zero target, learning_output)
        # pyformat: disable
        return _where(
            valid,  # pytype: disable=bad-return-type  # numpy-scalars
            _where(
                (player_id == player),
                (our_carry, (our_v_target, our_learning_output)),
                (opp_carry, (opp_v_target, opp_learning_output)),
            ),
            (reset_carry, (reset_v_target, reset_learning_output)),
        )
        # pyformat: enable

    _, (v_target, learning_output) = lax.scan(
        f=_loop_v_trace,
        init=init_state_v_trace,
        xs=(
            policy_ratio,
            player_id,
            v,
            reward,
            eta_reg_entropy,
            valid,
            inv_mu,
            actions_oh,
            eta_log_policy,
        ),
        reverse=True,
    )

    return v_target, has_played, learning_output


def get_loss_v(
    v_list: Sequence[chex.Array],
    v_target_list: Sequence[chex.Array],
    mask_list: Sequence[chex.Array],
) -> chex.Array:
    """Define the loss function for the critic."""
    chex.assert_trees_all_equal_shapes(v_list, v_target_list)
    # v_list and v_target_list come with a degenerate trailing dimension,
    # which mask_list tensors do not have.
    chex.assert_shape(mask_list, v_list[0].shape[:-1])
    loss_v_list = []
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        assert v_n.shape[0] == v_target.shape[0]

        loss_v = (
            jnp.expand_dims(mask, axis=-1) * (v_n - lax.stop_gradient(v_target)) ** 2
        )
        normalization = jnp.sum(mask)
        loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0))

        loss_v_list.append(loss_v)
    return sum(loss_v_list)


def apply_force_with_threshold(
    decision_outputs: chex.Array,
    force: chex.Array,
    threshold: float,
    threshold_center: chex.Array,
) -> chex.Array:
    """Apply the force with below a given threshold."""
    chex.assert_equal_shape((decision_outputs, force, threshold_center))
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = jnp.minimum(force, 0.0)
    force_positive = jnp.maximum(force, 0.0)
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * lax.stop_gradient(clipped_force)


def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def get_loss_nerd(
    logit_list: Sequence[chex.Array],
    policy_list: Sequence[chex.Array],
    q_vr_list: Sequence[chex.Array],
    valid: chex.Array,
    player_ids: Sequence[chex.Array],
    legal_actions: chex.Array,
    importance_sampling_correction: Sequence[chex.Array],
    clip: float = 100,
    threshold: float = 2,
) -> chex.Array:
    """Define the nerd loss."""
    assert isinstance(importance_sampling_correction, list)
    loss_pi_list = []
    num_valid_actions = jnp.sum(legal_actions, axis=-1, keepdims=True)

    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        assert logit_pi.shape[0] == q_vr.shape[0]
        # loss policy
        adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True)
        adv_pi = is_c * adv_pi  # importance sampling correction
        adv_pi = jnp.clip(adv_pi, a_min=-clip, a_max=clip)
        adv_pi = lax.stop_gradient(adv_pi)

        valid_logit_sum = jnp.sum(logit_pi * legal_actions, axis=-1, keepdims=True)
        mean_logit = valid_logit_sum / num_valid_actions

        # Subtract only the mean of the valid logits
        logits = logit_pi - mean_logit

        threshold_center = jnp.zeros_like(logits)

        nerd_loss = jnp.sum(
            legal_actions
            * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
            axis=-1,
        )
        nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
        loss_pi_list.append(nerd_loss)
    return sum(loss_pi_list)


def get_loss_recon(loss: chex.Array, valid: chex.Array) -> chex.Array:
    loss = -loss * valid
    return loss.sum() / valid.sum()


def get_loss_heuristic(
    log_pi: chex.Array,
    valid: chex.Array,
    heuristic_action: chex.Array,
    legal: chex.Array,
) -> chex.Array:

    num_actions = log_pi.shape[-1]
    target_probs = jnp.where(
        (heuristic_action < 0)[..., None],
        legal / legal.sum(axis=-1, keepdims=True),
        jax.nn.one_hot(heuristic_action, num_actions),
    )

    xentropy = log_pi * target_probs
    xentropy = -xentropy.sum(-1)
    return renormalize(xentropy, valid)


def get_loss_entropy(
    policy: chex.Array,
    log_policy: chex.Array,
    legal: chex.Array,
    valid: chex.Array,
) -> chex.Array:
    loss_entropy = (policy * log_policy).sum(-1)
    # num_legal_actions = legal.sum(-1)
    # denom = jnp.log(num_legal_actions)
    # denom = jnp.where(num_legal_actions <= 1, 1, denom)
    # loss_entropy = loss_entropy / denom

    return renormalize(loss_entropy, valid)
