from enum import Enum
from typing import Sequence

import chex
import haiku as hk
import jax

from jax import numpy as jnp


@chex.dataclass(frozen=True)
class FineTuning:
    """Fine tuning options, aka policy post-processing.

    Even when fully trained, the resulting softmax-based policy may put
    a small probability mass on bad actions. This results in an agent
    waiting for the opponent (itself in self-play) to commit an error.

    To address that the policy is post-processed using:
    - thresholding: any action with probability smaller than self.threshold
      is simply removed from the policy.
    - discretization: the probability values are rounded to the closest
      multiple of 1/self.discretization.

    The post-processing is used on the learner, and thus must be jit-friendly.
    """

    # The learner step after which the policy post processing (aka finetuning)
    # will be enabled when learning. A strictly negative value is equivalent
    # to infinity, ie disables finetuning completely.
    from_learner_steps: int = -1
    # All policy probabilities below `threshold` are zeroed out. Thresholding
    # is disabled if this value is non-positive.
    policy_threshold: float = 0.03
    # Rounds the policy probabilities to the "closest"
    # multiple of 1/`self.discretization`.
    # Discretization is disabled for non-positive values.
    policy_discretization: int = 32

    def __call__(
        self, policy: chex.Array, mask: chex.Array, learner_steps: int
    ) -> chex.Array:
        """A configurable fine tuning of a policy."""
        chex.assert_equal_shape((policy, mask))
        do_finetune = jnp.logical_and(
            self.from_learner_steps >= 0, learner_steps > self.from_learner_steps
        )

        return jnp.where(do_finetune, self.post_process_policy(policy, mask), policy)

    def post_process_policy(
        self,
        policy: chex.Array,
        mask: chex.Array,
    ) -> chex.Array:
        """Unconditionally post process a given masked policy."""
        chex.assert_equal_shape((policy, mask))
        policy = self._threshold(policy, mask)
        policy = self._discretize(policy)
        return policy

    def _threshold(self, policy: chex.Array, mask: chex.Array) -> chex.Array:
        """Remove from the support the actions 'a' where policy(a) < threshold."""
        chex.assert_equal_shape((policy, mask))
        mask = mask * (
            # Values over the threshold.
            (policy >= self.policy_threshold)
            +
            # Degenerate case is when policy is less than threshold *everywhere*.
            # In that case we just keep the policy as-is.
            (jnp.max(policy, axis=-1, keepdims=True) < self.policy_threshold)
        )
        return mask * policy / jnp.sum(mask * policy, axis=-1, keepdims=True)

    def _discretize(self, policy: chex.Array) -> chex.Array:
        """Round all action probabilities to a multiple of 1/self.discretize."""
        if self.policy_discretization <= 0:
            return policy

        # The unbatched/single policy case:
        if len(policy.shape) == 1:
            return self._discretize_single(policy)

        # policy may be [B, A] or [T, B, A], etc. Thus add hk.BatchApply.
        dims = len(policy.shape) - 1

        # TODO(author18): avoid mixing vmap and BatchApply since the two could
        # be folded into either a single BatchApply or a sequence of vmaps, but
        # not the mix.
        vmapped = jax.vmap(self._discretize_single)
        policy = hk.BatchApply(vmapped, num_dims=dims)(policy)

        return policy

    def _discretize_single(self, mu: chex.Array) -> chex.Array:
        """A version of self._discretize but for the unbatched data."""
        # TODO(author18): try to merge _discretize and _discretize_single
        # into one function that handles both batched and unbatched cases.
        if len(mu.shape) == 2:
            mu_ = jnp.squeeze(mu, axis=0)
        else:
            mu_ = mu
        n_actions = mu_.shape[-1]
        roundup = jnp.ceil(mu_ * self.policy_discretization).astype(jnp.int32)
        result = jnp.zeros_like(mu_)
        order = jnp.argsort(-mu_)  # Indices of descending order.
        weight_left = self.policy_discretization

        def f_disc(i, order, roundup, weight_left, result):
            x = jnp.minimum(roundup[order[i]], weight_left)
            result = jax.numpy.where(
                weight_left >= 0, result.at[order[i]].add(x), result
            )
            weight_left -= x
            return i + 1, order, roundup, weight_left, result

        def f_scan_scan(carry, x):
            i, order, roundup, weight_left, result = carry
            i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
                i, order, roundup, weight_left, result
            )
            carry_next = (
                i_next,
                order_next,
                roundup_next,
                weight_left_next,
                result_next,
            )
            return carry_next, x

        (_, _, _, weight_left_next, result_next), _ = jax.lax.scan(
            f_scan_scan,
            init=(jnp.asarray(0), order, roundup, weight_left, result),
            xs=None,
            length=n_actions,
        )

        result_next = jnp.where(
            weight_left_next > 0,
            result_next.at[order[0]].add(weight_left_next),
            result_next,
        )
        if len(mu.shape) == 2:
            result_next = jnp.expand_dims(result_next, axis=0)
        return result_next / self.policy_discretization


@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float
    weight_decay: float


@chex.dataclass(frozen=True)
class NerdConfig:
    """Nerd related params."""

    beta: float = 2.0
    clip: float = 10_000


class StateRepresentation(str, Enum):
    INFO_SET = "info_set"
    OBSERVATION = "observation"


class ActorCriticConfig:
    """Configuration parameters for the RNaDSolver."""

    # Num Training Steps
    num_steps = 1_000_000
    # The games longer than this value are truncated. Must be strictly positive.
    trajectory_max: int = 1000
    # num players in game
    num_players: int = 2
    # The batch size to use when learning/improving parameters.
    batch_size: int = 4
    # The learning rate for `params`.
    learning_rate: float = 0.00005
    # The config related to the ADAM optimizer used for updating `params`.
    adam: AdamConfig = AdamConfig(b1=0, b2=0.999, eps=1e-8, weight_decay=0)
    # All gradients values are clipped to [-clip_gradient, clip_gradient].
    clip_gradient: float = 10_000
    # The "speed" at which `params_target` is following `params`.
    target_network_avg: float = 0.01

    # RNaD algorithm configuration.
    # Entropy schedule configuration. See EntropySchedule class documentation.
    entropy_schedule_repeats: Sequence[int] = (1,)
    entropy_schedule_size: Sequence[int] = (5000,)

    # The weight of the reward regularisation term in RNaD.
    eta_reward_transform: float = 0.0
    gamma: float = 1.0
    nerd: NerdConfig = NerdConfig()
    c_vtrace: float = 1.0

    # Options related to fine tuning of the agent.
    finetune: FineTuning = FineTuning()

    heuristic_loss_coef: float = 0.0
    value_loss_coef: float = 1.0
    policy_loss_coef: float = 1.0
    entropy_loss_coef: float = 0.0

    # The seed that fully controls the randomness.
    seed: int = 42

    do_eval: bool = True
    num_eval_games: int = 200
    generation: int = 3


@chex.dataclass(frozen=True)
class VtraceConfig(ActorCriticConfig):
    # gamma: float = 0.995
    entropy_loss_coef: float = 1e-3


@chex.dataclass(frozen=True)
class RNaDConfig(ActorCriticConfig):
    eta_reward_transform: float = 0.2


@chex.dataclass(frozen=True)
class TeacherForceConfig(ActorCriticConfig):
    adam: AdamConfig = AdamConfig(b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-5)

    heuristic_loss_coef: float = 1.0
    value_loss_coef: float = 0.0
    policy_loss_coef: float = 0.0
    entropy_loss_coef: float = 0.0
