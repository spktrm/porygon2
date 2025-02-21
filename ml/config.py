import chex
import numpy as np
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
    policy_threshold: float = 0.05
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
        # policy = self._discretize(policy)
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


@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float
    weight_decay: float


@chex.dataclass(frozen=True)
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
    learning_rate: float = 3e-5
    # The config related to the ADAM optimizer used for updating `params`.
    adam: AdamConfig = AdamConfig(b1=0, b2=0.999, eps=1e-8, weight_decay=0)
    # All gradients values are clipped to [-clip_gradient, clip_gradient].
    clip_gradient: float = 10_000
    # The "speed" at which `params_target` is following `params`.
    target_network_avg: float = 1e-3

    gamma: float = 1.0
    c_vtrace: float = 1.0
    rho_vtrace: float = np.inf

    # Options related to fine tuning of the agent.
    finetune: FineTuning = FineTuning()

    heuristic_loss_coef: float = 0.0
    value_loss_coef: float = 1.0  # div 2 twice (2 players, derivative of **2)
    policy_loss_coef: float = 1.0
    entropy_loss_coef: float = 0.0

    # The seed that fully controls the randomness.
    seed: int = 42

    do_eval: bool = True
    num_eval_games: int = 200
    generation: int = 3
