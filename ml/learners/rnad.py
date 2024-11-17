import functools
import os
import pickle
from typing import Any, Sequence, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey
from flax import core, struct
from flax.training import train_state

from ml.config import ActorCriticConfig
from ml.func import (
    _player_others,
    get_loss_entropy,
    get_loss_nerd,
    get_loss_v,
    renormalize,
    rnad_v_trace,
)
from ml.utils import Params
from rlenv.env import get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep


@chex.dataclass(frozen=True)
class NerdConfig:
    """Nerd related params."""

    beta: float = 3
    clip: float = 10_000


@chex.dataclass(frozen=True)
class RNaDConfig(ActorCriticConfig):

    # RNaD algorithm configuration.
    # Entropy schedule configuration. See EntropySchedule class documentation.
    entropy_schedule_repeats: Sequence[int] = (1,)
    entropy_schedule_size: Sequence[int] = (10_000,)

    nerd: NerdConfig = NerdConfig()

    learning_rate: float = 3e-5
    eta_reward_transform: float = 0.05


def get_config():
    return RNaDConfig()


class EntropySchedule:
    """An increasing list of steps where the regularisation network is updated.

    Example
      EntropySchedule([3, 5, 10], [2, 4, 1])
      =>   [0, 3, 6, 11, 16, 21, 26, 36]
            | 3 x2 |      5 x4     | 10 x1
    """

    @staticmethod
    def init(sizes: Sequence[int], repeats: Sequence[int]):
        """Constructs a schedule of entropy iterations.

        Args:
          sizes: the list of iteration sizes.
          repeats: the list, parallel to sizes, with the number of times for each
            size from `sizes` to repeat.
        """
        try:
            if len(repeats) != len(sizes):
                raise ValueError("`repeats` must be parallel to `sizes`.")
            if not sizes:
                raise ValueError("`sizes` and `repeats` must not be empty.")
            if any([(repeat <= 0) for repeat in repeats]):
                raise ValueError("All repeat values must be strictly positive")
            if repeats[-1] != 1:
                raise ValueError(
                    "The last value in `repeats` must be equal to 1, "
                    "ince the last iteration size is repeated forever."
                )
        except ValueError as e:
            raise ValueError(
                f"Entropy iteration schedule: repeats ({repeats}) and sizes"
                f" ({sizes})."
            ) from e

        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

        return np.array(schedule, dtype=np.int32)

    @staticmethod
    def update(schedule: np.ndarray, learner_step: int) -> Tuple[float, bool]:
        """Entropy scheduling parameters for a given `learner_step`.

        Args:
          learner_step: The current learning step.

        Returns:
          alpha: The mixing weight (from [0, 1]) of the previous policy with
            the one before for computing the intrinsic reward.
          update_target_net: A boolean indicator for updating the target network
            with the current network.
        """

        # The complexity below is because at some point we might go past
        # the explicit schedule, and then we'd need to just use the last step
        # in the schedule and apply the logic of
        # ((learner_step - last_step) % last_iteration) == 0)

        # The schedule might look like this:
        # X----X-------X--X--X--X--------X
        # learner_step | might be here ^    |
        # or there     ^                    |
        # or even past the schedule         ^

        # We need to deal with two cases below.
        # Instead of going for the complicated conditional, let's just
        # compute both and then do the A * s + B * (1 - s) with s being a bool
        # selector between A and B.

        # 1. assume learner_step is past the schedule,
        #    ie schedule[-1] <= learner_step.
        last_size = schedule[-1] - schedule[-2]
        last_start = (
            schedule[-1] + (learner_step - schedule[-1]) // last_size * last_size
        )
        # 2. assume learner_step is within the schedule.
        start = jnp.amax(schedule * (schedule <= learner_step))
        finish = jnp.amin(
            schedule * (learner_step < schedule),
            initial=schedule[-1],
            where=(learner_step < schedule),
        )
        size = finish - start

        # Now select between the two.
        beyond = schedule[-1] <= learner_step  # Are we past the schedule?
        iteration_start = last_start * beyond + start * (1 - beyond)
        iteration_size = last_size * beyond + size * (1 - beyond)

        update_target_net = jnp.logical_and(
            learner_step > 0,
            jnp.sum(learner_step == iteration_start + iteration_size - 1),
        )
        alpha = jnp.minimum(
            (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0
        )

        return alpha, update_target_net  # pytype: disable=bad-return-type  # jax-types


class TrainState(train_state.TrainState):
    entropy_schedule: np.ndarray

    # params_target: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    params_prev: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    params_prev_: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    learner_steps: int = 0
    actor_steps: int = 0

    alpha: float = 0
    update_target_net: bool = False

    def step_entropy(self):
        alpha, update_target_net = EntropySchedule.update(
            self.entropy_schedule, self.learner_steps
        )
        return self.replace(
            alpha=alpha,
            update_target_net=update_target_net,
        )

    def apply_gradients(self, *, grads, config: RNaDConfig, **kwargs):
        """Applies gradients to parameters and updates EMA parameters."""
        # Apply gradients to update params and opt_state

        state = super().apply_gradients(grads=grads, **kwargs)

        # Update EMA parameters
        # params_target = optax.incremental_update(
        #     new_tensors=state.params,
        #     old_tensors=state.params_target,
        #     step_size=config.target_network_avg,
        # )

        params_prev, params_prev_ = jax.lax.cond(
            self.update_target_net,
            lambda: (self.params, self.params_prev),
            # lambda: (self.params_target, self.params_prev),
            lambda: (self.params_prev, self.params_prev_),
        )

        # Return new state with updated EMA params
        return state.replace(
            # params_target=params_target,
            params_prev=params_prev,
            params_prev_=params_prev_,
        )


def create_train_state(module: nn.Module, rng: PRNGKey, config: RNaDConfig):
    """Creates an initial `TrainState`."""
    ex = get_ex_step()

    params = module.init(rng, ex)
    # params_target = module.init(rng, ex)
    params_prev = module.init(rng, ex)
    params_prev_ = module.init(rng, ex)

    tx = optax.chain(
        optax.clip(config.clip_gradient),
        optax.adamw(
            learning_rate=config.learning_rate,
            eps_root=0.0,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        # params_target=params_target,
        params_prev=params_prev,
        params_prev_=params_prev_,
        entropy_schedule=EntropySchedule.init(
            sizes=config.entropy_schedule_size,
            repeats=config.entropy_schedule_repeats,
        ),
        tx=tx,
    )


def save(state: TrainState):
    with open(os.path.abspath(f"ckpts/ckpt_{state.learner_steps:08}"), "wb") as f:
        pickle.dump(
            dict(
                params=state.params,
                # params_target=state.params_target,
                params_prev=state.params_prev,
                params_prev_=state.params_prev_,
                opt_state=state.opt_state,
                step=state.step,
            ),
            f,
        )


def load(state: TrainState, path: str):
    print(f"loading checkpoint from {path}")
    with open(path, "rb") as f:
        step = pickle.load(f)

    state = state.replace(
        params=step["params"],
        # params_target=step["params_target"],
        params_prev=step["params"],
        params_prev_=step["params"],
        # opt_state=step["opt_state"],
        # step=step["step"],
    )

    return state


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: TimeStep, config: RNaDConfig):
    """Train for a single step."""
    rollout = jax.vmap(jax.vmap(state.apply_fn, (None, 0)), (None, 0))
    pred_prev: ModelOutput = rollout(state.params_prev, batch.env)
    pred_prev_: ModelOutput = rollout(state.params_prev_, batch.env)

    def loss_fn(params: Params):
        rollout_w_grad = jax.vmap(jax.vmap(state.apply_fn, (None, 0)), (None, 0))

        pred: ModelOutput = rollout_w_grad(params, batch.env)
        # pred_targ: ModelOutput = rollout(params_target, batch.env)

        logs = {}

        policy_pprocessed = config.finetune(
            pred.pi, batch.env.legal, state.learner_steps
        )

        # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
        # For the stability reasons, reward changes smoothly between iterations.
        # The mixing between old and new reward transform is a convex combination
        # parametrised by alpha.
        log_policy_reg = pred.log_pi - (
            state.alpha * pred_prev.log_pi + (1 - state.alpha) * pred_prev_.log_pi
        )

        valid = batch.env.valid * (batch.env.legal.sum(axis=-1) > 1)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        action_oh = jax.nn.one_hot(batch.actor.action, batch.actor.policy.shape[-1])

        rewards = batch.actor.win_rewards

        for player in range(config.num_players):
            reward = rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = rnad_v_trace(
                # pred_targ.v,
                pred.v,
                valid,
                batch.env.player_id,
                batch.actor.policy,
                policy_pprocessed,
                log_policy_reg,
                _player_others(batch.env.player_id, valid, player),
                action_oh,
                reward,
                player,
                lambda_=1.0,
                c=config.c_vtrace,
                rho=jnp.inf,
                eta=config.eta_reward_transform,
                gamma=config.gamma,
            )
            v_target_list.append(jax.lax.stop_gradient(v_target_))
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(jax.lax.stop_gradient(policy_target_))

        loss_v = get_loss_v(
            [pred.v] * config.num_players,
            v_target_list,
            has_played_list,
        )

        # is_vector = jnp.expand_dims(
        #     _policy_ratio(policy_pprocessed, ts.actor.policy, action_oh, valid),
        #     axis=-1,
        # )
        is_vector = jnp.expand_dims(jnp.ones_like(batch.env.valid), axis=-1)
        importance_sampling_correction = [is_vector] * config.num_players

        # Uses v-trace to define q-values for Nerd
        loss_nerd = get_loss_nerd(
            [pred.logit] * config.num_players,
            [pred.pi] * config.num_players,
            v_trace_policy_target_list,
            valid,
            batch.env.player_id,
            batch.env.legal,
            importance_sampling_correction,
            clip=config.nerd.clip,
            threshold=config.nerd.beta,
        )

        loss_norm = renormalize(jnp.square(pred.logit).sum(axis=-1), valid)

        loss_entropy = get_loss_entropy(pred.pi, pred.log_pi, batch.env.legal, valid)

        loss = config.value_loss_coef * loss_v + config.policy_loss_coef * loss_nerd

        move_entropy = get_loss_entropy(
            pred.pi[..., :4],
            pred.log_pi[..., :4],
            batch.env.legal[..., :4],
            valid & batch.env.legal[..., :4].any(axis=-1),
        )
        switch_entropy = get_loss_entropy(
            pred.pi[..., 4:],
            pred.log_pi[..., 4:],
            batch.env.legal[..., 4:],
            valid & batch.env.legal[..., 4:].any(axis=-1),
        )

        logs["loss_v"] = loss_v
        logs["loss_nerd"] = loss_nerd
        logs["loss_entropy"] = loss_entropy
        logs["move_entropy"] = move_entropy
        logs["switch_entropy"] = switch_entropy
        logs["loss_norm"] = loss_norm

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    state = state.step_entropy()
    (loss_val, logs), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads, config=config)
    state = state.replace(
        actor_steps=state.actor_steps + batch.env.valid.sum(),
        learner_steps=state.learner_steps + 1,
    )

    valid = batch.env.valid
    lengths = valid.sum(0)

    can_move = batch.env.legal[..., :4].any(axis=-1)
    can_switch = batch.env.legal[..., 4:].any(axis=-1)

    move_ratio = (
        (batch.actor.action < 4).mean(axis=0, where=(can_switch & valid)).mean()
    )
    switch_ratio = (
        (batch.actor.action >= 4).mean(axis=0, where=(can_move & valid)).mean()
    )

    extra_logs = dict(
        actor_steps=valid.sum(),
        loss=loss_val,
        trajectory_length_mean=lengths.mean(),
        trajectory_length_min=lengths.min(),
        trajectory_length_max=lengths.max(),
        early_finish_ratio=batch.actor.win_rewards[..., 0].any(axis=0).mean(),
        gradient_norm=optax.global_norm(grads),
        param_norm=optax.global_norm(state.params),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
    )
    logs.update(extra_logs)

    return state, logs
