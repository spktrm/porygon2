import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from flax import core, struct
from flax.training import train_state

from ml.config import ActorCriticConfig
from ml.func import get_loss_entropy, get_loss_pg, get_loss_v, reg_v_trace, renormalize
from ml.utils import Params
from rlenv.env import get_ex_step
from rlenv.interfaces import ModelOutput, TimeStep


class TrainState(train_state.TrainState):

    params_target: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    learner_steps: int = 0
    actor_steps: int = 0

    def apply_gradients(self, *, grads, config: ActorCriticConfig, **kwargs):
        """Applies gradients to parameters and updates EMA parameters."""
        # Apply gradients to update params and opt_state

        state = super().apply_gradients(grads=grads, **kwargs)

        # Update EMA parameters
        params_target = optax.incremental_update(
            new_tensors=state.params,
            old_tensors=state.params_target,
            step_size=config.target_network_avg,
        )

        # Return new state with updated EMA params
        return state.replace(params_target=params_target)


def create_train_state(module: nn.Module, rng: PRNGKey, config: ActorCriticConfig):
    """Creates an initial `TrainState`."""
    ex = get_ex_step()

    params = module.init(rng, ex)
    params_target = module.init(rng, ex)

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
        params_target=params_target,
        tx=tx,
    )


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: TimeStep, config: ActorCriticConfig):
    """Train for a single step."""

    def loss_fn(
        params: Params,
        params_target: Params,
        learner_steps: int,
    ):
        rollout = jax.vmap(jax.vmap(state.apply_fn, (None, 0)), (None, 0))

        pred: ModelOutput = rollout(params, batch.env)
        pred_targ: ModelOutput = rollout(params_target, batch.env)

        logs = {}

        policy_pprocessed = config.finetune(pred.pi, batch.env.legal, learner_steps)

        # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
        # For the stability reasons, reward changes smoothly between iterations.
        # The mixing between old and new reward transform is a convex combination
        # parametrised by alpha.

        valid = batch.env.valid * (batch.env.legal.sum(axis=-1) > 1)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        action_oh = jax.nn.one_hot(batch.actor.action, batch.actor.policy.shape[-1])

        rewards = batch.actor.win_rewards

        for player in range(config.num_players):
            reward = rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = reg_v_trace(
                pred_targ.v,
                valid,
                batch.env.player_id,
                batch.actor.policy,
                policy_pprocessed,
                action_oh,
                reward,
                player,
                lambda_=1.0,
                c=config.c_vtrace,
                rho=jnp.inf,
                gamma=config.gamma,
            )
            v_target_list.append(v_target_)
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(policy_target_)

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

        loss_nerd = get_loss_pg(
            [pred.log_pi] * config.num_players,
            v_trace_policy_target_list,
            action_oh,
            valid,
            batch.env.player_id,
            importance_sampling_correction,
        )

        loss_entropy = get_loss_entropy(pred.pi, pred.log_pi, batch.env.legal, valid)

        ssl_loss = renormalize(pred.ssl_loss, batch.env.valid)
        loss = (
            config.value_loss_coef * loss_v
            + config.policy_loss_coef * loss_nerd
            + config.entropy_loss_coef * loss_entropy
            + ssl_loss
        )

        logs["loss_v"] = loss_v
        logs["loss_nerd"] = loss_nerd
        logs["loss_entropy"] = loss_entropy
        logs["ssl_loss"] = ssl_loss

        return loss, logs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(
        state.params,
        state.params_target,
        state.learner_steps,
    )
    state = state.apply_gradients(grads=grads, config=config)
    state = state.replace(
        actor_steps=state.actor_steps + batch.env.valid.sum(),
        learner_steps=state.learner_steps + 1,
    )

    valid = batch.env.valid
    lengths = valid.sum(0)

    can_move = batch.env.legal[..., :4].any(axis=-1)
    can_switch = batch.env.legal[..., 4:].any(axis=-1)

    move_ratio = renormalize(batch.actor.action < 4, can_switch & valid)
    switch_ratio = renormalize(batch.actor.action >= 4, can_move & valid)

    extra_logs = dict(
        loss=loss_val,
        trajectory_length_mean=lengths.mean(),
        trajectory_length_min=lengths.min(),
        trajectory_length_max=lengths.max(),
        gradient_norm=optax.global_norm(grads),
        param_norm=optax.global_norm(state.params),
        move_ratio=move_ratio,
        switch_ratio=switch_ratio,
    )
    logs.update(extra_logs)

    return state, logs
