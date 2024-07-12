import jax
import optax
import functools

import numpy as np
import jax.numpy as jnp
import jax.tree_util as tree
import flax.linen as nn

from typing import Sequence, Tuple

from ml.config import RNaDConfig
from ml.utils import Params
from ml.func import (
    get_loss_entropy,
    get_loss_heuristic,
    get_loss_nerd,
    get_loss_v,
    v_trace,
    _player_others,
    _policy_ratio,
)

from rlenv.env import get_ex_step
from rlenv.interfaces import TimeStep


class EntropySchedule:
    """An increasing list of steps where the regularisation network is updated.

    Example
      EntropySchedule([3, 5, 10], [2, 4, 1])
      =>   [0, 3, 6, 11, 16, 21, 26, 36]
            | 3 x2 |      5 x4     | 10 x1
    """

    def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
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

        self.schedule = np.array(schedule, dtype=np.int32)

    def __call__(self, learner_step: int) -> Tuple[float, bool]:
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
        last_size = self.schedule[-1] - self.schedule[-2]
        last_start = (
            self.schedule[-1]
            + (learner_step - self.schedule[-1]) // last_size * last_size
        )
        # 2. assume learner_step is within the schedule.
        start = jnp.amax(self.schedule * (self.schedule <= learner_step))
        finish = jnp.amin(
            self.schedule * (learner_step < self.schedule),
            initial=self.schedule[-1],
            where=(learner_step < self.schedule),
        )
        size = finish - start

        # Now select between the two.
        beyond = self.schedule[-1] <= learner_step  # Are we past the schedule?
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


class Learner:
    def __init__(self, network: nn.Module, config: RNaDConfig):
        self.config = config
        self.network = network

        self.learner_steps = 0
        self.actor_steps = 0

        ex = get_ex_step()
        key = jax.random.key(0)

        # The machinery related to updating parameters/learner.
        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats,
        )
        self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=True)

        # Params
        self.params = network.init(key, ex)
        self.params_target = network.init(key, ex)
        self.params_prev = network.init(key, ex)
        self.params_prev_ = network.init(key, ex)

        # Parameter optimizers.
        self.optimizer = optax.chain(
            optax.clip(self.config.clip_gradient),
            optax.adamw(
                learning_rate=self.config.learning_rate,
                eps_root=0.0,
                b1=self.config.adam.b1,
                b2=self.config.adam.b2,
                eps=self.config.adam.eps,
                weight_decay=self.config.adam.weight_decay,
            ),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.optimizer_target = optax.sgd(self.config.target_network_avg)
        self.optimizer_target_state = self.optimizer_target.init(self.params_target)

    def loss(
        self,
        params: Params,
        params_target: Params,
        params_prev: Params,
        params_prev_: Params,
        ts: TimeStep,
        alpha: float,
        learner_steps: int,
    ) -> float:
        rollout = jax.vmap(jax.vmap(self.network.apply, (None, 0), 0), (None, 0), 0)
        pi, v, log_pi, logit = rollout(params, ts.env)

        policy_pprocessed = self.config.finetune(pi, ts.env.legal, learner_steps)

        _, v_target, _, _ = rollout(params_target, ts.env)

        if self.config.eta_reward_transform > 0:
            _, _, log_pi_prev, _ = rollout(params_prev, ts.env)
            _, _, log_pi_prev_, _ = rollout(params_prev_, ts.env)
            # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
            # For the stability reasons, reward changes smoothly between iterations.
            # The mixing between old and new reward transform is a convex combination
            # parametrised by alpha.
            log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)
        else:
            log_policy_reg = jnp.zeros_like(log_pi)

        valid = ts.env.valid * (ts.env.legal.sum(axis=-1) > 1)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        for player in range(self.config.num_players):
            reward = ts.actor.rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = v_trace(
                v_target,
                valid,
                ts.env.player_id,
                ts.actor.policy,
                policy_pprocessed,
                log_policy_reg,
                _player_others(ts.env.player_id, valid, player),
                ts.actor.action_oh,
                reward,
                player,
                lambda_=1.0,
                c=self.config.c_vtrace,
                rho=jnp.inf,
                eta=self.config.eta_reward_transform,
                gamma=self.config.gamma,
            )
            v_target_list.append(v_target_)
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(policy_target_)

        loss_v = get_loss_v(
            [v] * self.config.num_players, v_target_list, has_played_list
        )

        is_vector = jnp.expand_dims(
            _policy_ratio(
                policy_pprocessed, ts.actor.policy, ts.actor.action_oh, valid
            ),
            axis=-1,
        )
        importance_sampling_correction = [is_vector] * self.config.num_players

        # Uses v-trace to define q-values for Nerd
        loss_nerd = get_loss_nerd(
            [logit] * self.config.num_players,
            [pi] * self.config.num_players,
            v_trace_policy_target_list,
            valid,
            ts.env.player_id,
            ts.env.legal,
            importance_sampling_correction,
            clip=self.config.nerd.clip,
            threshold=self.config.nerd.beta,
        )

        # loss_heuristic = get_loss_heuristic(
        #     log_pi,
        #     valid,
        #     ts.actor.action_oh,
        #     ts.env.heuristic_move,
        #     ts.env.heuristic_switch,
        #     ts.env.legal,
        # )

        # loss_entropy = get_loss_entropy(pi, log_pi, ts.env.legal, valid)

        loss = (
            self.config.value_loss_coef * loss_v
            + self.config.policy_loss_coef * loss_nerd
            # + self.config.heuristic_loss_coef * loss_heuristic
            # + self.config.entropy_loss_coef * loss_entropy
        )

        return loss, (
            dict(
                loss_v=loss_v,
                loss_nerd=loss_nerd,
                # loss_heuristic=loss_heuristic,
                # loss_entropy=loss_entropy,
            ),
        )  # pytype: disable=bad-return-type  # numpy-scalars

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_parameters(
        self,
        params: Params,
        params_target: Params,
        params_prev: Params,
        params_prev_: Params,
        optimizer_state: optax.OptState,
        optimizer_target_state: optax.OptState,
        timestep: TimeStep,
        alpha: float,
        learner_steps: int,
        update_target_net: bool,
    ):
        """A jitted pure-functional part of the `step`."""
        (loss_val, (info,)), grad = self._loss_and_grad(
            params,
            params_target,
            params_prev,
            params_prev_,
            timestep,
            alpha,
            learner_steps,
        )

        # Update `params`` using the computed gradient.
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        # Update `params_target` towards `params`.
        target_updates, optimizer_target_state = self.optimizer_target.update(
            tree.tree_map(lambda a, b: a - b, params_target, params),
            optimizer_target_state,
        )
        params_target = optax.apply_updates(params_target, target_updates)

        # Rolls forward the prev and prev_ params if update_target_net is 1.
        params_prev, params_prev_ = jax.lax.cond(
            update_target_net,
            lambda: (params_target, params_prev),
            lambda: (params_prev, params_prev_),
        )

        lengths = (timestep.env.valid * (timestep.env.legal.sum(axis=-1) > 1)).sum(0)

        logs = {
            "loss": loss_val,
            **dict(info.items()),
            "trajectory_length_mean": lengths.mean(),
            "trajectory_length_min": lengths.min(),
            "trajectory_length_max": lengths.max(),
            "alpha": alpha,
        }

        return (
            params,
            params_target,
            params_prev,
            params_prev_,
            optimizer_state,
            optimizer_target_state,
        ), logs

    def step(self, timestep: TimeStep):
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)

        (
            self.params,
            self.params_target,
            self.params_prev,
            self.params_prev_,
            self.optimizer_state,
            self.optimizer_target_state,
        ), logs = self.update_parameters(
            self.params,
            self.params_target,
            self.params_prev,
            self.params_prev_,
            self.optimizer_state,
            self.optimizer_target_state,
            timestep,
            alpha,
            self.learner_steps,
            update_target_net,
        )

        self.learner_steps += 1
        self.actor_steps += timestep.env.valid.sum()
        logs.update(
            {
                "actor_steps": self.actor_steps,
                "learner_steps": self.learner_steps,
            }
        )
        return logs
