import functools
import threading
from typing import Callable, overload

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.interfaces import ActorReset, ActorStep, ModelOutput, TimeStep
from rl.model.utils import Params


def threshold_policy(pi: jax.Array, threshold: float = 0.03) -> jax.Array:
    """Thresholds the policy for evaluation."""
    thresholded_pi = jnp.where(pi < threshold, 0.0, pi)
    return thresholded_pi / jnp.sum(thresholded_pi, axis=-1, keepdims=True)


class Agent:
    """A stateless agent interface."""

    def __init__(
        self,
        player_apply_fn: Callable[[Params, TimeStep], ModelOutput],
        builder_apply_fn: Callable[[Params, jax.Array], ActorReset],
        gpu_lock: threading.Lock,
    ):
        """Constructs an Agent object."""

        self._player_apply_fn = player_apply_fn
        self._builder_apply_fn = builder_apply_fn
        self._lock = gpu_lock

    def reset(self, rng_key, params: Params) -> ActorReset:
        with self._lock:
            return self._reset(rng_key, params)

    def step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep:
        with self._lock:
            return self._step(rng_key, params, timestep)

    @overload
    def _reset(self, rng_key, params: Params) -> ActorStep: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset(self, rng_key, params: Params) -> ActorStep:
        return self._builder_apply_fn(params, rng_key, None)

    @overload
    def _step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.

        timestep = TimeStep(
            rng_key=rng_key,
            env=jax.tree.map(lambda t: t[None, None, ...], timestep.env),
            history=jax.tree.map(lambda t: t[:, None, ...], timestep.history),
        )

        model_output = self._player_apply_fn(params, timestep)
        # Remove the padding from above.
        model_output = jax.tree.map(lambda t: jnp.squeeze(t, axis=(0, 1)), model_output)
        # Sample an action and return.
        action = jax.random.choice(
            rng_key, np.arange(10), shape=(1,), p=model_output.pi
        ).squeeze()
        return ActorStep(action=action, model_output=model_output)
