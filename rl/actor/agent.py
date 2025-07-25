import functools
import threading
from typing import Callable, overload

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.interfaces import ActorStep, ModelOutput, TimeStep
from rl.model.utils import Params


class Agent:
    """A stateless agent interface."""

    def __init__(
        self, apply_fn: Callable[[TimeStep], ModelOutput], gpu_lock: threading.Lock
    ):
        """Constructs an Agent object."""

        self._apply_fn = apply_fn
        self._lock = gpu_lock

    def step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep:
        with self._lock:
            return self._step(rng_key, params, timestep)

    @overload
    def _step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(self, rng_key, params: Params, timestep: TimeStep) -> ActorStep:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.

        timestep = TimeStep(
            env=jax.tree.map(lambda t: t[None, None, ...], timestep.env),
            history=jax.tree.map(lambda t: t[:, None, ...], timestep.history),
        )

        model_output = self._apply_fn(params, timestep)
        # Remove the padding from above.
        model_output = jax.tree.map(lambda t: jnp.squeeze(t, axis=(0, 1)), model_output)
        # Sample an action and return.
        action = jax.random.choice(
            rng_key, np.arange(10), shape=(1,), p=model_output.pi
        ).squeeze()
        return ActorStep(action=action, model_output=model_output)

    def unroll(self, params: Params, trajectory: TimeStep) -> ActorStep:
        """Unroll the agent along trajectory."""
        model_output = self._apply_fn(params, trajectory)
        return ActorStep(model_output=model_output)
