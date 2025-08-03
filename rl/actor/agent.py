import functools
import threading
from typing import Callable, overload

import jax
import jax.numpy as jnp

from rl.environment.interfaces import (
    BuilderAgentOutput,
    PlayerAgentOutput,
    BuilderEnvOutput,
    PlayerActorOutput,
    PlayerActorInput,
)
from rl.model.utils import Params


def threshold_policy(pi: jax.Array, threshold: float = 0.03) -> jax.Array:
    """Thresholds the policy for evaluation."""
    thresholded_pi = jnp.where(pi < threshold, 0.0, pi)
    return thresholded_pi / jnp.sum(thresholded_pi, axis=-1, keepdims=True)


class Agent:
    """A stateless agent interface."""

    def __init__(
        self,
        player_apply_fn: Callable[[Params, PlayerActorInput], PlayerActorOutput],
        builder_apply_fn: Callable[[Params, BuilderEnvOutput], BuilderAgentOutput],
        gpu_lock: threading.Lock,
    ):
        """Constructs an Agent object."""

        self._player_apply_fn = player_apply_fn
        self._builder_apply_fn = builder_apply_fn
        self._lock = gpu_lock

    def step_builder(self, rng_key, params: Params) -> BuilderAgentOutput:
        with self._lock:
            return self._step_builder(rng_key, params)

    def step_player(
        self, rng_key, params: Params, timestep: PlayerActorInput
    ) -> PlayerAgentOutput:
        with self._lock:
            return self._step_player(rng_key, params, timestep)

    @overload
    def _step_builder(self, rng_key, params: Params) -> PlayerAgentOutput: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_builder(self, rng_key, params: Params) -> PlayerAgentOutput:
        return self._builder_apply_fn(params, rng_key, None)

    @overload
    def _step_player(
        self, rng_key, params: Params, timestep: PlayerActorInput
    ) -> PlayerAgentOutput: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_player(
        self, rng_key, params: Params, timestep: PlayerActorInput
    ) -> PlayerAgentOutput:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.

        timestep = PlayerActorInput(
            env=jax.tree.map(lambda t: t[None, None, ...], timestep.env),
            history=jax.tree.map(lambda t: t[:, None, ...], timestep.history),
        )

        model_output = self._player_apply_fn(params, timestep)
        # Remove the padding from above.
        model_output: PlayerActorOutput = jax.tree.map(
            lambda t: jnp.squeeze(t, axis=(0, 1)), model_output
        )

        # Sample an action and return.
        action_type_key, move_key, switch_key = jax.random.split(rng_key, 3)
        action_type_head = jax.random.categorical(
            action_type_key, model_output.action_type_head.logits
        )
        move_head = jax.random.categorical(move_key, model_output.move_head.logits)
        switch_head = jax.random.categorical(
            switch_key, model_output.switch_head.logits
        )

        return PlayerAgentOutput(
            action_type_head=action_type_head,
            move_head=move_head,
            switch_head=switch_head,
            model_output=model_output,
        )
