import functools
import threading
from typing import Callable, overload

import jax
import jax.numpy as jnp

from rl.concurrency.lock import NoOpLock
from rl.environment.interfaces import (
    BuilderActorOutput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerAgentOutput,
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
        player_apply_fn: (
            Callable[[Params, PlayerActorInput], PlayerActorOutput] | None
        ) = None,
        builder_apply_fn: (
            Callable[[Params, BuilderEnvOutput], BuilderAgentOutput] | None
        ) = None,
        gpu_lock: threading.Lock = None,
    ):
        """Constructs an Agent object."""

        self._player_apply_fn = player_apply_fn
        self._builder_apply_fn = builder_apply_fn
        self._lock = gpu_lock if gpu_lock is not None else NoOpLock()

    def step_builder(
        self, rng_key: jax.Array, params: Params, actor_input: BuilderEnvOutput
    ) -> BuilderAgentOutput:
        with self._lock:
            return self._step_builder(rng_key, params, actor_input)

    def step_player(
        self, rng_key: jax.Array, params: Params, actor_input: PlayerActorInput
    ) -> PlayerAgentOutput:
        with self._lock:
            return self._step_player(rng_key, params, actor_input)

    @overload
    def _step_builder(
        self, rng_key, params: Params, actor_input: BuilderEnvOutput
    ) -> BuilderAgentOutput: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_builder(
        self, rng_key: jax.Array, params: Params, actor_input: BuilderEnvOutput
    ) -> BuilderAgentOutput:

        actor_input = BuilderEnvOutput(
            tokens=actor_input.tokens[None, None, ...],
            mask=actor_input.mask[None, None, ...],
        )
        actor_output = self._builder_apply_fn(params, actor_input)
        # Remove the padding from above.
        actor_output: BuilderActorOutput = jax.tree.map(
            lambda t: jnp.squeeze(t, axis=(0, 1)), actor_output
        )

        action = jax.random.categorical(rng_key, actor_output.head.logits)
        return BuilderAgentOutput(action=action, actor_output=actor_output)

    @overload
    def _step_player(
        self, rng_key: jax.Array, params: Params, actor_input: PlayerActorInput
    ) -> PlayerAgentOutput: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_player(
        self, rng_key: jax.Array, params: Params, actor_input: PlayerActorInput
    ) -> PlayerAgentOutput:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.

        actor_input = PlayerActorInput(
            env=jax.tree.map(lambda t: t[None, None, ...], actor_input.env),
            history=jax.tree.map(lambda t: t[:, None, ...], actor_input.history),
        )

        actor_output = self._player_apply_fn(params, actor_input)
        # Remove the padding from above.
        actor_output: PlayerActorOutput = jax.tree.map(
            lambda t: jnp.squeeze(t, axis=(0, 1)), actor_output
        )

        # Sample an action and return.
        action_type_key, move_key, switch_key = jax.random.split(rng_key, 3)
        action_type_head = jax.random.categorical(
            action_type_key, actor_output.action_type_head.logits
        )
        move_head = jax.random.categorical(move_key, actor_output.move_head.logits)
        switch_head = jax.random.categorical(
            switch_key, actor_output.switch_head.logits
        )

        return PlayerAgentOutput(
            action_type_head=action_type_head,
            move_head=move_head,
            switch_head=switch_head,
            actor_output=actor_output,
        )
