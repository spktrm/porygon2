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
    SamplingConfig,
)
from rl.model.utils import BIAS_VALUE, Params


def sample_action(
    rng_key: jax.Array,
    logits: jax.Array,
    sampling_config: SamplingConfig = SamplingConfig(),
) -> jax.Array:
    """Samples an action from the logits using the provided mask."""
    # Must use float32 here since bfloat16 does some weird things.
    logits = logits.astype(jnp.float32) / sampling_config.temp
    if sampling_config.min_p is not None:
        policy = jax.nn.softmax(logits, axis=-1)
        logits = jnp.where(
            policy >= (policy.max() * sampling_config.min_p), logits, BIAS_VALUE
        )
    return jax.random.categorical(rng_key, logits, mode="high")


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
        player_sampling_config: SamplingConfig | None = None,
        builder_sampling_config: SamplingConfig | None = None,
    ):
        """Constructs an Agent object."""

        self._player_apply_fn = player_apply_fn
        self._builder_apply_fn = builder_apply_fn
        self._lock = gpu_lock if gpu_lock is not None else NoOpLock()
        self._player_sampling_config = player_sampling_config or SamplingConfig()
        self._builder_sampling_config = builder_sampling_config or SamplingConfig()

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

        action = sample_action(
            rng_key, actor_output.logits, self._builder_sampling_config
        )
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
        action_type_key, move_key, switch_key, wildcard_key = jax.random.split(
            rng_key, 4
        )
        action_type_head = sample_action(
            action_type_key,
            actor_output.action_type_logits,
            self._player_sampling_config,
        )
        move_head = sample_action(
            move_key,
            actor_output.move_logits,
            self._player_sampling_config,
        )
        switch_head = sample_action(
            switch_key,
            actor_output.switch_logits,
            self._player_sampling_config,
        )
        wildcard_head = sample_action(
            wildcard_key,
            actor_output.wildcard_logits[move_head],
            self._player_sampling_config,
        )

        return PlayerAgentOutput(
            action_type_head=action_type_head,
            move_head=move_head,
            switch_head=switch_head,
            wildcard_head=wildcard_head,
            actor_output=actor_output,
        )
