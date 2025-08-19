import functools
import threading
from typing import Callable, Sequence, overload

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
    *,
    shape: Sequence[int] | None = None,
    replace: bool = True,
) -> jax.Array:
    """Samples an action from the logits using the provided mask."""
    # Must use float32 here since bfloat16 does some weird things.
    logits = logits.astype(jnp.float32) / sampling_config.temp
    policy = jax.nn.softmax(logits, axis=-1)
    if sampling_config.min_p is not None:
        logits = jnp.where(
            policy >= (policy.max() * sampling_config.min_p), logits, BIAS_VALUE
        )
    return jax.random.choice(
        rng_key,
        policy.shape[-1],
        shape=shape or (1,),
        replace=replace,
        p=policy,
        axis=-1,
    ).squeeze()


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

        player_sampling_config = player_sampling_config or SamplingConfig()
        builder_sampling_config = builder_sampling_config or SamplingConfig()

        self._player_sample_fn = functools.partial(
            sample_action, sampling_config=player_sampling_config
        )
        self._builder_sample_fn = functools.partial(
            sample_action, sampling_config=builder_sampling_config
        )

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

        actor_input: BuilderEnvOutput = jax.tree.map(
            lambda x: x[None, None, ...], actor_input
        )

        actor_output = self._builder_apply_fn(params, actor_input)
        # Remove the padding from above.
        actor_output: BuilderActorOutput = jax.tree.map(
            lambda t: jnp.squeeze(t, axis=(0, 1)), actor_output
        )

        species_key, packed_set_key = jax.random.split(rng_key, 2)

        species = self._builder_sample_fn(species_key, actor_output.species_logits)
        packed_set = self._builder_sample_fn(
            packed_set_key, actor_output.packed_set_logits
        )

        return BuilderAgentOutput(
            species=species, packed_set=packed_set, actor_output=actor_output
        )

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
        action_type_head = self._player_sample_fn(
            action_type_key, actor_output.action_type_logits
        )
        move_head = self._player_sample_fn(move_key, actor_output.move_logits)
        switch_head = self._player_sample_fn(switch_key, actor_output.switch_logits)
        wildcard_head = self._player_sample_fn(
            wildcard_key, actor_output.wildcard_logits[move_head]
        )

        return PlayerAgentOutput(
            action_type=action_type_head,
            move_slot=move_head,
            switch_slot=switch_head,
            wildcard_slot=wildcard_head,
            actor_output=actor_output,
        )
