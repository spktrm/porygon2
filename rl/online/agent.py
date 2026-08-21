import functools
from _thread import LockType
from collections.abc import Callable
from contextlib import nullcontext
from typing import overload

import jax
import jax.numpy as jnp

from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerAgentOutput,
)
from rl.model.heads import HeadParams
from rl.model.utils import Params


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
        gpu_lock: LockType | None = None,
        player_head_params: HeadParams = HeadParams(),
        builder_head_params: HeadParams = HeadParams(),
    ):
        """Constructs an Agent object."""
        if player_apply_fn is None and builder_apply_fn is None:
            raise ValueError(
                "At least one of player_apply_fn or builder_apply_fn must be provided."
            )

        self.player_head_params = player_head_params
        self.builder_head_params = builder_head_params

        dummy_func = lambda *args, **kwargs: None
        # head_params is a per-CALL argument of the jitted steps (a traced
        # pytree of scalars, one trace regardless of value), not baked in
        # via functools.partial: the exploration ladder samples a fresh
        # continuous epsilon every game (see PlayerActor), which a
        # baked-in python float would turn into one recompile per value.
        self._player_apply_fn = player_apply_fn or dummy_func
        self._builder_apply_fn = builder_apply_fn or dummy_func
        self._gpu_lock = gpu_lock or nullcontext()

    def step_builder(
        self, rng_key: jax.Array, params: Params, actor_input: BuilderEnvOutput
    ) -> BuilderAgentOutput:
        with self._gpu_lock:
            return self._step_builder(
                rng_key, params, actor_input, self.builder_head_params
            )

    def step_player(
        self,
        rng_key: jax.Array,
        params: Params,
        actor_input: PlayerActorInput,
        head_params: HeadParams | None = None,
    ) -> PlayerAgentOutput:
        """head_params overrides this Agent's default for one call — the
        exploration ladder's per-game sampled epsilon (HeadParams.mix)."""
        with self._gpu_lock:
            return self._step_player(
                rng_key,
                params,
                actor_input,
                self.player_head_params if head_params is None else head_params,
            )

    @overload
    def _step_builder(
        self,
        rng_key,
        params: Params,
        actor_input: BuilderEnvOutput,
        head_params: HeadParams = ...,
    ) -> BuilderAgentOutput: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_builder(
        self,
        rng_key: jax.Array,
        params: Params,
        actor_input: BuilderActorInput,
        head_params: HeadParams = HeadParams(),
    ) -> BuilderAgentOutput:

        actor_input: BuilderActorInput = BuilderActorInput(
            env=jax.tree.map(lambda x: x[None, ...], actor_input.env),
            history=jax.tree.map(lambda x: x[:, ...], actor_input.history),
        )

        actor_output = self._builder_apply_fn(
            params,
            actor_input,
            BuilderActorOutput(),
            head_params=head_params,
            rngs={"sampling": rng_key},
        )
        # Remove the padding from above.
        actor_output: BuilderActorOutput = jax.tree.map(
            lambda t: jnp.squeeze(t, axis=0), actor_output
        )

        return BuilderAgentOutput(actor_output=actor_output)

    @overload
    def _step_player(
        self,
        rng_key: jax.Array,
        params: Params,
        actor_input: PlayerActorInput,
        head_params: HeadParams = ...,
    ) -> PlayerAgentOutput: ...
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_player(
        self,
        rng_key: jax.Array,
        params: Params,
        actor_input: PlayerActorInput,
        head_params: HeadParams = HeadParams(),
    ) -> PlayerAgentOutput:
        """For a given single-step, unbatched timestep, output the chosen action."""
        # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.

        actor_input = PlayerActorInput(
            env=jax.tree.map(lambda t: t[None, ...], actor_input.env),
            packed_history=jax.tree.map(
                lambda t: t[:, ...], actor_input.packed_history
            ),
            history=jax.tree.map(lambda t: t[:, ...], actor_input.history),
        )

        actor_output = self._player_apply_fn(
            params,
            actor_input,
            PlayerActorOutput(),
            head_params=head_params,
            rngs={"sampling": rng_key},
        )
        # Remove the padding from above.
        actor_output: PlayerActorOutput = jax.tree.map(
            lambda t: jnp.squeeze(t, axis=0), actor_output
        )

        return PlayerAgentOutput(actor_output=actor_output)
