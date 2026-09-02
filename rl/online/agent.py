import functools
from collections.abc import Callable
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


def _no_apply(*args, **kwargs):
    """Stand-in for an absent apply_fn (an actor may drive only one head)."""
    return None


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

        # head_params is a per-CALL argument of the jitted steps (a traced
        # pytree of scalars, one trace regardless of value), not baked in
        # via functools.partial: eval actors run at temp 0.5 and training
        # actors at 1.0 (main.py), which a
        # baked-in python float would turn into one recompile per value.
        self._player_apply_fn = player_apply_fn or _no_apply
        self._builder_apply_fn = builder_apply_fn or _no_apply

    def step_builder(
        self, rng_key: jax.Array, params: Params, actor_input: BuilderEnvOutput
    ) -> BuilderAgentOutput:
        return self._step_builder(
            rng_key, params, actor_input, self.builder_head_params
        )

    def step_player(
        self,
        rng_key: jax.Array,
        params: Params,
        actor_input: PlayerActorInput,
    ) -> PlayerAgentOutput:
        return self._step_player(rng_key, params, actor_input, self.player_head_params)

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
            history_carry=actor_input.history_carry,
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
