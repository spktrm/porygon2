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
from rl.model.utils import BIAS_VALUE, Params


def threshold_policy(pi: jax.Array, min_p: float = 0.1) -> jax.Array:
    """Thresholds the policy for evaluation."""
    thresholded_pi = jnp.where(pi < pi.max() * min_p, 0.0, pi)
    return thresholded_pi / jnp.sum(thresholded_pi, axis=-1, keepdims=True)


def sample_action(
    rng_key: jax.Array, logits: jax.Array, pi: jax.Array, do_threshold: bool = False
) -> jax.Array:
    """Samples an action from the logits using the provided mask."""
    if do_threshold:
        thresholded_pi = threshold_policy(pi)
        logits = jnp.where(thresholded_pi > 0, logits.astype(jnp.float32), BIAS_VALUE)
    return jax.random.categorical(
        rng_key,
        # Must use float32 here since bfloat16 does some weird things.
        logits.astype(jnp.float32),
        mode="high",
    )


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
        do_threshold: bool = False,
    ):
        """Constructs an Agent object."""

        self._player_apply_fn = player_apply_fn
        self._builder_apply_fn = builder_apply_fn
        self._lock = gpu_lock if gpu_lock is not None else NoOpLock()
        self._do_threshold = do_threshold

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
            rng_key,
            actor_output.head.logits,
            actor_output.head.policy,
            self._do_threshold,
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
            actor_output.action_type_head.logits,
            actor_output.action_type_head.policy,
            self._do_threshold,
        )
        move_head = sample_action(
            move_key,
            actor_output.move_head.logits,
            actor_output.move_head.policy,
            self._do_threshold,
        )
        switch_head = sample_action(
            switch_key,
            actor_output.switch_head.logits,
            actor_output.switch_head.policy,
            self._do_threshold,
        )
        wildcard_head = sample_action(
            wildcard_key,
            actor_output.wildcard_head.logits[move_head],
            actor_output.wildcard_head.policy[move_head],
            self._do_threshold,
        )

        return PlayerAgentOutput(
            action_type_head=action_type_head,
            move_head=move_head,
            switch_head=switch_head,
            wildcard_head=wildcard_head,
            actor_output=actor_output,
        )
