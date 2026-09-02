import functools
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import overload

import jax
import jax.numpy as jnp

from rl.environment.actor_stats import ActorStats, timed
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerAgentOutput,
)
from rl.environment.utils import (
    ACTOR_HISTORY_MIN_LENGTH,
    joint_history_level,
    pad_history_to_level,
)
from rl.model.config import DEFAULT_DTYPE
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer


def _no_apply(*args, **kwargs):
    """Stand-in for an absent apply_fn (an actor may drive only one head)."""
    return None


def resolve_actor_device(name: str) -> tuple[jax.Device, jnp.dtype]:
    """config.player_actor_device -> (device the actors' params are
    committed to, the actor network's COMPUTE dtype). f32 on the host
    because XLA:CPU only emulates bf16; the default bf16 on the GPU."""
    if name == "cpu":
        return jax.devices("cpu")[0], jnp.float32
    if name == "gpu":
        return jax.devices()[0], DEFAULT_DTYPE
    raise ValueError(f"player_actor_device must be 'cpu' or 'gpu', got {name!r}")


class DeviceParamsCache:
    """Host ParamsContainer -> one field of it committed to ``device``, LRU
    by container IDENTITY. The league hands out ONE container object per
    params version (League.materialize caches by step, update_live
    publishes a fresh object), so identity is the version; the entry
    holds the container itself so its id cannot be recycled while cached.
    Not (step_count, frame_count): the eval thread's main and EMA
    containers share both and would alias. Thread-safe — every actor
    thread playing a version shares its one device copy, and a miss is
    transferred under the lock so a new version is copied once, not once
    per actor that sees it first."""

    def __init__(self, device: jax.Device, field: str, size: int = 16):
        self._device = device
        self._field = field
        self._size = size
        self._entries: "OrderedDict[int, tuple[ParamsContainer, Params]]" = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def get(self, container: ParamsContainer) -> Params:
        key = id(container)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                params = jax.device_put(getattr(container, self._field), self._device)
                self._entries[key] = (container, params)
                while len(self._entries) > self._size:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
                params = entry[1]
        return params

    def __len__(self) -> int:
        return len(self._entries)


class Agent:
    """A stateless agent interface: params arrive as the host
    ParamsContainer and are committed to ``device`` behind two versioned
    caches (player, builder), so computation lands where the params live
    — the CPU actor path (config.player_actor_device) is this class with
    device = the host, nothing else."""

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
        device: jax.Device | None = None,
        params_cache_size: int = 16,
    ):
        """Constructs an Agent object."""
        if player_apply_fn is None and builder_apply_fn is None:
            raise ValueError(
                "At least one of player_apply_fn or builder_apply_fn must be provided."
            )
        if device is None:
            device = jax.devices()[0]
        self.device = device
        self._player_params = DeviceParamsCache(
            device, "player_params", params_cache_size
        )
        self._builder_params = DeviceParamsCache(
            device, "builder_params", params_cache_size
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
        self,
        rng_key: jax.Array,
        params_container: ParamsContainer,
        actor_input: BuilderEnvOutput,
    ) -> BuilderAgentOutput:
        return self._step_builder(
            rng_key,
            self._builder_params.get(params_container),
            actor_input,
            self.builder_head_params,
        )

    def step_player(
        self,
        rng_key: jax.Array,
        params_container: ParamsContainer,
        actor_input: PlayerActorInput,
        stats: ActorStats | None = None,
    ) -> PlayerAgentOutput:
        """One request, batch 1. The request is padded to its JOINT history
        bucket level first (the same shape the InferenceServer groups by),
        so this path compiles one variant per level rather than per
        (history level x packed level) pair. ``stats`` receives the same
        forward timer the server records (dispatch + completion), nested
        inside the actor's actor_time_inference."""
        level = joint_history_level(actor_input, ACTOR_HISTORY_MIN_LENGTH)
        actor_input = pad_history_to_level(actor_input, level, ACTOR_HISTORY_MIN_LENGTH)
        params = self._player_params.get(params_container)
        if stats is not None:
            stats.record("actor_infer_history_level", level)
        with timed(stats, "actor_infer_forward"):
            output = self._step_player(
                rng_key, params, actor_input, self.player_head_params
            )
            jax.block_until_ready(output)
        return output

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
