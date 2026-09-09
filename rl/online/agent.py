import functools
import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable

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

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        device: jax.Device,
        field: str,
        size: int = 16,
        params_view: Callable[[Params], Params] | None = None,
    ):
        self._params_view = params_view
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
                params = getattr(container, self._field)
                if self._params_view is not None:
                    params = self._params_view(params)
                params = jax.device_put(params, self._device)
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
        player_params_view: Callable[[Params], Params] | None = None,
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
            device, "player_params", params_cache_size, player_params_view
        )
        self._builder_params = DeviceParamsCache(
            device, "builder_params", params_cache_size
        )

        self.player_head_params = player_head_params
        self.builder_head_params = builder_head_params

        # head_params is a per-CALL argument of the jitted step (a traced
        # pytree of scalars, one trace regardless of value), not baked in
        # via functools.partial: the eval slots pass their own HeadParams
        # (main.py -- the thresholded slot's prune_threshold), which a
        # baked-in python float would turn into one recompile per value.
        self._player_apply_fn = player_apply_fn or _no_apply
        self._builder_apply_fn = builder_apply_fn or _no_apply

    def step_builder(
        self,
        rng_key: jax.Array,
        params_container: ParamsContainer,
        actor_input: BuilderEnvOutput,
    ) -> BuilderAgentOutput:
        return _step(
            self._builder_apply_fn,
            BuilderAgentOutput,
            rng_key,
            self._builder_params.get(params_container),
            actor_input,
            BuilderActorOutput(),
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
            output = _step(
                self._player_apply_fn,
                PlayerAgentOutput,
                rng_key,
                params,
                actor_input,
                PlayerActorOutput(),
                self.player_head_params,
            )
            jax.block_until_ready(output)
        return output


@functools.partial(jax.jit, static_argnums=(0, 1))
def _step(
    apply_fn,
    agent_output_type,
    rng_key: jax.Array,
    params: Params,
    actor_input: PlayerActorInput | BuilderActorInput,
    placeholder: PlayerActorOutput | BuilderActorOutput,
    head_params: HeadParams,
) -> PlayerAgentOutput | BuilderAgentOutput:
    """One unbatched request through `apply_fn`: the env leaves gain the
    leading T=1 axis the network expects and the output loses it again.
    `agent_output_type` and `placeholder` name the head (player or builder)."""
    # Executes during tracing only, once per new abstract input signature.
    leaves, structure = jax.tree.flatten(params)
    signature = repr((structure, [(leaf.shape, str(leaf.dtype)) for leaf in leaves]))
    fingerprint = hashlib.sha256(signature.encode()).hexdigest()[:12]
    logger.info(
        "Actor trace: %s params=%s input=%s",
        agent_output_type.__name__,
        fingerprint,
        jax.tree.structure(actor_input),
    )
    actor_input = actor_input.replace(
        env=jax.tree.map(lambda leaf: leaf[None, ...], actor_input.env)
    )
    actor_output = apply_fn(
        params,
        actor_input,
        placeholder,
        head_params=head_params,
        rngs={"sampling": rng_key},
    )
    return agent_output_type(
        actor_output=jax.tree.map(lambda leaf: jnp.squeeze(leaf, axis=0), actor_output)
    )
