"""Centralized, zero-wait batched inference for training PlayerActors.

Every training actor thread used to dispatch its own batch-1
``Agent.step_player`` under the shared gpu_lock — ~20 concurrent actors
each paying full dispatch/kernel-launch overhead per environment step,
serialized against each other AND against the learner's train_step by
that same lock. One inference-server thread now runs ONE vmapped
forward over however many requests are pending, so each gpu_lock
acquisition serves a whole batch instead of one step.

Batching is ZERO-WAIT (the Sample Factory / SEED RL scheme): the server
never waits for a batch to fill and never runs a timer — it takes
whatever is queued at the moment it becomes free. While the GPU runs
the current forward (or the learner holds the gpu_lock for a train
step), new requests pile up naturally: GPU busy-time is the batching
window. A lone request runs at batch 1 immediately — exactly the old
path's latency — and under load batches self-size up to
inference_max_batch. There is deliberately NO min-batch and NO max-wait
knob: a previous attempt at this foundered on tuning exactly those two
(any wait stalls actors at game boundaries when no further requests are
coming; the fix is that with a single dispatcher, waiting is never
needed for batching to emerge — the queue depth IS the batch).

Requests batch per params version: (step_count, player_frame_count)
uniquely identifies both a league snapshot and each successive
update_live of a live population (frame_count strictly increases). The
server owns host->device transfer behind an LRU keyed the same way, so
all of main's actors share ONE device copy per params version instead
of each `jax.device_put`-ing their own every game — a strict VRAM and
PCIe improvement over the per-actor copies it replaces.

Compile-shape budget: batch sizes are padded to powers of two (<=
inference_max_batch) and history lengths arrive already geometrically
bucketed by PlayerActor.clip_actor_history, so the number of distinct
traced shapes is (log2(max_batch)+1) x (#history buckets) — the same
bounded-recompilation reasoning as rl/environment/utils.geometric_bucket.
"""

import dataclasses
import functools
import queue
import threading
from collections import OrderedDict
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np

from constants import NUM_HISTORY
from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerAgentOutput,
)
from rl.environment.utils import _bucket_level, _bucket_value
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer

# Must match PlayerActor.clip_actor_history's min_length: incoming history
# lengths are that clip's geometric bucket values, and the shared level
# below re-buckets against the same base.
_HISTORY_MIN_LENGTH = 64


@dataclasses.dataclass
class _InferenceRequest:
    container: ParamsContainer
    rng_key: jax.Array
    actor_input: PlayerActorInput
    done: threading.Event
    output: PlayerAgentOutput | None = None
    error: BaseException | None = None


def _stack_axis0_to(target: int):
    """Returns a tree.map-able stacker: per-request history leaves ->
    one batch array, each zero-padded on axis 0 to ``target``. Zero IS
    this codebase's history-padding convention (rl/environment/utils.
    padnstack zero-fills; a zero FIELD_FEATURE__VALID /
    SPECIES_ENUM___UNSPECIFIED row is an empty slot).

    ``target`` comes from ONE shared bucket level across the batch's
    history AND packed_history (computed in _run_group) rather than each
    leaf group's own max — the same reasoning as the learner's
    the learner's old _stack_and_pad_batch: both lengths describe the same fact (how far
    the batch's longest game has run), so bucketing them independently
    makes XLA trace the PRODUCT of the two axes' bucket sets (4 field x
    5 packed = 20 combos) instead of the shared level's 5, on top of the
    5 batch-size buckets."""

    def stack(*leaves: np.ndarray) -> np.ndarray:
        padded = [
            (
                x
                if x.shape[0] == target
                else np.pad(x, [(0, target - x.shape[0])] + [(0, 0)] * (x.ndim - 1))
            )
            for x in leaves
        ]
        return np.stack(padded)

    return stack


class InferenceServer:
    """One dispatcher thread + one request queue serving batched
    step_player calls for every training PlayerActor. Constructed and
    started once in main.py; PlayerActors submit via step_player() and
    block until their slice of the batched forward comes back (as host
    numpy — the server does one device_get per batch, which also removes
    the per-actor .item() device syncs the old path paid)."""

    def __init__(
        self,
        player_apply_fn,
        gpu_lock=None,
        head_params: HeadParams = HeadParams(),
        max_batch: int = 16,
        params_cache_size: int = 16,
    ):
        self._queue: "queue.SimpleQueue[_InferenceRequest]" = queue.SimpleQueue()
        self._gpu_lock = gpu_lock or nullcontext()
        self._max_batch = max_batch
        # LRU of device-resident params, keyed by version — see module
        # docstring. Only ever touched by the server thread; no lock.
        self._params_cache: "OrderedDict[tuple[int, int], Params]" = OrderedDict()
        self._params_cache_size = params_cache_size

        apply_with_heads = functools.partial(player_apply_fn, head_params=head_params)

        def _single(params, rng_key, actor_input: PlayerActorInput):
            # Mirrors Agent._step_player's call exactly (same positional
            # actor_output placeholder, same rngs collection name).
            return apply_with_heads(
                params,
                actor_input,
                PlayerActorOutput(),
                rngs={"sampling": rng_key},
            )

        # in_axes: shared params, per-request rng key, per-request input.
        # Each vmap slice sees env [1(T), ...] / history [H, ...] — the
        # identical layout Agent._step_player feeds the same apply_fn, so
        # no model change is involved; this is the same "vmap the
        # single-step apply over a batch axis" move the learner's
        # capacity probe already makes.
        self._forward = jax.jit(jax.vmap(_single, in_axes=(None, 0, 0)))
        self._thread = threading.Thread(
            target=self._run, name="inference-server", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    # --- actor-facing API -------------------------------------------------

    def step_player(
        self,
        rng_key: jax.Array,
        params_container: ParamsContainer,
        actor_input: PlayerActorInput,
    ) -> PlayerAgentOutput:
        """Drop-in for Agent.step_player from an actor thread, except the
        params argument is the HOST ParamsContainer (the server owns
        device transfer — see the params cache in the module docstring)."""
        request = _InferenceRequest(
            container=params_container,
            rng_key=rng_key,
            actor_input=actor_input,
            done=threading.Event(),
        )
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise RuntimeError(
                "inference server forward failed for this request"
            ) from request.error
        return request.output

    # --- server internals ---------------------------------------------------

    @staticmethod
    def _version_key(container: ParamsContainer) -> tuple[int, int]:
        return (container.step_count, container.player_frame_count)

    def _get_device_params(self, key: tuple[int, int], container) -> Params:
        params = self._params_cache.get(key)
        if params is None:
            params = jax.device_put(container.player_params)
            self._params_cache[key] = params
            while len(self._params_cache) > self._params_cache_size:
                self._params_cache.popitem(last=False)
        else:
            self._params_cache.move_to_end(key)
        return params

    def _run(self) -> None:
        while True:
            # Zero-wait drain: block only for the FIRST request, then take
            # whatever else is already queued. No timer, no minimum.
            requests = [self._queue.get()]
            while len(requests) < self._max_batch:
                try:
                    requests.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            # One forward per (params version, history bucket level), in
            # arrival order (dict preserves first-appearance order) so no
            # group's requests can be starved by a chattier one. The
            # bucket level is part of the key deliberately: vmap forces
            # one padded history length on the whole batch, so batching a
            # turn-3 game (level-64 history) with a turn-90 one (level
            # 256) made the short game's forward pay the long game's
            # attention FLOPs — with ~12 actors at random game stages,
            # most batches contained one long game, so nearly EVERY step
            # ran at worst-case history length and the batching win
            # leaked away as padding compute. Splitting by level trades a
            # little batch size for every slice running at its own true
            # cost; the traced-shape budget is unchanged (batch buckets x
            # history buckets, same product as before — grouping only
            # changes which combinations actually occur).
            groups: "dict[tuple[int, int, int], list[_InferenceRequest]]" = {}
            for request in requests:
                groups.setdefault(
                    self._version_key(request.container)
                    + (self._history_level(request),),
                    [],
                ).append(request)
            for key, group in groups.items():
                try:
                    self._run_group(group)
                except BaseException as e:  # noqa: BLE001 — must reach requesters
                    for request in group:
                        request.error = e
                        request.done.set()

    @staticmethod
    def _history_level(request: _InferenceRequest) -> int:
        """One shared bucket level across a request's history AND
        packed_history (see _stack_axis0_to's docstring) — also the
        grouping key that keeps different-length games out of the same
        vmap batch (see _run)."""
        field_len = request.actor_input.history.field.shape[0]
        packed_len = request.actor_input.packed_history.revealed_cache.shape[0]
        return max(
            _bucket_level(field_len, _HISTORY_MIN_LENGTH),
            _bucket_level(packed_len, _HISTORY_MIN_LENGTH),
        )

    def _run_group(self, group: "list[_InferenceRequest]") -> None:
        params = self._get_device_params(
            self._version_key(group[0].container), group[0].container
        )

        # All requests in a group share one bucket level by construction
        # (it's part of the grouping key); each target is capped at its
        # own natural maximum exactly like the learner's version —
        # capping only ever pads less, never truncates data.
        level = self._history_level(group[0])
        history_target = _bucket_value(level, _HISTORY_MIN_LENGTH, NUM_HISTORY)
        packed_target = _bucket_value(level, _HISTORY_MIN_LENGTH, 2 * NUM_HISTORY)

        stacked = PlayerActorInput(
            # [B, T=1, ...]: each vmap slice must see env with the same
            # leading T=1 axis Agent._step_player's `t[None, ...]` adds —
            # history/packed_history pass through un-expanded there, so
            # they stack to [B, H, ...] with no extra axis.
            env=jax.tree.map(
                lambda *xs: np.expand_dims(np.stack(xs), 1),
                *[r.actor_input.env for r in group],
            ),
            packed_history=jax.tree.map(
                _stack_axis0_to(packed_target),
                *[r.actor_input.packed_history for r in group],
            ),
            history=jax.tree.map(
                _stack_axis0_to(history_target),
                *[r.actor_input.history for r in group],
            ),
        )
        # Pad the batch axis up to the next power of two by replicating
        # row 0 (extra rows' outputs are simply never read back) — keeps
        # the traced batch sizes to log2(max_batch)+1 distinct values.
        batch = len(group)
        padded_batch = 1 << (batch - 1).bit_length()
        if padded_batch > batch:
            pad = padded_batch - batch
            stacked = jax.tree.map(
                lambda x: np.concatenate([x, np.repeat(x[:1], pad, axis=0)]),
                stacked,
            )
        rng_keys = jnp.stack(
            [r.rng_key for r in group] + [group[0].rng_key] * (padded_batch - batch)
        )

        with self._gpu_lock:
            batched_output = self._forward(params, rng_keys, stacked)
        # One transfer for the whole batch; actors receive plain numpy.
        batched_output = jax.device_get(batched_output)

        for i, request in enumerate(group):
            request.output = PlayerAgentOutput(
                actor_output=jax.tree.map(
                    # [B, T=1, ...] -> drop this request's T axis, same
                    # squeeze Agent._step_player applies.
                    lambda x: np.squeeze(x[i], axis=0),
                    batched_output,
                )
            )
            request.done.set()
