"""Centralized, zero-wait batched inference for training PlayerActors.

Every training actor thread used to dispatch its own batch-1
``Agent.step_player`` — ~20 concurrent actors each paying full
dispatch/kernel-launch overhead per environment step. One
inference-server thread now runs ONE vmapped forward over however many
requests are pending, so one dispatch serves a whole batch instead of
one step. (Until 2026-09-03 those dispatches were also serialised
against the learner's train step by a shared gpu_lock; the lock ordered
host dispatch only — JAX dispatch is asynchronous, so it never bounded
GPU execution or VRAM — and it was deleted.)

Batching is ZERO-WAIT (the Sample Factory / SEED RL scheme): the server
never waits for a batch to fill and never runs a timer — it takes
whatever is queued at the moment it becomes free. While the GPU runs
the current forward, new requests pile up naturally: GPU busy-time is the batching
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
server owns host->device transfer behind a DeviceParamsCache (the same
one the direct Agent path uses, rl/online/agent.py), so all of main's
actors share ONE device copy per params version instead of each
`jax.device_put`-ing their own every game — a strict VRAM and PCIe
improvement over the per-actor copies it replaces.

Compile-shape budget: batch sizes are padded to powers of two (<=
inference_max_batch) and history lengths arrive already geometrically
bucketed by PlayerActor.clip_actor_history, so the number of distinct
traced shapes is (log2(max_batch)+1) x (#history buckets) — the same
bounded-recompilation reasoning as rl/environment/utils.geometric_bucket.

This is the config.player_actor_device="gpu" arm. Under "cpu" every
actor runs its own f32 batch-1 forward on the host through Agent.step_
player and the server is not constructed.
"""

import dataclasses
import functools
import queue
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.actor_stats import ActorStats, timed
from rl.environment.env import ActorStopped
from rl.environment.interfaces import (
    HistoryCarry,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerAgentOutput,
)
from rl.environment.utils import (
    ACTOR_HISTORY_MIN_LENGTH,
    joint_history_level,
    pad_history_to_level,
)
from rl.model.heads import HeadParams
from rl.model.history_encoder import invalid_history_carry
from rl.model.utils import ParamsContainer
from rl.online.agent import DeviceParamsCache


@dataclasses.dataclass
class _InferenceRequest:
    container: ParamsContainer
    rng_key: jax.Array
    actor_input: PlayerActorInput
    done: threading.Event
    # perf_counter at step_player's enqueue — queue wait is measured
    # from here to the start of the group that serves the request.
    enqueued_at: float = 0.0
    output: PlayerAgentOutput | None = None
    error: BaseException | None = None


def _stack_history_carries(carries: "list[HistoryCarry]") -> HistoryCarry:
    """Carry leaves stack on axis 0 like rng_keys — a request that resumes
    and one that does not differ only in `valid`. A request with NO carry
    (empty leaves, a caller that never carries) in a group beside one that
    does gets an invalid carry of the same width, so the group still
    stacks and that request still computes from h0; a group with no carry
    leaves at all keeps the empty carry (the encoder's static h0 branch)."""
    widths = [
        carry.slot_states.shape[-1]
        for carry in carries
        if not isinstance(carry.slot_states, tuple)
    ]
    if not widths:
        return HistoryCarry()
    filled = []
    for carry in carries:
        if isinstance(carry.slot_states, tuple):
            filled.append(invalid_history_carry(widths[0]))
        else:
            filled.append(carry)
    return jax.tree.map(lambda *xs: np.stack(xs), *filled)


_STOP_POLL_SECONDS = 1.0


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
        head_params: HeadParams = HeadParams(),
        max_batch: int = 16,
        params_cache_size: int = 16,
        stats: ActorStats | None = None,
    ):
        self._queue: "queue.SimpleQueue[_InferenceRequest]" = queue.SimpleQueue()
        # Per-phase timing sink for _run_group (rl/environment/actor_stats.py).
        self._stats = stats
        # Set by stop() at shutdown: step_player polls its result wait
        # against it and unwinds the calling actor thread with
        # ActorStopped, the same poll-against-stop the game-server
        # receive got in f9401f8. Without it an actor whose request was
        # queued behind a dispatcher that will never run again blocked in
        # Event.wait() forever, and the executor's non-daemon worker
        # threads then wedged the interpreter's exit-time join (the
        # 2026-08-23 post-checkpoint hang).
        self._stop = threading.Event()
        self._max_batch = max_batch
        # Device-resident params per version — see module docstring.
        self._params_cache = DeviceParamsCache(
            jax.devices()[0], "player_params", params_cache_size
        )

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

    def stop(self) -> None:
        """Unblocks every actor waiting in step_player (they raise
        ActorStopped within one poll interval). The dispatcher thread is
        a daemon and needs no join."""
        self._stop.set()

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
            enqueued_at=time.perf_counter(),
        )
        self._queue.put(request)
        while not request.done.wait(timeout=_STOP_POLL_SECONDS):
            if self._stop.is_set():
                raise ActorStopped("training stopped while waiting on inference")
        if request.error is not None:
            raise RuntimeError(
                "inference server forward failed for this request"
            ) from request.error
        return request.output

    @staticmethod
    def _version_key(container: ParamsContainer) -> tuple[int, int]:
        """Grouping key only; the params cache is keyed by container
        identity (DeviceParamsCache)."""
        return (container.step_count, container.player_frame_count)

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
        packed_history (rl/environment/utils.joint_history_level) — also
        the grouping key that keeps different-length games out of the
        same vmap batch (see _run)."""
        return joint_history_level(request.actor_input, ACTOR_HISTORY_MIN_LENGTH)

    def _run_group(self, group: "list[_InferenceRequest]") -> None:
        group_start = time.perf_counter()
        if self._stats is not None:
            for request in group:
                self._stats.record(
                    "actor_infer_queue_wait", (group_start - request.enqueued_at) * 1e3
                )
            self._stats.record("actor_infer_batch_size", len(group))
            self._stats.record(
                "actor_infer_history_level", self._history_level(group[0])
            )
        params = self._params_cache.get(group[0].container)

        # All requests in a group share one bucket level by construction
        # (it's part of the grouping key), so padding each to that level
        # gives the group one shape to stack — the same pad the direct
        # Agent path applies to a lone request.
        level = self._history_level(group[0])

        with timed(self._stats, "actor_infer_stack"):
            padded = [
                pad_history_to_level(r.actor_input, level, ACTOR_HISTORY_MIN_LENGTH)
                for r in group
            ]
            stacked = PlayerActorInput(
                # [B, T=1, ...]: each vmap slice must see env with the same
                # leading T=1 axis Agent._step_player's `t[None, ...]` adds —
                # history/packed_history pass through un-expanded there, so
                # they stack to [B, H, ...] with no extra axis.
                env=jax.tree.map(
                    lambda *xs: np.expand_dims(np.stack(xs), 1),
                    *[p.env for p in padded],
                ),
                packed_history=jax.tree.map(
                    lambda *xs: np.stack(xs), *[p.packed_history for p in padded]
                ),
                history=jax.tree.map(
                    lambda *xs: np.stack(xs), *[p.history for p in padded]
                ),
                history_carry=_stack_history_carries([p.history_carry for p in padded]),
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

        # forward = dispatch + completion: device_get blocked on the
        # result anyway, so block_until_ready only moves the wait onto
        # its own timer rather than adding one.
        with timed(self._stats, "actor_infer_forward"):
            batched_output = self._forward(params, rng_keys, stacked)
            jax.block_until_ready(batched_output)
        # One transfer for the whole batch; actors receive plain numpy.
        with timed(self._stats, "actor_infer_device_get"):
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
