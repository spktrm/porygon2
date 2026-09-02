"""Offline play-and-read harness (2026-08-23).

Plays self-play games against a game service with a plain params pytree —
no Learner, InferenceServer, league, replay buffer or wandb — and re-runs
the learner-side model (train=True heads) over the collected chunks so
critic questions can be answered against real trajectories without a
training run. Born from the critic-weakness check that needed a stub
learner, a monkeypatched SERVER_URI and a sed'd copy of the service to
exist at all.

Typical use (a second service instance keeps the training one's battles
untouched):

    PORT=8081 MAX_WORKERS=1 MEMORY_STATS_PATH=/tmp/svc.json \\
        node service/dist/server/index.js &
    PS_SERVICE_URI=ws://localhost:8081 env/bin/python -c '
    from rl.offline.harness import *
    params = load_params("ckpts/gen9/ckpt_00100000")
    sides = play_games(params, n_games=100, pairs=6, tag="chk")
    dump(sides, "/tmp/chk.pkl")
    for pred, batch in forward(params, flatten(sides)):
        ...
    '

Every game is bounded by `deadline_s` in play_games: the service has no
turn cap yet, and a battle that never resolves (seen on 2026-08-23, ~2%
of games) would otherwise pin a slot forever — stragglers are dropped,
not waited on, and the count is logged.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from rl import checkpoint
from rl.environment.data import CAT_VF_SUPPORT
from rl.environment.env import SinglePlayerSyncEnvironment
from rl.environment.interfaces import PlayerActorInput, PlayerActorOutput, Trajectory
from rl.model.config import get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.online.agent import Agent
from rl.online.config import Porygon2LearnerConfig, get_learner_config
from rl.online.player_actor import PlayerActor
from rl.online.training.batching import stack_batch

logger = logging.getLogger(__name__)


class _RunState:
    done = False


@dataclass
class OfflineContext:
    """The slice of Learner that PlayerActor.unroll actually reads:
    config.{player_history_length, player_chunk_length, smogon_format}
    and run_state.done. Flip `run_state.done` to unwind every in-flight
    unroll with ActorStopped."""

    config: Porygon2LearnerConfig = field(default_factory=get_learner_config)
    run_state: _RunState = field(default_factory=_RunState)


def load_params(ckpt_dir: str, which: str = "target_params"):
    """`target_params` (EMA — what actors and the league play) by default;
    `params` for the raw learner leaf, `reg_params` for the NashPG
    reference."""
    return checkpoint.load_component(ckpt_dir, "player", which)


def play_games(
    params,
    n_games: int,
    pairs: int = 6,
    tag: str = "offline",
    deadline_s: float = 1200.0,
    generation: int = 9,
    smogon_format: str = "randombattle",
    seed: int = 0,
) -> list[list[Trajectory]]:
    """Plays n_games self-play games (both sides `params`), `pairs` at a
    time, and returns one chunk list per SIDE (2 per game) in completion
    order. Games still running at `deadline_s` are abandoned."""
    ctx = OfflineContext()
    ctx.config = get_learner_config()
    actor_net = get_player_model(get_player_model_config(generation, train=False))
    agent = Agent(actor_net.apply)
    dev_params = jax.device_put(params)

    def play_one(game_no: int) -> list[list[Trajectory]]:
        actors, envs = [], []
        for p in range(2):
            env = SinglePlayerSyncEnvironment(
                f"{tag}:g{game_no}p{p}",
                generation=generation,
                smogon_format=smogon_format,
            )
            envs.append(env)
            actor = PlayerActor(
                agent,
                env,
                unroll_length=ctx.config.unroll_length,
                learner=ctx,
                rng_seed=seed * 100_003 + game_no * 2 + p,
                inference_client=None,
            )
            actor.set_game_id(f"{tag}-{game_no}")
            actors.append(actor)
        try:
            with cf.ThreadPoolExecutor(2) as ex:
                futs = [
                    ex.submit(
                        a.unroll, jax.random.key(seed * 7 + game_no * 2 + i), dev_params
                    )
                    for i, a in enumerate(actors)
                ]
                return [f.result() for f in futs]
        finally:
            for env in envs:
                env.close()

    sides: list[list[Trajectory]] = []
    t0 = time.time()
    ex = cf.ThreadPoolExecutor(pairs)
    futs = [ex.submit(play_one, g) for g in range(n_games)]
    done = failed = 0
    try:
        for f in cf.as_completed(futs, timeout=deadline_s):
            try:
                sides.extend(f.result())
                done += 1
            except Exception:  # noqa: BLE001 — one bad game must not end the sweep
                failed += 1
                logger.warning("game failed", exc_info=True)
            if done % 10 == 0:
                logger.info("%d/%d games, %.0fs", done, n_games, time.time() - t0)
    except cf.TimeoutError:
        pass
    ctx.run_state.done = True  # unwinds the stragglers' recv polls
    ex.shutdown(wait=False, cancel_futures=True)
    logger.info(
        "play_games: %d complete, %d failed, %d abandoned at the %.0fs deadline",
        done,
        failed,
        n_games - done - failed,
        deadline_s,
    )
    return sides


def flatten(sides: list[list[Trajectory]]) -> list[Trajectory]:
    return [c for chunks in sides for c in chunks]


def outcome(chunks: list[Trajectory]) -> float:
    """A side's terminal result in CAT_VF_SUPPORT units (+1 win / -1 loss),
    read from its last chunk's final row (the chunk contract)."""
    return float(
        chunks[-1].player_transitions.env_output.win_reward[-1] @ CAT_VF_SUPPORT
    )


def dump(sides, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(sides, f)


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def forward(
    params,
    chunks: list[Trajectory],
    batch: int = 4,
    generation: int = 9,
) -> Iterator[tuple[PlayerActorOutput, object]]:
    """Yields (prediction, stacked batch) over `chunks` in batches, using
    the learner-side model (train=True, so the full-support log_policy is
    populated). Batch axis is 1, matching the learner's apply_fn. Leaves are
    host numpy-able jax arrays; read the action cells with
    decode_log_policy(pred, flat_action_mask) and V as
    pred.value_head.expectation.""" ""
    net = get_player_model(get_player_model_config(generation, train=True))
    apply = jax.jit(jax.vmap(net.apply, in_axes=(None, 1, 1, None), out_axes=1))
    dev_params = jax.device_put(params)
    for i in range(0, len(chunks), batch):
        b = stack_batch(chunks[i : i + batch])
        pt = b.player_transitions
        actor_input = PlayerActorInput(
            env=pt.env_output,
            packed_history=b.player_packed_history,
            history=b.player_history,
        )
        yield apply(
            dev_params, actor_input, pt.agent_output.actor_output, HeadParams()
        ), b


def decode_log_policy(pred: PlayerActorOutput, flat_action_mask) -> np.ndarray:
    """The full-support log-policy over the block cells, (T, B, A), masked
    to legal cells.

    Replaces decode_q on 2026-08-29. The Q readout it decoded went with the
    advantage head; the action grid the policy itself scores is what is left,
    and it is what the separation probe now regresses onto."""
    return np.asarray(
        jnp.where(jnp.asarray(flat_action_mask, bool), pred.action_head.log_policy, 0.0)
    )


def gpu_headroom_env(fraction: float = 0.12) -> None:
    """Call BEFORE importing jax in a process that must coexist with a
    live training run on the same GPU."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(fraction))
