import asyncio
import functools
import random
import statistics
import time
from abc import ABC, abstractmethod
import traceback
from typing import Callable, List, Sequence, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import uvloop
import websockets
from tqdm import tqdm

from ml.arch.model import get_dummy_model, get_model
from ml.config import FineTuning
from ml.learners.func import collect_batch_telemetry_data
from ml.utils import Params
from rlenv.env import as_jax_arr, get_ex_step, process_state
from rlenv.interfaces import ActorStep, EnvStep, HistoryStep, ModelOutput, TimeStep
from rlenv.protos.service_pb2 import (
    Action,
    ClientMessage,
    ConnectMessage,
    ResetMessage,
    ServerMessage,
    StepMessage,
)
from rlenv.protos.state_pb2 import State
from rlenv.utils import stack_steps

# Define the server URI
SERVER_URI = "ws://localhost:8080"

uvloop.install()


class SinglePlayerEnvironment:
    websocket: websockets.WebSocketClientProtocol

    def __init__(self, player_id: int, game_id: int):
        self.player_id = player_id
        self.game_id = game_id
        self.state = None
        self.mess = None
        self.queue = asyncio.Queue()

    def is_done(self):
        return self.state.info.done

    @classmethod
    async def create(cls, player_id: int, game_id: int):
        """Async factory method for initialization."""
        self = cls(player_id, game_id)
        await self._init()
        return self

    async def _init(self):
        self.websocket = await websockets.connect(SERVER_URI)
        connect_message = ClientMessage(
            player_id=self.player_id, game_id=self.game_id, connect=ConnectMessage()
        )
        await self.websocket.send(connect_message.SerializeToString())
        asyncio.create_task(self.read_continuously())

    async def read_continuously(self):
        while True:
            try:
                incoming_message = await self.websocket.recv()
                await self.queue.put(incoming_message)
            except websockets.ConnectionClosed:
                print(f"Connection closed for player {self.player_id}")
                break

    async def reset(self):
        await self._reset()
        return process_state(self.state)

    async def _reset(self):
        reset_message = ClientMessage(
            player_id=self.player_id, game_id=self.game_id, reset=ResetMessage()
        )
        await self.websocket.send(reset_message.SerializeToString())
        return await self._recv()

    async def step(self, action: int):
        await self._step(action)
        return process_state(self.state)

    async def _step(self, action: int):
        if self.state and self.state.info.done:
            return self.state
        step_message = ClientMessage(
            player_id=self.player_id,
            game_id=self.game_id,
            step=StepMessage(
                action=Action(rqid=self.mess.game_state.rqid, value=action)
            ),
        )
        await self.websocket.send(step_message.SerializeToString())
        return await self._recv()

    async def _recv(self):
        server_message_data = await self.queue.get()
        server_message = ServerMessage.FromString(server_message_data)
        self.mess = server_message
        self.state = State.FromString(server_message.game_state.state)
        return self.state


class TwoPlayerEnvironment:
    def __init__(self, game_id: int, player_ids: List[int]):
        self.game_id = game_id
        self.player_ids = player_ids
        self.players = []
        self.state_queue = asyncio.Queue()
        self.current_state: State = None
        self.current_step: Tuple[EnvStep, HistoryStep] = None
        self.current_player = None
        self.dones = np.zeros(2)

    async def initialize_players(self):
        """Initialize both players asynchronously."""
        self.players = await asyncio.gather(
            *[
                SinglePlayerEnvironment.create(pid, self.game_id)
                for pid in self.player_ids
            ]
        )
        self.current_player = self.players[0]  # Start with the first player

    def is_done(self):
        return self.dones.all()

    async def _reset(self):
        """Reset both players, enqueue their initial states, and return the first state to act on."""
        # Reset both players and add their states to the queue
        while not self.state_queue.empty():
            await self.state_queue.get()
        self.dones = np.zeros(2)

        states = await asyncio.gather(*[player._reset() for player in self.players])

        for player, state in zip(self.players, states):
            await self.state_queue.put((player, state))

        # Retrieve the first state to act on and set the corresponding player
        self.current_player, self.current_state = await self.state_queue.get()
        self.current_step = process_state(self.current_state)
        return self.current_step

    async def _step(self, action: int):
        """Process the action for the current player asynchronously."""
        # Start the action processing in the background without waiting for it to complete
        asyncio.create_task(self._perform_action(self.current_player, action))

        # Retrieve the next player and state from the queue, waiting if necessary
        self.current_player, self.current_state = await self.state_queue.get()
        self.current_step = process_state(self.current_state)
        return self.current_step

    async def _perform_action(self, player: SinglePlayerEnvironment, action: int):
        """Helper method to send the action to the player and enqueue the resulting state."""
        # Perform the step and add the resulting state along with the player back into the queue
        if not player.is_done():
            state = await player._step(action)
            self.dones[int(state.info.player_index)] = state.info.done
            await self.state_queue.put((player, state))

        if self.is_done():
            await self.state_queue.put((self.current_player, self.current_state))


class BatchEnvironment(ABC):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[List[EnvStep], List[HistoryStep]]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Tuple[List[EnvStep], List[HistoryStep]]:
        raise NotImplementedError


class BatchTwoPlayerEnvironment(BatchEnvironment):
    def __init__(self, num_envs: int):
        super().__init__(num_envs)
        try:
            self.loop = asyncio.get_event_loop()
        except:
            self.loop = asyncio.new_event_loop()

        self.envs: List[TwoPlayerEnvironment] = []
        self.player_id_count = 0
        for game_id in range(num_envs):
            env = TwoPlayerEnvironment(
                game_id, [self.player_id_count, self.player_id_count + 1]
            )
            self.loop.run_until_complete(env.initialize_players())
            self.envs.append(env)
            self.player_id_count += 2

    def is_done(self):
        return np.array([env.is_done() for env in self.envs]).all()

    def step(self, actions: List[int]):
        out = self.loop.run_until_complete(
            asyncio.gather(
                *[env._step(action) for env, action in zip(self.envs, actions)]
            ),
        )
        return list(zip(*out))

    def reset(self):
        out = self.loop.run_until_complete(
            asyncio.gather(*[env._reset() for env in self.envs])
        )
        return list(zip(*out))


class BatchSinglePlayerEnvironment(BatchEnvironment):
    def __init__(self, num_envs: int, is_eval: bool = True):
        super().__init__(num_envs)
        try:
            self.loop = asyncio.get_event_loop()
        except:
            self.loop = asyncio.new_event_loop()

        self.envs: List[SinglePlayerEnvironment] = []
        self.player_id_count = 0
        if is_eval:
            for game_id in range(num_envs):
                env = self.loop.run_until_complete(
                    SinglePlayerEnvironment.create(10_000 + game_id, 10_000 + game_id)
                )
                self.envs.append(env)
                self.player_id_count += 1
        else:
            for game_id in range(num_envs):
                for _ in range(2):
                    env = self.loop.run_until_complete(
                        SinglePlayerEnvironment.create(self.player_id_count, game_id)
                    )
                    self.envs.append(env)
                    self.player_id_count = self.player_id_count + 1

    def is_done(self):
        return np.array([env.is_done() for env in self.envs]).all()

    def step(self, actions: List[int]):
        states = self.loop.run_until_complete(
            asyncio.gather(
                *[env._step(action) for env, action in zip(self.envs, actions)]
            ),
        )
        return zip(*[process_state(state) for state in states])

    def reset(self):
        states = self.loop.run_until_complete(
            asyncio.gather(*[env._reset() for env in self.envs])
        )
        return zip(*[process_state(state) for state in states])


class BatchCollectorV2:
    def __init__(
        self,
        network: nn.Module,
        batch_size: int,
        env_constructor: BatchEnvironment = BatchTwoPlayerEnvironment,
        finetune: bool = False,
    ):
        self.game: BatchEnvironment = env_constructor(batch_size)
        self.network = network
        self.finetuning = FineTuning()
        self.batch_size = batch_size
        self.finetune = finetune
        self.num_steps = 0

    def _batch_of_states_apply_action(self, actions: chex.Array) -> Sequence[EnvStep]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        return self.game.step(list(actions))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(
        self, params: Params, env_steps: EnvStep, history_step: HistoryStep
    ) -> chex.Array:
        rollout: Callable[[Params, EnvStep], ModelOutput] = jax.vmap(
            self.network.apply, (None, 0, 0), 0
        )
        output = rollout(params, env_steps, history_step)
        finetuned_pi = self.finetuning._threshold(output.pi, env_steps.legal)
        return jnp.where(self.finetune, finetuned_pi, output.pi), output.log_pi

    def actor_step(self, params: Params, env_step: EnvStep, history_step: HistoryStep):
        env_step = as_jax_arr(env_step)
        history_step = as_jax_arr(history_step)
        pi, log_pi = self._network_jit_apply(params, env_step, history_step)
        action = np.apply_along_axis(
            lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        return action, ActorStep(
            action=action, policy=pi, log_policy=log_pi, rewards=()
        )

    def collect_batch_trajectory(
        self, params: Params, resolution: int = 32
    ) -> TimeStep:
        timesteps = []

        ex, hx = self.game.reset()
        env_step: EnvStep = stack_steps(ex)
        history_step: HistoryStep = stack_steps(hx)

        state_index = 0
        while True:
            prev_env_step = env_step
            prev_history_step = history_step

            a, actor_step = self.actor_step(params, env_step, history_step)

            ex, hx = self._batch_of_states_apply_action(a)
            env_step = stack_steps(ex)
            history_step = stack_steps(hx)

            timestep = TimeStep(
                env=prev_env_step,
                history=prev_history_step,
                actor=ActorStep(
                    action=actor_step.action,
                    policy=actor_step.policy,
                    log_policy=actor_step.log_policy,
                    rewards=env_step.rewards,
                ),
            )
            timesteps.append(timestep)

            if self.game.is_done() and state_index % resolution == 0:
                break

            state_index += 1

        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        batch: TimeStep = stack_steps(timesteps)
        self.num_steps += batch.env.valid.sum()

        if (batch.env.turn[1:] < batch.env.turn[:-1]).any():
            raise ValueError

        if (batch.env.seed_hash != batch.env.seed_hash[0]).any():
            raise ValueError

        if not np.all(np.array([env.is_done() for env in self.game.envs])):
            raise ValueError

        # if not np.all(
        #     np.abs((batch.actor.win_rewards * batch.env.valid[..., None]).sum(0)) == 1
        # ):
        #     raise ValueError

        return as_jax_arr(batch)


class SingleTrajectoryTrainingBatchCollector(BatchCollectorV2):
    def __init__(self, network: nn.Module, batch_size: int):
        super().__init__(network, batch_size, BatchTwoPlayerEnvironment)


class DoubleTrajectoryTrainingBatchCollector(BatchSinglePlayerEnvironment):
    def __init__(
        self,
        network: nn.Module,
        batch_size: int,
        initial_min_batch_size: int = 4,
        initial_max_wait_time: float = 0.01,
        enable_auto_tuning: bool = True,
    ):
        super().__init__(batch_size, False)

        self.network = network
        self.finetuning = FineTuning()
        self.batch_size = batch_size
        self.num_steps = 0

        # Tunable parameters
        self.min_batch_size = initial_min_batch_size
        self.max_wait_time = initial_max_wait_time
        self.enable_auto_tuning = enable_auto_tuning

        # Metrics for auto-tuning
        self.batch_sizes = []
        self.wait_times = []
        self.inference_times = []
        self.last_tune_step = 0
        self.tune_frequency = 100  # Tune parameters every 100 batches

        # Collection phase tracking
        self.collection_phase = "start"  # "start", "middle", "end"
        self.phase_lock = asyncio.Lock()
        self.active_envs = batch_size
        self.active_envs_lock = asyncio.Lock()
        self.episode_lengths = []  # Track lengths to predict completion
        self.current_steps = 0

        # Phase-specific parameters
        self.phase_params = {
            "start": {"min_batch_ratio": 0.5, "wait_time_scale": 1.0},
            "middle": {"min_batch_ratio": 0.3, "wait_time_scale": 0.8},
            "end": {"min_batch_ratio": 0.15, "wait_time_scale": 0.5},
        }

        # Dynamic batching parameters
        self.dynamic_min_batch_size = initial_min_batch_size
        self.dynamic_wait_time = initial_max_wait_time

        # Batch inference infrastructure
        self.batch_length = 0
        self.sync_point = asyncio.Barrier(batch_size * 2)
        self.inference_queue = asyncio.Queue()
        self.inference_results = {}
        self.inference_lock = asyncio.Lock()
        self.inference_task = None
        self.stop_inference = asyncio.Event()
        self.current_params = None

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(
        self, params: Params, ex: EnvStep, hx: HistoryStep
    ) -> chex.Array:
        output = self.network.apply(params, ex, hx)
        return output.pi, output.log_pi

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_batch_inference(
        self, params: Params, ex_batch: EnvStep, hx_batch: HistoryStep
    ) -> tuple:
        outputs = jax.vmap(self.network.apply, (None, 0, 0), 0)(
            params, ex_batch, hx_batch
        )
        return outputs.pi, outputs.log_pi

    async def update_collection_phase(self, steps_completed):
        """Update the collection phase based on progress."""
        self.current_steps = steps_completed

        if not self.episode_lengths:
            # No history yet, use step-based estimation
            if steps_completed < 10:
                phase = "start"
            elif steps_completed < 100:  # Arbitrary threshold
                phase = "middle"
            else:
                phase = "end"
        else:
            # Use episode length history for better estimation
            avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
            progress_ratio = steps_completed / avg_length

            if progress_ratio < 0.3:
                phase = "start"
            elif progress_ratio < 0.7:
                phase = "middle"
            else:
                phase = "end"

        async with self.phase_lock:
            if phase != self.collection_phase:
                old_phase = self.collection_phase
                self.collection_phase = phase
                # print(f"Collection phase transition: {old_phase} -> {phase}")

                # Apply phase-specific parameter adjustments
                phase_config = self.phase_params[phase]
                self.dynamic_min_batch_size = max(
                    1, int(self.batch_size * phase_config["min_batch_ratio"])
                )
                self.dynamic_wait_time = (
                    self.max_wait_time * phase_config["wait_time_scale"]
                )

    async def batch_inference_worker(self, params: Params):
        """Worker that processes batched observations."""
        try:
            while not self.stop_inference.is_set():
                # Get current phase parameters
                async with self.phase_lock:
                    current_phase = self.collection_phase
                    min_batch = self.dynamic_min_batch_size
                    wait_time_scale = self.phase_params[current_phase][
                        "wait_time_scale"
                    ]

                # Get current active environment count
                async with self.active_envs_lock:
                    current_active_envs = self.active_envs

                # Calculate actual parameters based on phase and active environments
                effective_min_batch = min(min_batch, max(1, current_active_envs // 2))
                effective_wait_time = self.max_wait_time * wait_time_scale

                # Log adaptive parameters periodically
                # if random.random() < 0.01:  # Log ~1% of the time to avoid spam
                #     print(
                #         f"Phase: {current_phase}, Active: {current_active_envs}/{self.batch_size}, "
                #         f"Min batch: {effective_min_batch}, Wait time: {effective_wait_time:.4f}s"
                #     )

                # Collect pending observations
                batch_obs = []
                batch_hist = []
                request_ids = []

                # Get at least one observation or exit if stopped
                try:
                    # Use wait_for with a timeout to check stop_inference periodically
                    request_id, (ex, hx) = await asyncio.wait_for(
                        self.inference_queue.get(), timeout=0.1
                    )
                    batch_obs.append(ex)
                    batch_hist.append(hx)
                    request_ids.append(request_id)
                except asyncio.TimeoutError:
                    # No observations received, check if we should stop
                    if self.stop_inference.is_set():
                        break
                    continue

                # Try to get more observations up to batch_size
                timeout = effective_wait_time
                start_time = time.time()

                # Collect metrics for tuning
                batch_start_time = time.time()

                try:
                    while (
                        len(batch_obs) < self.batch_size
                        and (time.time() - start_time) < timeout
                        and not self.stop_inference.is_set()
                    ):

                        # If we have the minimum batch size, reduce the wait time
                        if len(batch_obs) >= effective_min_batch:
                            remaining = max(0, timeout - (time.time() - start_time))
                            wait_time = min(
                                remaining, 0.001
                            )  # Short wait if we have enough samples
                        else:
                            wait_time = timeout - (time.time() - start_time)

                        try:
                            req_id, (ex, hx) = await asyncio.wait_for(
                                self.inference_queue.get(), wait_time
                            )
                            batch_obs.append(ex)
                            batch_hist.append(hx)
                            request_ids.append(req_id)
                        except asyncio.TimeoutError:
                            break
                except Exception as e:
                    print(f"Error collecting batch: {e}")

                # Collect wait time metric
                wait_time = time.time() - start_time
                self.wait_times.append(wait_time)
                self.batch_sizes.append(len(batch_obs))

                # Process batch if we have observations and haven't been stopped
                if batch_obs and not self.stop_inference.is_set():
                    # Convert to batch and call network
                    ex_batch = as_jax_arr(stack_steps(batch_obs))
                    hx_batch = as_jax_arr(stack_steps(batch_hist))

                    pi_batch, log_pi_batch = self._network_batch_inference(
                        params, ex_batch, hx_batch
                    )

                    # Record inference time
                    inference_time = time.time() - batch_start_time
                    self.inference_times.append(inference_time)

                    # Distribute results
                    async with self.inference_lock:
                        for i, req_id in enumerate(request_ids):
                            self.inference_results[req_id] = (
                                pi_batch[i],
                                log_pi_batch[i],
                            )
                            # Mark task as done for the queue
                            self.inference_queue.task_done()

                    # Tune parameters if needed
                    if (
                        self.enable_auto_tuning
                        and len(self.batch_sizes) - self.last_tune_step
                        >= self.tune_frequency
                    ):
                        self._tune_parameters()

        except Exception as e:
            print(f"Inference worker exception: {e}")
        finally:
            # Clean up any remaining requests if we're shutting down
            async with self.inference_lock:
                for req_id in request_ids:
                    if req_id not in self.inference_results:
                        # Create dummy results for any pending requests
                        # (this ensures no agent is left waiting forever)
                        self.inference_results[req_id] = (None, None)
            # print("Inference worker stopped")

    def _tune_parameters(self):
        """Adjust parameters based on collected metrics."""
        # Only look at metrics since last tuning
        recent_batch_sizes = self.batch_sizes[self.last_tune_step :]
        recent_wait_times = self.wait_times[self.last_tune_step :]
        recent_inference_times = self.inference_times[self.last_tune_step :]

        if not recent_batch_sizes:
            return

        # Update last tune step
        self.last_tune_step = len(self.batch_sizes)

        # Calculate metrics
        avg_batch_size = sum(recent_batch_sizes) / len(recent_batch_sizes)
        avg_wait_time = sum(recent_wait_times) / len(recent_wait_times)
        avg_inference_time = sum(recent_inference_times) / len(recent_inference_times)

        # Target: Maximize throughput while keeping latency reasonable
        # Estimate throughput as observations processed per time unit
        throughput = sum(recent_batch_sizes) / (
            sum(recent_wait_times) + sum(recent_inference_times)
        )

        # Adaptive adjustments
        if avg_batch_size < self.min_batch_size * 1.1:
            # Rarely hitting min_batch_size, might be waiting too long
            # Decrease min_batch_size to reduce wait time
            self.min_batch_size = max(1, int(self.min_batch_size * 0.9))
        elif avg_batch_size > self.batch_size * 0.9:
            # Usually getting close to max batch size, could increase min_batch_size
            # to reduce overhead of small batches
            self.min_batch_size = min(self.batch_size, int(self.min_batch_size * 1.1))

        # Adjust wait time based on inference time
        # We want wait time to be proportional to inference time
        # More expensive inference = worth waiting longer for larger batches
        target_ratio = 0.5  # Target wait_time / inference_time ratio
        current_ratio = avg_wait_time / max(avg_inference_time, 0.001)

        if current_ratio > target_ratio * 1.2:
            # Waiting too long compared to inference time
            self.max_wait_time = max(0.001, self.max_wait_time * 0.9)
        elif current_ratio < target_ratio * 0.8:
            # Not waiting long enough for efficient batching
            self.max_wait_time = min(0.05, self.max_wait_time * 1.1)

        # print(
        #     f"Auto-tuning: min_batch_size={self.min_batch_size}, "
        #     f"max_wait_time={self.max_wait_time:.4f}s, "
        #     f"avg_batch={avg_batch_size:.1f}, throughput={throughput:.1f} obs/s"
        # )

    async def update_inference_worker(self, params: Params):
        """Update or start the inference worker with new parameters."""
        # Check if params are different
        if self.current_params is params:
            # Same params, no need to restart
            if self.inference_task is None or self.inference_task.done():
                # Task is not running, start it
                self.stop_inference.clear()
                self.inference_task = asyncio.create_task(
                    self.batch_inference_worker(params)
                )
            return

        # Different params, need to restart the worker
        await self.stop_inference_worker()

        # Update params and start new worker
        self.current_params = params
        self.stop_inference.clear()
        self.inference_task = asyncio.create_task(self.batch_inference_worker(params))

    async def stop_inference_worker(self):
        """Stop the inference worker task."""
        if self.inference_task and not self.inference_task.done():
            # Signal the worker to stop
            self.stop_inference.set()

            # Wait for the worker to finish (with timeout)
            try:
                await asyncio.wait_for(self.inference_task, timeout=1.0)
            except asyncio.TimeoutError:
                # Force cancel if it doesn't stop in time
                self.inference_task.cancel()
                try:
                    await self.inference_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                print(f"Error stopping inference worker: {e}")

    async def get_actor_inference(
        self, params: Params, env_step: EnvStep, history_step: HistoryStep
    ):
        """Get network inference for an actor, potentially batched with others."""
        # Ensure inference worker is running with correct params
        await self.update_inference_worker(params)

        # Create a unique request ID
        request_id = id(env_step)

        # Submit observation to the inference queue
        await self.inference_queue.put((request_id, (env_step, history_step)))

        # Wait for result
        while True:
            async with self.inference_lock:
                if request_id in self.inference_results:
                    pi, log_pi = self.inference_results.pop(request_id)
                    # Handle the case where we got dummy results during shutdown
                    if pi is None:
                        # Fall back to direct inference
                        env_step_jax = as_jax_arr(env_step)
                        history_step_jax = as_jax_arr(history_step)
                        pi, log_pi = self._network_jit_apply(
                            params, env_step_jax, history_step_jax
                        )
                    break
            await asyncio.sleep(0.001)  # Small wait to avoid busy loop

        # Sample action from policy
        action = np.apply_along_axis(
            lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )

        return action, ActorStep(
            action=action, policy=pi, log_policy=log_pi, rewards=()
        )

    def actor_step(self, params: Params, env_step: EnvStep, history_step: HistoryStep):
        """Legacy actor_step - primarily for compatibility, prefer get_actor_inference"""
        env_step = as_jax_arr(env_step)
        history_step = as_jax_arr(history_step)
        pi, log_pi = self._network_jit_apply(params, env_step, history_step)
        action = np.apply_along_axis(
            lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
        )
        return action, ActorStep(
            action=action, policy=pi, log_policy=log_pi, rewards=()
        )

    async def collect_trajectory(
        self, params, env: SinglePlayerEnvironment, resolution: int = 32
    ):
        """Collect a single trajectory using batched inference."""
        # Register this environment as active
        async with self.active_envs_lock:
            self.active_envs += 1

        try:
            trajectory = []
            ex, hx = await env.reset()
            steps = 0

            while True:
                # Update collection phase every few steps
                if steps % 10 == 0:
                    await self.update_collection_phase(steps)

                pex = ex
                phx = hx
                steps += 1

                a, actor_step = await self.get_actor_inference(params, pex, phx)
                ex, hx = await env.step(a.item())

                timestep = TimeStep(
                    env=pex,
                    history=phx,
                    actor=ActorStep(
                        action=actor_step.action,
                        policy=actor_step.policy,
                        log_policy=actor_step.log_policy,
                        rewards=pex.rewards,
                    ),
                )
                trajectory.append(timestep)

                if env.is_done():
                    # Store episode length for future estimation
                    self.episode_lengths.append(steps)
                    # Keep only recent history
                    if len(self.episode_lengths) > 100:
                        self.episode_lengths.pop(0)

                    # Mark environment as inactive
                    async with self.active_envs_lock:
                        self.active_envs -= 1
                    break

            trajectory_length = (
                (len(trajectory) + resolution - 1) // resolution
            ) * resolution
            self.batch_length = max(trajectory_length, self.batch_length)
            await self.sync_point.wait()

            # Pad trajectory to batch_length
            while len(trajectory) < self.batch_length:
                ex, hx = await env.step(a.item())
                timestep = TimeStep(
                    env=ex,
                    history=hx,
                    actor=ActorStep(
                        action=actor_step.action,
                        policy=actor_step.policy,
                        log_policy=actor_step.log_policy,
                        rewards=ex.rewards,
                    ),
                )
                trajectory.append(timestep)

            return stack_steps(trajectory)
        finally:
            # Ensure environment is marked as inactive if there's an exception
            async with self.active_envs_lock:
                if env.is_done():  # Only decrement if we haven't already
                    self.active_envs -= 1

    async def collect_trajectories(self, params: Params, resolution: int = 32):
        """Collect trajectories from all environments in parallel."""
        # Reset active environments counter at the start of collection
        async with self.active_envs_lock:
            self.active_envs = 0

        # Reset collection phase at the start of collection
        async with self.phase_lock:
            self.collection_phase = "start"

        return await asyncio.gather(
            *[self.collect_trajectory(params, env, resolution) for env in self.envs]
        )

    def collect_batch_trajectory(
        self, params: Params, resolution: int = 32
    ) -> TimeStep:
        """Collect and stack trajectories from all environments."""
        try:
            trajectories = self.loop.run_until_complete(
                self.collect_trajectories(params, resolution)
            )
            self.state_index = 0
            return as_jax_arr(stack_steps(trajectories, 1))
        finally:
            # Ensure inference worker is stopped when collection is complete
            self.loop.run_until_complete(self.stop_inference_worker())

    async def benchmark_environment_speed(self, params, num_steps=100):
        """Measure how frequently observations arrive."""
        env = self.envs[0]  # Use first environment for testing
        arrival_times = []

        ex, hx = await env.reset()
        start_time = time.time()

        for _ in range(num_steps):
            # Use direct inference for benchmarking
            env_step_jax = as_jax_arr(ex)
            history_step_jax = as_jax_arr(hx)
            pi, log_pi = self._network_jit_apply(params, env_step_jax, history_step_jax)

            action = np.apply_along_axis(
                lambda x: np.random.choice(range(pi.shape[-1]), p=x), axis=-1, arr=pi
            )

            ex, hx = await env.step(action.item())
            arrival_times.append(time.time() - start_time)

        # Calculate median time between observations
        intervals = [
            arrival_times[i] - arrival_times[i - 1]
            for i in range(1, len(arrival_times))
        ]
        median_interval = statistics.median(intervals)
        # print(f"Median time between observations: {median_interval*1000:.2f}ms")

        return median_interval


class EvalBatchCollector(BatchCollectorV2):
    def __init__(self, network: nn.Module, batch_size: int):
        super().__init__(
            network, batch_size, BatchSinglePlayerEnvironment, finetune=True
        )


def main():
    batch_progress = tqdm(desc="Batch: ")
    game_progress = tqdm(desc="Games: ")
    state_progress = tqdm(desc="States: ")

    num_envs = 4
    network = get_dummy_model()
    # training_env = SingleTrajectoryTrainingBatchCollector(network, num_envs)
    training_env = DoubleTrajectoryTrainingBatchCollector(network, num_envs)
    # evaluation_env = EvalBatchCollector(network, 4)

    ex, hx = get_ex_step()
    params = network.init(jax.random.PRNGKey(42), ex, hx)

    np.zeros(num_envs)
    batch = training_env.collect_batch_trajectory(params)

    while True:

        batch = training_env.collect_batch_trajectory(params)
        # with open("rlenv/ex_batch", "wb") as f:
        #     pickle.dump(batch, f)

        collect_batch_telemetry_data(batch)

        # training_progress.update(batch.env.valid.sum())
        # batch = evaluation_env.collect_batch_trajectory(params)
        # evaluation_progress.update(batch.env.valid.sum())

        # win_rewards = np.sign(
        #     (batch.actor.win_rewards[..., 0] * batch.env.valid).sum(0)
        # )
        # avg_reward = avg_reward * (1 - tau) + win_rewards.astype(float) * tau
        state_progress.update(batch.env.valid.sum())
        game_progress.update(num_envs)
        batch_progress.update(1)
        # batch_progress.set_description(np.array2string(avg_reward))


if __name__ == "__main__":
    main()
