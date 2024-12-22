import os
from collections import deque

import jax
import numpy as np
from tqdm import tqdm

from rlenv.env import process_state
from rlenv.interfaces import EnvStep, TimeStep
from rlenv.protos.features_pb2 import FeatureMoveset
from rlenv.protos.state_pb2 import Dataset
from rlenv.utils import stack_steps


class ReplayBuffer:
    def __init__(self, max_buffer_size: int = 1_000):
        self.buffer = deque()
        self.max_buffer_size = max_buffer_size

    def __len__(self):
        return len(self.buffer)

    def add_trajectories(self, batch: TimeStep):
        valids = batch.env.valid.sum(axis=0, keepdims=True)
        batch = jax.tree.map(lambda x: np.resize(x, (1000, *x.shape[1:])), batch)
        batch = TimeStep(
            env=EnvStep(
                ts=batch.env.ts,
                draw_ratio=batch.env.draw_ratio,
                valid=valids > np.arange(1000)[:, None].repeat(valids.shape[1], axis=1),
                draw=batch.env.draw,
                turn=batch.env.turn,
                game_id=batch.env.game_id,
                player_id=batch.env.player_id,
                seed_hash=batch.env.seed_hash,
                moveset=batch.env.moveset,
                legal=batch.env.legal,
                team=batch.env.team,
                win_rewards=batch.env.win_rewards,
                fainted_rewards=batch.env.fainted_rewards,
                switch_rewards=batch.env.switch_rewards,
                longevity_rewards=batch.env.longevity_rewards,
                hp_rewards=batch.env.hp_rewards,
                history_edges=batch.env.history_edges,
                history_entities=batch.env.history_entities,
                history_side_conditions=batch.env.history_side_conditions,
                history_field=batch.env.history_field,
                heuristic_action=batch.env.heuristic_action,
            ),
            actor=batch.actor,
        )
        for batch_index in range(batch.env.valid.shape[1]):
            stretched: TimeStep = jax.tree.map(lambda x: x[:, batch_index], batch)
            self.buffer.append(stretched)

        while len(self) > self.max_buffer_size:
            self.buffer.popleft()

    def sample(self, batch_size: int, resolution: int = 32) -> TimeStep:
        indices = np.random.choice(
            range(len(self.buffer)), size=batch_size, replace=False
        )
        samples = [self.buffer[index] for index in indices]
        batch: TimeStep = stack_steps(samples, axis=1)
        max_size = batch.env.valid.sum(0).max()
        max_size_quant = resolution * ((max_size // resolution) + 1)
        return jax.tree.map(lambda x: x[:max_size_quant], batch)


class OfflineReplayBuffer:
    def __init__(self, batch_size: int = 512):
        self.batch_size = batch_size

    @classmethod
    def from_replay_dir(cls, replay_dir: str) -> "OfflineReplayBuffer":
        buffer = cls()

        buffer.samples = []
        buffer.labels = []
        buffer.values = []
        buffer.progress = tqdm()

        dataset_files = [f for f in os.listdir(replay_dir) if f.endswith(".dat")]
        for dataset_path in dataset_files:
            full_path = os.path.join(replay_dir, dataset_path)
            with open(full_path, "rb") as f:
                dataset_binary = f.read()
            dataset = Dataset.FromString(dataset_binary)
            buffer._process_dataset(dataset)
            del dataset

        buffer.samples: EnvStep = stack_steps(buffer.samples)  # type: ignore
        buffer.labels = np.array(buffer.labels)
        buffer.values = np.array(buffer.values)

        is_move = (
            buffer.samples.moveset[..., 0, :, FeatureMoveset.MOVESET_ACTION_TYPE] == 0
        )
        is_switch = (
            buffer.samples.moveset[..., 0, :, FeatureMoveset.MOVESET_ACTION_TYPE] == 1
        )

        true_labels = buffer.samples.moveset[
            ..., 0, :, FeatureMoveset.MOVESET_ACTION_ID
        ] == np.expand_dims(buffer.labels, axis=-1)

        all_unknown = (~true_labels).all(axis=-1)

        true_labels[all_unknown] = np.where(is_move[all_unknown], 1, 0)
        true_labels[all_unknown] = np.where(is_switch[all_unknown], 1, 0)

        true_labels = true_labels / true_labels.sum(axis=-1, keepdims=True)
        buffer.targets = true_labels
        return buffer

    @staticmethod
    def split_train_test(buffer1, split: float = 0.2):
        buffer2 = OfflineReplayBuffer()
        indices = np.arange(len(buffer1))
        np.random.shuffle(indices)

        cutoff = int(split * len(buffer1))
        indices1 = indices[:cutoff]
        indices2 = indices[cutoff:]

        buffer2.samples = jax.tree.map(lambda x: x[indices1], buffer1.samples)
        buffer1.samples = jax.tree.map(lambda x: x[indices2], buffer1.samples)

        buffer2.labels = buffer1.labels[indices1]
        buffer1.labels = buffer1.labels[indices2]

        buffer2.values = buffer1.values[indices1]
        buffer1.values = buffer1.values[indices2]

        buffer2.targets = buffer1.targets[indices1]
        buffer1.targets = buffer1.targets[indices2]

        return buffer1, buffer2

    def __len__(self):
        return len(self.labels)

    def _process_dataset(self, dataset: Dataset):
        for trajectory in dataset.trajectories:
            for state, action, reward in zip(
                trajectory.states, trajectory.actions, trajectory.rewards
            ):
                self.samples.append(process_state(state))
                self.labels.append(action)
                self.values.append(reward)
            self.progress.update(len(trajectory.states))

    def __iter__(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        for i in range(len(self) // self.batch_size):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]

            samples = jax.tree.map(lambda x: x[batch_indices], self.samples)
            targets = self.targets[batch_indices]
            labels = self.labels[batch_indices]
            values = self.values[batch_indices]

            yield samples, targets, labels, values
