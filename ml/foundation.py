import json
import os
import pickle
import jax
import jax.numpy as jnp
import optax

from flax.training import train_state
from typing import Any, Dict, Sequence

from ml.arch.config import get_model_cfg
from ml.arch.encoder import SupervisedBackbone
from ml.utils import breakpoint_if_nonfinite

from rlenv.env import get_ex_step, process_state
from rlenv.interfaces import EnvStep
from rlenv.protos.enums_pb2 import MovesEnum
from rlenv.protos.state_pb2 import State
from rlenv.utils import stack_steps


# Load and preprocess the Iris dataset
def load_and_preprocess_data(
    formats: Sequence[str] = ("gen3randombattle",),
) -> EnvStep:

    steps = []

    for format in formats:
        format_root = os.path.join("replays", "data", format)
        chunk_fpaths = [
            os.path.join(format_root, fname)
            for fname in os.listdir(format_root)
            if fname.endswith(".bin")
        ]
        metadata_fpaths = [fpath.replace(".bin", ".json") for fpath in chunk_fpaths]

        for chunk_path, metadata_path in zip(chunk_fpaths, metadata_fpaths):
            with open(metadata_path, "r") as mf:
                state_lengths = json.load(mf)

            with open(chunk_path, "rb") as cf:
                chunk_data = cf.read()
                offset = 0
                for length in state_lengths:
                    state_bytes = chunk_data[offset : offset + length]
                    state = State.FromString(state_bytes)
                    step = process_state(state)

                    steps.append(step)
                    offset += length

    return stack_steps(steps)


# Create a train state
def create_train_state(
    rng: jax.random.PRNGKey, model: SupervisedBackbone, learning_rate: float
) -> train_state.TrainState:
    params = model.init(rng, get_ex_step())
    tx = optax.adamw(learning_rate, weight_decay=1e-5)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def normalize(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return (values * mask).sum() / mask.sum().clip(min=1)


# Define the loss function
def compute_loss(apply_fn: Any, params: Any, batch: EnvStep) -> jnp.ndarray:
    next_turn, next_move, next_action, value = jax.vmap(apply_fn, (None, 0))(
        params, batch
    )

    rewards = jnp.where(
        batch.player_id == 0, batch.rewards[..., 0], batch.rewards[..., 1]
    )
    mask = batch.valid[:-1]

    next_move_loss = normalize(
        optax.softmax_cross_entropy(
            next_move[:-1], jax.nn.one_hot(batch.prev_move[1:], next_move.shape[-1])
        ),
        mask & batch.prev_move[1:] != MovesEnum.moves_none,
    )
    next_action_loss = normalize(
        optax.softmax_cross_entropy(
            next_action[:-1],
            jax.nn.one_hot(batch.prev_action[1:], next_action.shape[-1]),
        ),
        mask,
    )
    value_loss = normalize(jnp.square(value.reshape(-1) - rewards)[:-1], mask)

    loss = next_move_loss + next_action_loss + value_loss

    return loss, dict(
        next_move_loss=next_move_loss,
        next_action_loss=next_action_loss,
        value_loss=value_loss,
    )


# Training step
@jax.jit
def train_step(
    state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
) -> train_state.TrainState:
    def loss_fn(params):
        return compute_loss(state.apply_fn, params, batch)

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, info, state


def format_info(info: Dict[str, Any]):
    formatted_info = ""
    for key, value in info.items():
        formatted_info += f"{key}: {value:.4f}, "
    return formatted_info[:-2]


# Training loop
def train_model(num_epochs: int, batch_size: int, learning_rate: float) -> None:
    data = load_and_preprocess_data()
    print(len(data.valid))
    key = jax.random.key(0)

    model_cfg = get_model_cfg().encoder
    model = SupervisedBackbone(model_cfg)
    state = create_train_state(key, model, learning_rate)

    for epoch in range(num_epochs):
        # Shuffle the data
        key, subkey = jax.random.split(key, 2)
        permutation = jax.random.permutation(subkey, len(data.valid))
        data_shuffled: EnvStep = jax.tree_util.tree_map(
            lambda xs: xs[permutation], data
        )

        for i in range(0, len(data_shuffled.valid), batch_size):
            batch = jax.tree_util.tree_map(lambda xs: xs[i : i + batch_size], data)
            loss, info, state = train_step(state, batch)
            # if jnp.isnan(loss).item():
            #     with open("replays/bad_batch", "wb") as f:
            #         pickle.dump(batch, f)
            #     exit(0)
            print(f"Epoch {epoch+1}, {format_info(info)}", end="\r")

        with open("replays/foundation.ckpt", "wb") as f:
            pickle.dump(state.params, f)

    print("Training complete.")


def main() -> None:
    num_epochs = 1000
    batch_size = 512
    learning_rate = 1e-3
    train_model(num_epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()
