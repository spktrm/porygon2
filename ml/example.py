import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Tuple


# Load and preprocess the Iris dataset
def load_and_preprocess_data() -> (
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return jnp.array(X_train), jnp.array(X_test), jnp.array(y_train), jnp.array(y_test)


# Define the model
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=3)(x)
        return x


# Create a train state
def create_train_state(
    rng: jax.random.PRNGKey, model: nn.Module, learning_rate: float
) -> train_state.TrainState:
    params = model.init(rng, jnp.ones((1, 4)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Define the loss function
def compute_loss(
    apply_fn: Any, params: Any, batch: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    logits = apply_fn(params, batch["X"])
    loss = optax.softmax_cross_entropy(
        logits, jax.nn.one_hot(batch["y"], num_classes=3)
    ).mean()
    return loss


# Define the accuracy function
def compute_accuracy(
    apply_fn: Any, params: Any, X: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    logits = apply_fn(params, X)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)


# Training step
@jax.jit
def train_step(
    state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
) -> train_state.TrainState:
    def loss_fn(params):
        return compute_loss(state.apply_fn, params, batch)

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


# Shuffle and batch data (without JIT compilation)
def shuffle_and_batch_data(
    rng: jax.random.PRNGKey, X: jnp.ndarray, y: jnp.ndarray, batch_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    permutation = jax.random.permutation(rng, len(X))
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]
    num_batches = len(X) // batch_size
    X_batches = jnp.array_split(X_shuffled[: num_batches * batch_size], num_batches)
    y_batches = jnp.array_split(y_shuffled[: num_batches * batch_size], num_batches)
    return jnp.array(X_batches), jnp.array(y_batches)


# Training loop
def train_model(
    model: nn.Module, num_epochs: int, batch_size: int, learning_rate: float
) -> None:
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate)

    for epoch in range(num_epochs):
        # Shuffle and batch the data
        rng, shuffle_key = jax.random.split(rng)
        X_train_batches, y_train_batches = shuffle_and_batch_data(
            shuffle_key, X_train, y_train, batch_size
        )

        for batch_X, batch_y in zip(X_train_batches, y_train_batches):
            batch = {"X": batch_X, "y": batch_y}
            state = train_step(state, batch)

        train_acc = compute_accuracy(state.apply_fn, state.params, X_train, y_train)
        test_acc = compute_accuracy(state.apply_fn, state.params, X_test, y_test)
        print(
            f"Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}"
        )

    print("Training complete.")


def main() -> None:
    model = MLP()
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.01
    train_model(model, num_epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()
