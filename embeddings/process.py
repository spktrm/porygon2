import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, random
from jax.example_libraries import optimizers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class JAXEntityEmbedding(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_columns,
        continuous_columns,
        embedding_dim=5,
        learning_rate=0.01,
        epochs=10,
        batch_size=32,
    ):
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoders = {}
        self.params = None
        self.key = random.PRNGKey(0)

    def init_params(self, input_shapes):
        embeddings = {}
        for col, shape in input_shapes.items():
            k1, k2 = random.split(self.key)
            embeddings[col] = random.normal(k1, (shape, self.embedding_dim))
            self.key = k2

        k1, k2 = random.split(self.key)
        dense = random.normal(
            k1,
            (
                sum(input_shapes.values()) * self.embedding_dim
                + len(self.continuous_columns),
                1,
            ),
        )
        bias = random.normal(k2, (1,))

        return {"embeddings": embeddings, "dense": dense, "bias": bias}

    def model(self, params, inputs, continuous):
        embeddings = []
        for col, x in inputs.items():
            embed = params["embeddings"][col][x]
            embeddings.append(embed.reshape(-1, self.embedding_dim))

        concatenated = jnp.concatenate(embeddings + [continuous], axis=1)
        return jnp.dot(concatenated, params["dense"]) + params["bias"]

    def loss(self, params, inputs, continuous, targets):
        preds = self.model(params, inputs, continuous)
        return jnp.mean((preds - targets) ** 2)

    @partial(jit, static_argnums=(0,))
    def update(self, params, inputs, continuous, targets, opt_state):
        value, grads = jax.value_and_grad(self.loss)(
            params, inputs, continuous, targets
        )
        updates, opt_state = self.opt_update(0, grads, opt_state)
        params = optimizers.apply_updates(params, updates)
        return params, opt_state, value

    def fit(self, X, y):
        # Encode categorical variables
        for col in self.categorical_columns:
            self.encoders[col] = {val: i for i, val in enumerate(X[col].unique())}

        # Prepare input shapes
        input_shapes = {
            col: len(self.encoders[col]) for col in self.categorical_columns
        }

        # Initialize parameters
        self.params = self.init_params(input_shapes)

        # Prepare data
        X_cat = {
            col: jnp.array([self.encoders[col].get(val, -1) for val in X[col]])
            for col in self.categorical_columns
        }
        X_cont = jnp.array(X[self.continuous_columns])
        y = jnp.array(y).reshape(-1, 1)

        # Initialize optimizer
        self.opt_init, self.opt_update, get_params = optimizers.adam(self.learning_rate)
        opt_state = self.opt_init(self.params)

        # Training loop
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_cat = {k: v[i : i + self.batch_size] for k, v in X_cat.items()}
                batch_cont = X_cont[i : i + self.batch_size]
                batch_y = y[i : i + self.batch_size]

                self.params, opt_state, loss = self.update(
                    self.params, batch_cat, batch_cont, batch_y, opt_state
                )

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return self

    def transform(self, X):
        X_cat = {
            col: jnp.array([self.encoders[col].get(val, -1) for val in X[col]])
            for col in self.categorical_columns
        }
        X_cont = jnp.array(X[self.continuous_columns])

        embeddings = []
        for col, x in X_cat.items():
            embed = self.params["embeddings"][col][x]
            embeddings.append(embed)

        return jnp.concatenate(embeddings + [X_cont], axis=1)


def preprocess_and_combine(df, categorical_columns, continuous_columns):
    # Normalize continuous variables
    scaler = StandardScaler()
    df_continuous = pd.DataFrame(
        scaler.fit_transform(df[continuous_columns]), columns=continuous_columns
    )

    # Apply entity embedding to categorical variables and combine with continuous
    embedder = JAXEntityEmbedding(categorical_columns, continuous_columns)
    combined_features = embedder.fit(
        df[categorical_columns + continuous_columns], df["num"]
    )
    combined_features = embedder.fit_transform(df)

    return combined_features


def identify_column_types(df: pd.DataFrame, discrete_threshold: int = 20) -> tuple:
    """
    Identify discrete and continuous columns in a DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame
    discrete_threshold (int): Maximum number of unique values for a column to be considered discrete

    Returns:
    tuple: Lists of discrete and continuous column names
    """
    discrete_columns = []
    continuous_columns = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < discrete_threshold:
                discrete_columns.append(col)
            else:
                continuous_columns.append(col)
        else:
            discrete_columns.append(col)

    return discrete_columns, continuous_columns


def main(df: pd.DataFrame) -> np.ndarray:
    discrete_cols, continuous_cols = identify_column_types(df)
    return preprocess_and_combine(df, discrete_cols, continuous_cols)


# Example usage
if __name__ == "__main__":

    with open("data/data/gen3/species.json", "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    embeddings = main(df)

    print(f"Generated embeddings shape: {embeddings.shape}")
