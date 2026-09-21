"""Price observable shaping after fitting only its fresh linear decoder."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl import checkpoint
from rl.environment.consequence_labels import (
    NUM_OBSERVABLE_LOGITS,
    observed_consequences,
)
from rl.environment.utils import acted_rows
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.offline import harness
from rl.online.artifact import (
    merge_opt_state,
    player_model_config_for,
    player_optimiser,
)
from rl.online.config import get_learner_config
from rl.online.training.batching import stack_batch
from rl.online.training.observable_consequence import observable_terms
from rl.probes.separation_probe import actor_input_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = get_learner_config()
    network = get_player_model(player_model_config_for(config))
    params = harness.load_params(args.checkpoint)
    old_leaves = dict(
        (jax.tree_util.keystr(path), np.asarray(leaf))
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    )
    width = network.cfg.action_head.qk_size + 2
    decoder = dict(
        kernel=jnp.zeros((width, NUM_OBSERVABLE_LOGITS)),
        bias=jnp.zeros(NUM_OBSERVABLE_LOGITS),
    )
    params["params"]["consequence"]["observable_outcomes"] = decoder
    optimiser = player_optimiser(config)
    loaded_state = checkpoint.load_component(args.checkpoint, "player", "opt_state")
    adam_count = loaded_state.inner_states["consequence"].inner_state[1][0].count
    merged_state = merge_opt_state(jax.jit(optimiser.init)(params), loaded_state)
    restored = dict(
        (jax.tree_util.keystr(path), np.asarray(leaf))
        for path, leaf in jax.tree_util.tree_flatten_with_path(merged_state)[0]
    )
    for path, leaf in jax.tree_util.tree_flatten_with_path(loaded_state)[0]:
        np.testing.assert_array_equal(restored[jax.tree_util.keystr(path)], leaf)
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = jax.tree_util.keystr(path)
        if name in old_leaves:
            np.testing.assert_array_equal(leaf, old_leaves[name])
    print(
        "Checkpoint verification: every existing weight, optimiser moment and counter matches exactly",
        flush=True,
    )
    del loaded_state, merged_state, restored, old_leaves
    params = jax.device_put(params)
    forward = jax.vmap(network.apply, in_axes=(None, 1, 1, None), out_axes=1)

    def features(variables, batch):
        return forward(
            variables,
            actor_input_of(batch),
            batch.player_transitions.agent_output.actor_output,
            HeadParams(),
        ).consequence_inputs.action_features

    read_features = jax.jit(features)
    read_targets = jax.jit(observed_consequences)

    def labels(batch):
        transitions = batch.player_transitions
        return read_targets(
            transitions.env_output,
            batch.player_history,
            batch.player_packed_history,
            transitions.agent_output.actor_output.action_head.action_index,
            jnp.asarray(acted_rows(np.asarray(transitions.env_output.done))),
        )

    def decode(decoder_params, action_features):
        return nn.Dense(
            NUM_OBSERVABLE_LOGITS, dtype=action_features.dtype, parent=None
        ).apply({"params": decoder_params}, action_features)

    fitting = optax.adam(
        config.player_learning_rate,
        b1=config.player_adam.b1,
        b2=config.player_adam.b2,
        eps=config.player_adam.eps,
    )
    fitting_state = fitting.init(decoder)
    fitting_state = (
        fitting_state[0]._replace(count=jnp.asarray(adam_count)),
    ) + fitting_state[1:]

    @jax.jit
    def fit(decoder_params, optimiser_state, action_features, targets):
        loss, gradients = jax.value_and_grad(
            lambda weights: config.player_observable_coef
            * observable_terms(decode(weights, action_features), targets)[0]
        )(decoder_params)
        updates, optimiser_state = fitting.update(
            gradients, optimiser_state, decoder_params
        )
        return optax.apply_updates(decoder_params, updates), optimiser_state, loss

    games = harness.load(f"{args.root}/games.pkl")
    batches = [
        stack_batch([side[0] for side in games[start : start + 2]])
        for start in (0, 2, 4, 6)
    ]
    labelled = [(read_features(params, batch), labels(batch)) for batch in batches]
    for iteration in range(200):
        action_features, targets = labelled[iteration % 2]
        decoder, fitting_state, loss = fit(
            decoder, fitting_state, action_features, targets
        )
    print(f"Decoder-only calibration fit: loss={float(loss):.5f}", flush=True)

    @jax.jit
    def measure(variables, batch, targets):
        def objective(variables):
            return observable_terms(
                decode(decoder, features(variables, batch)), targets
            )

        (loss, logs), gradients = jax.value_and_grad(objective, has_aux=True)(variables)
        return dict(
            loss=loss,
            encoder_norm=optax.global_norm(gradients["params"]["encoder"]),
            action_head_norm=optax.global_norm(gradients["params"]["action_head"]),
            **logs,
        )

    results = []
    for batch, (_, targets) in zip(batches[2:], labelled[2:]):
        results.append(
            {
                name: float(value)
                for name, value in measure(params, batch, targets).items()
            }
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
