"""How much gradient each consequence loss WOULD send into the encoder at full
trunk gradient, per unit coefficient -- the number the two coefficients are
set from before `player_consequence_trunk_grad` leaves 0.

The reference is the live run's own `player_encoder_gradient_norm` (the
policy and value losses' encoder gradient, read while the consequence model
is still an observer). A coefficient is then `share * reference / norm`. Not
a training panel: it costs a backward through the trunk per loss.

    env/bin/python -m rl.probes.consequence_calibration \\
        --root runtime/tactical-cohort-20260920 --checkpoint ckpts/gen9/ckpt_N
"""

import argparse
import dataclasses
import functools
import json
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.environment.utils import acted_rows
from rl.model.consequence import CONSEQUENCE_NOISE_SIZE
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.offline import harness
from rl.online.artifact import consequence_apply_fn, player_model_config_for
from rl.online.config import get_learner_config
from rl.online.training.batching import stack_batch
from rl.online.training.consequence import consequence_terms
from rl.probes.separation_probe import actor_input_of

logger = logging.getLogger(__name__)
LOSSES = ("mean", "sampler")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = get_learner_config()
    network = get_player_model(player_model_config_for(config))
    variables = jax.device_put(harness.load_params(arguments.checkpoint))
    forward = jax.vmap(network.apply, in_axes=(None, 1, 1, None), out_axes=1)
    consequences = consequence_apply_fn(network)
    unit = dict(
        mean=dataclasses.replace(
            config,
            player_consequence_mean_coef=1.0,
            player_consequence_sampler_coef=0.0,
        ),
        sampler=dataclasses.replace(
            config,
            player_consequence_mean_coef=0.0,
            player_consequence_sampler_coef=1.0,
        ),
    )

    def encoder_gradient_norms(params, actor_input, actor_output, acted, log_prob, key):
        def loss(params, which):
            prediction = forward(params, actor_input, actor_output, HeadParams())
            inputs = prediction.consequence_inputs
            noise = jax.random.normal(
                key, (arguments.draws,) + acted.shape + (CONSEQUENCE_NOISE_SIZE,)
            )
            value, _ = consequence_terms(
                consequences(params, inputs, noise),
                inputs,
                actor_input.env.info,
                acted,
                prediction.state_value_head.expectation,
                log_prob,
                unit[which],
            )
            return value

        norms = {}
        for which in LOSSES:
            gradients = jax.grad(functools.partial(loss, which=which))(params)
            norms[which] = optax.global_norm(gradients["params"]["encoder"])
        return norms

    measure = jax.jit(encoder_gradient_norms)
    sides = harness.load(f"{arguments.root}/games.pkl")
    chunks = [chunk for side in sides for chunk in side]
    rng = np.random.default_rng(0)
    norms = {which: [] for which in LOSSES}
    for index in range(arguments.batches):
        picked = rng.choice(len(chunks), arguments.batch, replace=False)
        batch = stack_batch([chunks[chosen] for chosen in picked])
        transitions = batch.player_transitions
        actor_output = transitions.agent_output.actor_output
        measured = measure(
            variables,
            actor_input_of(batch),
            actor_output,
            jnp.asarray(acted_rows(np.asarray(transitions.env_output.done))),
            actor_output.action_head.log_prob,
            jax.random.key(index),
        )
        for which in LOSSES:
            norms[which].append(float(measured[which]))
        logger.info(
            "batch %d: %s", index, {k: round(v[-1], 4) for k, v in norms.items()}
        )
    print(
        json.dumps(
            {
                which: dict(
                    median=float(np.median(values)),
                    mean=float(np.mean(values)),
                    low=float(np.min(values)),
                    high=float(np.max(values)),
                )
                for which, values in norms.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
