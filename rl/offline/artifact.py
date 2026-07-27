"""Loading offline critic artifacts for the RL learner.

The offline critic is a STANDALONE model: it shares Encoder code with the
RL model for convenience, but its trained params never enter the RL
network. The single consumption mode is the learned state potential — set
``Porygon2LearnerConfig.offline_critic_ckpt_path`` and the learner loads
the critic once, holds its params outside the train state (never in the
optimizer, never donated, stop-gradient at use), and feeds Φ(s) into
compute_player_targets' potential advantage channel, gated by
``player_potential_advantage_coef_fn``. The RL model itself trains fully
from scratch — no frozen or warm-started subtrees.

The critic is public-only by construction: it operates exclusively on the
recurrent history pathway plus an offline-only antisymmetric probe, so
private fields, movesets, and action masks are architecturally
unreachable, and the history inputs are built from the same protocol
events offline and live — the frozen Φ carries no train/serve
distribution bias into RL training.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from rl.environment.interfaces import PlayerActorInput
from rl.learner import checkpoint as checkpoint_lib
from rl.model.config import get_player_model_config
from rl.model.utils import Params
from rl.offline.model import Porygon2OfflineCritic


def load_critic_params(artifact_path: str) -> Params:
    return checkpoint_lib.load_component(artifact_path, "player", "params")


def make_potential_apply(
    generation: int,
) -> Callable[[Params, PlayerActorInput], jax.Array]:
    """Builds the frozen-critic potential: (params, (T, B, ...) actor input)
    -> Φ in [-1, 1] with shape (T, B), float32, stop-gradient. The critic
    reads only the public history pathway, so no input projection exists
    or is needed."""
    model = Porygon2OfflineCritic(get_player_model_config(generation, train=False))
    apply_fn = jax.vmap(model.apply, in_axes=(None, 1), out_axes=1)

    def potential(params: Params, actor_input: PlayerActorInput) -> jax.Array:
        value_head = apply_fn(params, actor_input)
        return jax.lax.stop_gradient(
            value_head.expectation.astype(jnp.float32)
        )

    return potential