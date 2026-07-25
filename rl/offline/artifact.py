"""Loading offline critic artifacts outside the offline trainer.

Two consumption modes:

1. Warm start: set ``Porygon2LearnerConfig.init_params_ckpt_path`` to an
   artifact directory and launch the RL learner with
   ``LOAD_STATE_MODE=params``. The artifact's encoder + v_head subtrees
   merge into the fresh RL init via merge_params; the policy heads keep
   their fresh initialization. Combine with
   ``player_frozen_param_patterns`` to pin warm-started subtrees.

2. Learned state potential: set
   ``Porygon2LearnerConfig.offline_critic_ckpt_path``. The learner loads
   the critic once, keeps its params outside the optimizer (frozen by
   construction), and feeds Φ(s) into compute_player_targets' potential
   advantage channel, gated by ``player_potential_advantage_coef_fn``.

The critic is public-only by construction: it operates exclusively on the
recurrent history pathway (Encoder.encode_history -> PerSlotHistoryEncoder)
plus an offline-only outcome head. Private fields, movesets, and action
masks are architecturally unreachable, and the history inputs are built
from the same protocol events offline and live, so the frozen Φ carries no
train/serve distribution bias into RL training.
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
        return jax.lax.stop_gradient(value_head.expectation.astype(jnp.float32))

    return potential
