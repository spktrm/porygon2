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

The critic was trained on the PUBLIC VIEW ONLY — private team
all-unspecified, own moveset all-PAD, all-ones action mask, request type
always MOVE — so live observations are projected with
``mask_to_public_view`` before evaluation; otherwise Φ is computed
off-distribution.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from rl.environment.interfaces import PlayerActorInput
from rl.environment.protos.enums_pb2 import MovesEnum
from rl.environment.protos.features_pb2 import InfoFeature, MovesetFeature, RequestType
from rl.learner import checkpoint as checkpoint_lib
from rl.model.config import get_player_model_config
from rl.model.utils import Params
from rl.offline.model import Porygon2OfflineCritic


def load_critic_params(artifact_path: str) -> Params:
    return checkpoint_lib.load_component(artifact_path, "player", "params")


def mask_to_public_view(actor_input: PlayerActorInput) -> PlayerActorInput:
    """Projects a live observation onto the offline critic's training
    distribution (see module docstring), matching what the replay exporter
    emits for a request-less spectator log."""
    env = actor_input.env

    my_moveset = jnp.zeros_like(env.my_moveset)
    my_moveset = my_moveset.at[..., MovesetFeature.MOVESET_FEATURE__MOVE_ID].set(
        MovesEnum.MOVES_ENUM___PAD
    )

    info = env.info
    info = info.at[..., InfoFeature.INFO_FEATURE__REQUEST_TYPE].set(
        RequestType.REQUEST_TYPE__MOVE
    )
    info = info.at[..., InfoFeature.INFO_FEATURE__HAS_PREV_ACTION].set(0)
    info = info.at[..., InfoFeature.INFO_FEATURE__PREV_ACTION_SRC].set(0)
    info = info.at[..., InfoFeature.INFO_FEATURE__PREV_ACTION_TGT].set(0)

    return actor_input.replace(
        env=env.replace(
            private_team=jnp.zeros_like(env.private_team),
            my_moveset=my_moveset,
            info=info,
            action_mask=jnp.ones_like(env.action_mask),
        )
    )


def make_potential_apply(
    generation: int,
) -> Callable[[Params, PlayerActorInput], jax.Array]:
    """Builds the frozen-critic potential: (params, (T, B, ...) actor input)
    -> Φ in [-1, 1] with shape (T, B), float32, stop-gradient."""
    model = Porygon2OfflineCritic(get_player_model_config(generation, train=False))
    apply_fn = jax.vmap(model.apply, in_axes=(None, 1), out_axes=1)

    def potential(params: Params, actor_input: PlayerActorInput) -> jax.Array:
        value_head = apply_fn(params, mask_to_public_view(actor_input))
        return jax.lax.stop_gradient(value_head.expectation.astype(jnp.float32))

    return potential
