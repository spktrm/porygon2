import functools
from pprint import pprint

import cloudpickle as pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import NUM_ACTION_FEATURES
from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.utils import get_ex_player_step
from rl.model.builder_model import RegressionValueLogitHead
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import HeadParams, PointerLogits, sample_categorical
from rl.model.utils import get_num_params, legal_log_policy, legal_policy


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.action_head = PointerLogits(**self.cfg.action_head.qk_logits.to_dict())
        self.value_head = RegressionValueLogitHead(self.cfg.value_head)

    def post_head(
        self,
        logits: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        min_p: float,
    ):
        log_policy = legal_log_policy(logits, valid_mask)
        policy = legal_policy(logits, valid_mask)
        entropy = -jnp.sum(policy * log_policy, axis=-1)

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(valid_mask, log_policy, jnp.finfo(log_policy.dtype).min),
                self.make_rng("sampling"),
                min_p=min_p,
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)

        src_index = (jnp.floor(action_index / NUM_ACTION_FEATURES)).astype(
            action_index.dtype
        )
        tgt_index = (action_index - src_index * NUM_ACTION_FEATURES).astype(
            action_index.dtype
        )

        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            entropy=entropy,
            src_index=src_index,
            tgt_index=tgt_index,
        )

    def get_head_outputs(
        self,
        state_embedding: jax.Array,
        action_embeddings: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):

        action_logits = (
            self.action_head(action_embeddings, action_embeddings) / head_params.temp
        )
        action_head = self.post_head(
            action_logits.reshape(-1),
            env_step.action_mask.reshape(-1),
            actor_output.action_head,
            train=self.cfg.train,
            min_p=head_params.min_p,
        )

        value_head = self.value_head(state_embedding)

        return PlayerActorOutput(action_head=action_head, value_head=value_head)

    def __call__(
        self,
        actor_input: PlayerActorInput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        latent_state_embeddings, action_embeddings = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(
            latent_state_embeddings,
            action_embeddings,
            actor_input.env,
            actor_output,
        )


def get_player_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_player_model_config()
    return Porygon2PlayerModel(config)


def main(generation: int = 9):
    actor_network = get_player_model(get_player_model_config(generation, train=False))
    learner_network = get_player_model(get_player_model_config(generation, train=True))

    ex_actor_input, ex_actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    )
    key = jax.random.key(42)

    latest_ckpt = None  # get_most_recent_file(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["player_state"]["params"]
    else:
        params = learner_network.init(
            key, ex_actor_input, ex_actor_output, HeadParams()
        )

    actor_output = actor_network.apply(
        params,
        ex_actor_input,
        PlayerActorOutput(),
        HeadParams(temp=0.8, min_p=0.1),
        rngs={"sampling": key},
    )
    pprint(actor_output)

    learner_output = learner_network.apply(
        params, ex_actor_input, actor_output, HeadParams()
    )
    pprint(learner_output)

    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
