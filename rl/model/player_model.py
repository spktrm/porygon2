import functools
import pickle
from pprint import pprint

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
)
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_model_config
from rl.model.encoder import Encoder
from rl.model.heads import MoveHead, PolicyHead, ScalarHead
from rl.model.utils import get_num_params
from rl.utils import init_jax_jit_cache


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)

        self.action_type_head = PolicyHead(self.cfg.action_type_head)
        self.move_head = MoveHead(self.cfg.move_head)
        self.switch_head = PolicyHead(self.cfg.switch_head)

        self.value_head = ScalarHead(self.cfg.value_head)

    def get_head_outputs(
        self,
        entity_embeddings: jax.Array,
        action_embeddings: jax.Array,
        entity_mask: jax.Array,
        env_step: PlayerEnvOutput,
        temp: float = 1.0,
    ):

        move_embeddings = action_embeddings[:4]
        switch_embeddings = action_embeddings[4:]

        average_move_embedding = (
            env_step.move_mask @ move_embeddings
        ) / env_step.move_mask.sum(axis=-1, keepdims=True).clip(min=1)
        average_switch_embedding = (
            env_step.switch_mask @ switch_embeddings
        ) / env_step.switch_mask.sum(axis=-1, keepdims=True).clip(min=1)
        action_embeddings = jnp.stack(
            (average_move_embedding, average_switch_embedding, average_switch_embedding)
        )

        # Apply the value head
        value = jnp.tanh(self.value_head(entity_embeddings, entity_mask))

        # Get the moves and wild cards
        move_head, wildcard_head = self.move_head(
            move_embeddings,
            entity_embeddings,
            env_step.move_mask,
            entity_mask,
            env_step.wildcard_mask,
            temp,
        )

        # Return the model output
        return PlayerActorOutput(
            action_type_head=self.action_type_head(
                action_embeddings,
                entity_embeddings,
                env_step.action_type_mask,
                entity_mask,
                temp,
            ),
            move_head=move_head,
            wildcard_head=wildcard_head,
            switch_head=self.switch_head(
                switch_embeddings,
                entity_embeddings,
                env_step.switch_mask,
                entity_mask,
                temp,
            ),
            v=value,
        )

    def __call__(self, actor_input: PlayerActorInput, temp: float = 1.0):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        entity_embeddings, action_embeddings, entity_mask = self.encoder(
            actor_input.env, actor_input.history
        )

        return jax.vmap(functools.partial(self.get_head_outputs, temp=temp))(
            entity_embeddings,
            action_embeddings,
            entity_mask,
            actor_input.env,
        )


def get_player_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_model_config()
    return Porygon2PlayerModel(config)


def main():
    init_jax_jit_cache()
    network = get_player_model()
    ts = jax.device_put(jax.tree.map(lambda x: x[:, 0], get_ex_player_step()))

    latest_ckpt = None  # get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["player_state"]["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ts)

    network.apply(params, ts)
    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
