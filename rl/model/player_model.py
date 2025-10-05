from dotenv import load_dotenv

load_dotenv()

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
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import MoveHead, PolicyHead, PolicyQKHead, ScalarHead
from rl.model.utils import get_most_recent_file, get_num_params


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)

        self.action_type_head = PolicyHead(self.cfg.action_type_head)
        self.move_head = MoveHead(self.cfg.move_head)
        self.switch_head = PolicyQKHead(self.cfg.switch_head)

        self.value_head1 = ScalarHead(self.cfg.value_head)
        self.value_head2 = ScalarHead(self.cfg.value_head)

    def get_head_outputs(
        self,
        entity_embeddings: jax.Array,
        action_embeddings: jax.Array,
        entity_mask: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
    ):

        move_embeddings = action_embeddings[:4]
        switch_embeddings = action_embeddings[4:]

        # Apply the value head
        move_switch_mask = jnp.concatenate(
            (
                env_step.action_type_mask[0] * env_step.move_mask,
                env_step.action_type_mask[1:].any(axis=-1) * env_step.switch_mask,
            ),
            axis=-1,
        )
        value = self.value_head1(entity_embeddings, entity_mask) + self.value_head2(
            action_embeddings, move_switch_mask
        )

        # Get the moves and wild cards
        move_head, wildcard_head = self.move_head(
            move_embeddings,
            entity_embeddings,
            env_step.move_mask,
            entity_mask,
            env_step.wildcard_mask,
            actor_output.move_head,
            actor_output.wildcard_head,
        )

        # Return the model output
        return PlayerActorOutput(
            action_type_head=self.action_type_head(
                entity_embeddings,
                entity_mask,
                env_step.action_type_mask,
                actor_output.action_type_head,
            ),
            move_head=move_head,
            wildcard_head=wildcard_head,
            switch_head=self.switch_head(
                switch_embeddings,
                entity_embeddings,
                env_step.switch_mask,
                entity_mask,
                actor_output.switch_head,
            ),
            v=value,
        )

    def __call__(
        self,
        actor_input: PlayerActorInput,
        actor_output: PlayerActorOutput = PlayerActorOutput(),
    ):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        entity_embeddings, action_embeddings, entity_mask = self.encoder(
            actor_input.env, actor_input.history
        )

        return jax.vmap(self.get_head_outputs)(
            entity_embeddings,
            action_embeddings,
            entity_mask,
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

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["player_state"]["params"]
    else:
        key = jax.random.key(42)
        params = learner_network.init(key, ex_actor_input, ex_actor_output)

    actor_output = actor_network.apply(params, ex_actor_input, rngs={"sampling": key})
    learner_network.apply(params, ex_actor_input, actor_output)

    pprint(get_num_params(params))
    pprint(actor_output)


if __name__ == "__main__":
    main()
