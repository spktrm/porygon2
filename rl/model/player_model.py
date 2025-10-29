from dotenv import load_dotenv

load_dotenv()

from pprint import pprint

import cloudpickle as pickle
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
from rl.model.heads import PolicyLogitHead, PolicyQKHead, ValueLogitHead
from rl.model.modules import SumEmbeddings
from rl.model.utils import get_most_recent_file, get_num_params


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.action_type_head = PolicyLogitHead(self.cfg.action_type_head)
        self.move_head = PolicyQKHead(self.cfg.move_head)
        self.switch_head = PolicyQKHead(self.cfg.switch_head)
        self.wildcard_merge = SumEmbeddings(
            output_size=self.cfg.entity_size, dtype=self.cfg.dtype
        )
        self.wildcard_head = PolicyLogitHead(self.cfg.wildcard_head)
        self.value_head = ValueLogitHead(self.cfg.value_head)

    def get_head_outputs(
        self,
        state_query: jax.Array,
        contextual_moves: jax.Array,
        contextual_switches: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
    ):
        action_type_head = self.action_type_head(
            state_query,
            env_step.action_type_mask,
            actor_output.action_type_head,
        )
        move_head = self.move_head(
            state_query,
            contextual_moves,
            env_step.move_mask,
            actor_output.move_head,
        )
        selected_move_embedding = jnp.take(
            contextual_moves, move_head.action_index, axis=0
        )
        wildcard_merge = self.wildcard_merge(state_query, selected_move_embedding)
        wildcard_head = self.wildcard_head(
            wildcard_merge,
            env_step.wildcard_mask,
            actor_output.wildcard_head,
        )
        switch_head = self.switch_head(
            state_query,
            contextual_switches,
            env_step.switch_mask,
            actor_output.switch_head,
        )
        value = self.value_head(state_query)

        return PlayerActorOutput(
            action_type_head=action_type_head,
            move_head=move_head,
            switch_head=switch_head,
            wildcard_head=wildcard_head,
            v=value,
        )

    def __call__(self, actor_input: PlayerActorInput, actor_output: PlayerActorOutput):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        state_query, contextual_moves, contextual_switches = self.encoder(
            actor_input.env, actor_input.history
        )

        return jax.vmap(self.get_head_outputs)(
            state_query,
            contextual_moves,
            contextual_switches,
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

    latest_ckpt = get_most_recent_file(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["player_state"]["params"]
    else:
        params = learner_network.init(key, ex_actor_input, ex_actor_output)

    actor_output = actor_network.apply(
        params, ex_actor_input, PlayerActorOutput(), rngs={"sampling": key}
    )
    pprint(actor_output)

    learner_output = learner_network.apply(params, ex_actor_input, actor_output)
    pprint(learner_output)

    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
