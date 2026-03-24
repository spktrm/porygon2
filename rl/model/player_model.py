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
    RegressionValueHeadOutput,
)
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import (
    CategoricalValueLogitHead,
    HeadParams,
    PointerLogits,
    sample_categorical,
)
from rl.model.modules import MLP
from rl.model.utils import (
    get_most_recent_file,
    get_num_params,
    legal_log_policy,
    legal_policy,
)


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.action_head = PointerLogits(**self.cfg.action_head.qk_logits.to_dict())
        self.wm_head = MLP((self.cfg.entity_size // 2, 2 * self.cfg.entity_size))
        self.value_head = CategoricalValueLogitHead(self.cfg.value_head)

    def post_head(
        self,
        logits: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
    ):
        log_policy = legal_log_policy(logits, valid_mask)
        policy = legal_policy(logits, valid_mask)
        entropy = -jnp.sum(policy * log_policy, axis=-1)

        valid_sum = valid_mask.sum(axis=-1)
        log_factor = 1 / jnp.log(valid_sum).astype(entropy.dtype)
        entropy_scale = jnp.where(valid_sum <= 1, 1, log_factor)
        normalized_entropy = entropy * entropy_scale

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(valid_mask, logits, jnp.finfo(logits.dtype).min),
                self.make_rng("sampling"),
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
            normalized_entropy=normalized_entropy,
            src_index=src_index,
            tgt_index=tgt_index,
        )

    def _forward_value_head(self, x: jax.Array):
        return self.value_head(x)

    def _forward_wm_head(self, x: jax.Array, z: jax.Array):
        """
        x: state-action embedding
        z: next state embedding
        """
        p = self.wm_head(x)

        z = jax.lax.stop_gradient(z)

        # 2. L2-Normalize both to the unit hypersphere
        p = p / jnp.clip(jnp.linalg.norm(p, axis=-1, keepdims=True), a_min=1e-8)
        z = z / jnp.clip(jnp.linalg.norm(z, axis=-1, keepdims=True), a_min=1e-8)

        # 3. Negative Cosine Similarity
        # Since they are normalized, the dot product is the cosine similarity
        return RegressionValueHeadOutput(logits=jnp.sum(p * z, axis=-1))

    def get_head_outputs(
        self,
        state_embedding: jax.Array,
        action_embeddings: jax.Array,
        next_state_embedding: jax.Array,
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
        )

        src_embedding = jnp.take(
            action_embeddings, action_head.src_index, axis=0, mode="clip"
        )
        tgt_embedding = jnp.take(
            action_embeddings, action_head.tgt_index, axis=0, mode="clip"
        )
        state_action_embedding = jnp.concatenate(
            (state_embedding, src_embedding, tgt_embedding), axis=-1
        )
        wm_head = self._forward_wm_head(state_action_embedding, next_state_embedding)

        value_head = self._forward_value_head(state_embedding)

        return PlayerActorOutput(
            action_head=action_head, value_head=value_head, wm_head=wm_head
        )

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
        state_embedding, action_embeddings, next_state_embedding = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(
            state_embedding,
            action_embeddings,
            next_state_embedding,
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
        params = learner_network.init(
            key, ex_actor_input, ex_actor_output, HeadParams()
        )

    actor_output = actor_network.apply(
        params,
        ex_actor_input,
        PlayerActorOutput(),
        HeadParams(temp=0.8),
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
