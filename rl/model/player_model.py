import functools
from pprint import pprint

import cloudpickle as pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import NUM_ACTION_FEATURES
from rl.environment.interfaces import (
    CategoricalValueHeadOutput,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import HeadParams, PointerLogits, sample_categorical
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
        self.value_head = PointerLogits(**self.cfg.value_head.qk_logits.to_dict())

    def _forward_action_head(
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

    def _forward_value_head(
        self, action_logits: jax.Array, value_logits: jax.Array, valid_mask: jax.Array
    ):

        flat_valid_mask = valid_mask.reshape(-1)
        flat_action_logits = action_logits.reshape(-1)

        action_policy = legal_policy(flat_action_logits, flat_valid_mask)
        flat_value_logits = value_logits.reshape(value_logits.shape[0], -1)
        aggregate_value_logits = (
            flat_value_logits * jax.lax.stop_gradient(action_policy[None])
        ).sum(axis=-1)

        log_probs = jax.nn.log_softmax(aggregate_value_logits)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        expectation = probs @ self.cfg.value_head.category_values.astype(probs.dtype)

        return CategoricalValueHeadOutput(
            logits=aggregate_value_logits,
            log_probs=log_probs,
            entropy=entropy,
            expectation=expectation,
        )

    def get_head_outputs(
        self,
        action_embeddings: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):

        action_logits = (
            self.action_head(action_embeddings, action_embeddings) / head_params.temp
        )
        value_logits = self.value_head(action_embeddings, action_embeddings)

        action_head = self._forward_action_head(
            action_logits.reshape(-1),
            env_step.action_mask.reshape(-1),
            actor_output.action_head,
            train=self.cfg.train,
        )

        value_head = self._forward_value_head(
            action_logits, value_logits, env_step.action_mask
        )

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
        action_embeddings = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(action_embeddings, actor_input.env, actor_output)


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
