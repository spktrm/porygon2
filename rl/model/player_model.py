import functools
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
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
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
        self.winloss_head = CategoricalValueLogitHead(self.cfg.winloss_head)

    def _forward_action_head(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        head_logits = self.action_head(action_embeddings, action_embeddings)
        logits = head_logits.squeeze(-1).reshape(-1) / temp

        flat_valid_mask = valid_mask.reshape(-1)
        valid_sum = jnp.maximum(flat_valid_mask.sum(axis=-1), 1)

        log_policy = legal_log_policy(logits, flat_valid_mask)
        policy = legal_policy(logits, flat_valid_mask)
        entropy = -jnp.sum(policy * log_policy, axis=-1)

        # Cross-entropy with uniform distribution over valid actions (for KL)
        uniform_policy = flat_valid_mask / valid_sum
        safe_prior = jnp.where(flat_valid_mask, uniform_policy, 1e-9)
        uniform_log_policy = jnp.where(flat_valid_mask, jnp.log(safe_prior), 0.0)
        cross_entropy = -jnp.sum(policy * uniform_log_policy, axis=-1)

        # KL to uniform "prior"
        magnet_kl = cross_entropy - entropy

        log_factor = 1 / jnp.log(valid_sum).astype(entropy.dtype)
        entropy_scale = jnp.where(valid_sum <= 1, 1, log_factor)
        normalized_entropy = entropy * entropy_scale

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(flat_valid_mask, logits, -1e9), self.make_rng("sampling")
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)

        mask_width = valid_mask.shape[-1]
        src_index = action_index // mask_width
        tgt_index = action_index % mask_width

        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            src_index=src_index,
            tgt_index=tgt_index,
            kl_prior=magnet_kl,
        )

    def _forward_value_head(self, state_embedding: jax.Array):
        return self.winloss_head(state_embedding)

    def get_head_outputs(
        self,
        value_embedding: jax.Array,
        action_embeddings: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):

        action_head = self._forward_action_head(
            action_embeddings,
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
        )
        value_head = self._forward_value_head(value_embedding)

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
        value_embedding, action_embeddings = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(
            value_embedding,
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
