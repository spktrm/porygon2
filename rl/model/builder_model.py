import pickle
from pprint import pprint
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.actor.agent import Agent
from rl.environment.data import PACKED_SETS
from rl.environment.env import TeamBuilderEnvironment
from rl.environment.interfaces import (
    BuilderActorOutput,
    BuilderAgentOutput,
    BuilderEnvOutput,
    BuilderTransition,
    PolicyHeadOutput,
)
from rl.environment.utils import get_ex_builder_step
from rl.model.config import get_model_config
from rl.model.modules import (
    MLP,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import (
    BIAS_VALUE,
    Params,
    get_most_recent_file,
    get_num_params,
    legal_log_policy,
    legal_policy,
)
from rl.utils import init_jax_jit_cache


class Porygon2BuilderModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        entity_size = self.cfg.entity_size
        dtype = self.cfg.dtype

        sets_data = PACKED_SETS[f"gen{self.cfg.generation}ou"]
        num_sets = len(sets_data["sets"])
        embedding_init_fn = nn.initializers.normal()

        self.embeddings = nn.Embed(
            num_embeddings=num_sets,
            features=entity_size,
            dtype=dtype,
            embedding_init=embedding_init_fn,
        )

        transformer_config = self.cfg.action_type_head.transformer.to_dict()
        transformer_config["need_pos"] = True

        self.decoder = TransformerDecoder(**transformer_config)
        self.encoder = TransformerEncoder(**transformer_config)

        self.policy_head = MLP((entity_size, num_sets), dtype=dtype)
        self.value_head = MLP((entity_size, 1), dtype=dtype)

        self.mask_embedding = self.param(
            "mask_embedding", embedding_init_fn, (1, entity_size), dtype=dtype
        )

    def _forward_head(self, embedding: jax.Array, sample_mask: jax.Array):
        """
        Samples a token from the embeddings using the policy head and returns the token, log probability, and entropy.
        """
        logits = self.policy_head(embedding)
        masked_logits = jnp.where(sample_mask, logits, BIAS_VALUE)
        pi = legal_policy(logits, sample_mask)
        log_pi = legal_log_policy(logits, sample_mask)

        return PolicyHeadOutput(logits=masked_logits, policy=pi, log_policy=log_pi)

    def _forward(self, input: BuilderEnvOutput) -> BuilderAgentOutput:
        """Autoregressively generates a team and returns (tokens, log_pi)."""
        masked = input.tokens == -1
        not_masked_sum = (6 - masked.sum(axis=-1)).clip(0, 5)
        num_tokens = input.tokens.shape[-1]

        attn_mask = jnp.ones_like(input.tokens, dtype=jnp.bool)

        position_indices = jnp.arange(num_tokens, dtype=jnp.int32)

        embeddings = jnp.where(
            masked[..., None], self.mask_embedding, self.embeddings(input.tokens)
        )
        pred_embeddings = self.encoder(
            embeddings, create_attention_mask(attn_mask), position_indices
        )
        head_output = self._forward_head(
            jnp.take(pred_embeddings, not_masked_sum, axis=0), input.mask
        )
        pooled = self.decoder(
            pred_embeddings.mean(axis=0, keepdims=True),
            pred_embeddings,
            create_attention_mask(attn_mask.any(keepdims=True), attn_mask),
        )
        v = (2 / jnp.pi) * jnp.atan((jnp.pi / 2) * self.value_head(pooled)).squeeze()

        return BuilderActorOutput(head=head_output, v=v)

    def __call__(self, input: BuilderEnvOutput) -> BuilderAgentOutput:
        return jax.vmap(self._forward)(input)


def get_builder_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_model_config()
    return Porygon2BuilderModel(config)


def main():
    init_jax_jit_cache()

    generation = 9
    network = get_builder_model(get_model_config(generation))

    ex = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_builder_step(generation=generation))
    )
    key = jax.random.key(42)

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["builder_state"]["params"]
    else:
        params = network.init(key, ex)

    pprint(get_num_params(params))

    apply_fn: Callable[[Params, jax.Array, jax.Array | None], BuilderAgentOutput]

    apply_fn = network.apply

    agent = Agent(
        builder_apply_fn=jax.vmap(apply_fn, in_axes=(None, 1), out_axes=1),
        do_threshold=True,
    )

    while True:
        rng_key, key = jax.random.split(key)
        builder_subkeys = jax.random.split(rng_key, 7)

        build_traj = []
        builder_env = TeamBuilderEnvironment(generation=generation)

        builder_env_output = builder_env.reset()
        for subkey in builder_subkeys:
            builder_agent_output = agent.step_builder(
                subkey, params, builder_env_output
            )
            builder_transition = BuilderTransition(
                env_output=builder_env_output, agent_output=builder_agent_output
            )
            build_traj.append(builder_transition)
            if builder_env_output.done.item():
                break
            builder_env_output = builder_env.step(builder_agent_output.action.item())

        builder_trajectory: BuilderTransition = jax.tree.map(
            lambda *xs: np.stack(xs), *build_traj
        )

        tokens_buffer = np.asarray(builder_env_output.tokens, dtype=np.int16)
        print("tokens:", tokens_buffer)
        print("value:", builder_trajectory.agent_output.actor_output.v)
        print(
            "\n".join(
                PACKED_SETS[f"gen{generation}ou"]["sets"][t]
                for t in tokens_buffer.reshape(-1).tolist()
            )
        )


if __name__ == "__main__":
    main()
