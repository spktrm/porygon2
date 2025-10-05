import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import HeadOutput
from rl.environment.protos.features_pb2 import ActionMaskFeature
from rl.model.modules import (
    OutputLayer,
    TransformerDecoder,
    TransformerEncoder,
    create_attention_mask,
)
from rl.model.utils import legal_log_policy, legal_policy


def sample_categorical(
    logits: jax.Array,
    log_policy: jax.Array,
    mask: jax.Array,
    rng: jax.random.PRNGKey,
    min_p: float = 0,
):
    masked_logits = jnp.where(mask, logits, -jnp.inf)
    if 0.0 < min_p < 1.0:
        masked_log_policy = jnp.where(mask, log_policy, -jnp.inf)
        max_logp = masked_log_policy.max(keepdims=True, axis=-1)
        keep = masked_log_policy >= (max_logp + math.log(min_p))
        masked_logits = jnp.where(keep & mask, masked_logits, -jnp.inf)
    return jax.random.categorical(rng, masked_logits.astype(jnp.float32))


class MoveHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())

        self.move_head = OutputLayer(1, dtype=self.cfg.dtype)
        self.wildcard_head = OutputLayer(
            ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TERA
            - ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL
            + 1,
            dtype=self.cfg.dtype,
        )

    def _forward_move_head(
        self,
        query_embeddings: jax.Array,
        query_mask: jax.Array,
        head: HeadOutput,
    ):
        temp = self.cfg.get("temp", 1.0)
        logits = self.move_head(query_embeddings).reshape(-1)
        logits = logits / temp
        log_policy = legal_log_policy(logits, query_mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, query_mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            action_index = sample_categorical(
                logits,
                log_policy,
                query_mask,
                self.make_rng("sampling"),
                self.cfg.get("min_p", 0.0),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def _forward_wildcard_head(
        self,
        query_embeddings: jax.Array,
        move_action_index: jax.Array,
        mask: jax.Array,
        head: HeadOutput,
    ):
        temp = self.cfg.get("temp", 1.0)
        logits = self.wildcard_head(
            jnp.take(query_embeddings, move_action_index, axis=0)
        )
        logits = logits / temp
        log_policy = legal_log_policy(logits, mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            action_index = sample_categorical(
                logits,
                log_policy,
                mask,
                self.make_rng("sampling"),
                self.cfg.get("min_p", 0.0),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)

    def __call__(
        self,
        query_embeddings: jax.Array,
        key_value_embeddings: jax.Array,
        query_mask: jax.Array,
        key_value_mask: jax.Array,
        wildcard_mask: jax.Array,
        move_head: HeadOutput,
        wildcard_head: HeadOutput,
    ):
        query_embeddings = self.encoder(
            query_embeddings,
            create_attention_mask(query_mask),
        )

        query_embeddings = self.decoder(
            query_embeddings,
            key_value_embeddings,
            create_attention_mask(query_mask, key_value_mask),
        )

        move_head = self._forward_move_head(query_embeddings, query_mask, move_head)

        wildcard_head = self._forward_wildcard_head(
            query_embeddings, move_head.action_index, wildcard_mask, wildcard_head
        )

        return move_head, wildcard_head


class PolicyQKHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_mlp = OutputLayer(1, dtype=self.cfg.dtype)

    def __call__(
        self,
        query_embeddings: jax.Array,
        key_value_embeddings: jax.Array,
        query_mask: jax.Array,
        key_value_mask: jax.Array,
        head: HeadOutput,
    ):
        query_embeddings = self.encoder(
            query_embeddings,
            create_attention_mask(query_mask),
        )
        query_embeddings = self.decoder(
            query_embeddings,
            key_value_embeddings,
            create_attention_mask(query_mask, key_value_mask),
        )

        temp = self.cfg.get("temp", 1.0)
        logits = self.final_mlp(query_embeddings)
        logits = logits.reshape(-1) / temp

        log_policy = legal_log_policy(logits, query_mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, query_mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            action_index = sample_categorical(
                logits,
                log_policy,
                query_mask,
                self.make_rng("sampling"),
                self.cfg.get("min_p", 0.0),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_mlp = OutputLayer(self.cfg.output_features, dtype=self.cfg.dtype)

    def __call__(
        self,
        entity_embeddings: jax.Array,
        entity_mask: jax.Array,
        policy_mask: jax.Array,
        head: HeadOutput,
    ):
        entity_embeddings = self.encoder(
            entity_embeddings,
            create_attention_mask(entity_mask),
        )
        query = entity_embeddings.mean(
            where=entity_mask[..., None], axis=0, keepdims=True
        )
        query_mask = entity_mask.any(keepdims=True)
        pooled = self.decoder(
            query, entity_embeddings, create_attention_mask(query_mask, entity_mask)
        )

        temp = self.cfg.get("temp", 1.0)
        logits = self.final_mlp(pooled)
        logits = logits.reshape(-1) / temp

        log_policy = legal_log_policy(logits, policy_mask)

        entropy = ()
        train = self.cfg.get("train", False)
        if train:
            action_index = head.action_index
            policy = legal_policy(logits, policy_mask)
            entropy = -jnp.sum(policy * log_policy, axis=-1)
        else:
            action_index = sample_categorical(
                logits,
                log_policy,
                policy_mask,
                self.make_rng("sampling"),
                self.cfg.get("min_p", 0.0),
            )

        log_prob = jnp.take(log_policy, action_index, axis=-1)
        return HeadOutput(action_index=action_index, log_prob=log_prob, entropy=entropy)


class ScalarHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = TransformerEncoder(**self.cfg.transformer.to_dict())
        self.decoder = TransformerDecoder(**self.cfg.transformer.to_dict())
        self.final_mlp = OutputLayer(self.cfg.output_features, dtype=self.cfg.dtype)

    def __call__(self, entity_embeddings: jax.Array, entity_mask: jax.Array):
        entity_embeddings = self.encoder(
            entity_embeddings,
            create_attention_mask(entity_mask),
        )
        query = entity_embeddings.mean(
            where=entity_mask[..., None], axis=0, keepdims=True
        )
        query_mask = entity_mask.any(keepdims=True)
        pooled = self.decoder(
            query, entity_embeddings, create_attention_mask(query_mask, entity_mask)
        )
        value = self.final_mlp(pooled)

        return value.squeeze()
