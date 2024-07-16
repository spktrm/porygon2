import chex

import jax
import jax.numpy as jnp
import flax.linen as nn

from ml_collections import ConfigDict

from ml.arch.modules import Logits, PointerLogits, Resnet


def _legal_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
    """A soft-max policy that respects legal_actions."""
    chex.assert_equal_shape((logits, legal_actions))
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(axis=-1, keepdims=True)
    logits = jnp.where(legal_actions, logits, l_min)
    logits -= logits.max(axis=-1, keepdims=True)
    logits *= legal_actions
    exp_logits = jnp.where(
        legal_actions, jnp.exp(logits), 0
    )  # Illegal actions become 0.
    exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / exp_logits_sum


def legal_log_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
    """Return the log of the policy on legal action, 0 on illegal action."""
    chex.assert_equal_shape((logits, legal_actions))
    # logits_masked has illegal actions set to -inf.
    logits_masked = logits + jnp.log(legal_actions)
    max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
    logits_masked = logits_masked - max_legal_logit
    # exp_logits_masked is 0 for illegal actions.
    exp_logits_masked = jnp.exp(logits_masked)

    baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True))
    # Subtract baseline from logits. We do not simply return
    #     logits_masked - baseline
    # because that has -inf for illegal actions, or
    #     legal_actions * (logits_masked - baseline)
    # because that leads to 0 * -inf == nan for illegal actions.
    log_policy = jnp.multiply(legal_actions, (logits - max_legal_logit - baseline))
    return log_policy


def _prenorm_softmax(
    logits: chex.Array, mask: chex.Array, axis: int = -1, eps: float = 1e-5
):
    mask = mask + (mask.sum(axis=axis) == 0)
    mean = jnp.mean(logits, where=mask, axis=axis, keepdims=True)
    variance = jnp.var(logits, where=mask, axis=axis, keepdims=True)
    eps = jax.lax.convert_element_type(eps, variance.dtype)
    inv = jax.lax.rsqrt(variance + eps)
    return inv * (logits - mean)


class PolicyHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.query = Resnet(**self.cfg.query.to_dict())
        self.select_logits = PointerLogits(**self.cfg.pointer_logits.to_dict())
        self.action_logits = PointerLogits(**self.cfg.pointer_logits.to_dict())

    def __call__(
        self,
        state_embedding: chex.Array,
        select_embeddings: chex.Array,
        action_embeddings: chex.Array,
        legal: chex.Array,
    ):
        query = self.query(state_embedding)

        action_logits = self.action_logits(query, action_embeddings)
        select_logits = self.select_logits(query, select_embeddings)

        logits = jnp.concatenate((action_logits, select_logits))
        # denom = jnp.array(self.cfg.key_size, dtype=jnp.float32)
        # logits = logits * jax.lax.rsqrt(denom)

        policy = _legal_policy(logits, legal)
        log_policy = legal_log_policy(logits, legal)

        return (logits, policy, log_policy)


class ValueHead(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.resnet = Resnet(**self.cfg.resnet.to_dict())
        self.logits = Logits(**self.cfg.logits.to_dict())

    def __call__(self, x: chex.Array):
        x = self.resnet(x)
        return self.logits(x)
