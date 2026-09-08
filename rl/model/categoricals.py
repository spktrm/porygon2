"""The categorical latent primitives the transition model and the search
front-ends are built from: the unimix floor, the straight-through sample,
the masked log-softmax, the nucleus support set and the Gumbel-top-k draw
without replacement.

They live here rather than in `rl/model/transition.py` because eight
modules read them -- the model, both search front-ends, the learner and
three offline probes -- and none of those wants the 900-line model file
in its import graph.
"""

import jax
import jax.numpy as jnp

# The 1% unimix floor DreamerV3 puts under every categorical: the KL can
# never see a zero, and the straight-through sample keeps a gradient.
UNIMIX = 0.01
NEGATIVE_INFINITY_LOGIT = -1e9


def unimix_probs(logits: jax.Array) -> jax.Array:
    """f32 softmax over the last axis with the unimix floor."""
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return (1.0 - UNIMIX) * probs + UNIMIX / logits.shape[-1]


def draw_or_mode(log_probs: jax.Array, rng: jax.Array | None) -> jax.Array:
    """The index: a categorical DRAW from `log_probs` with a key, the
    argmax without one. Every single-index latent draw in the model goes
    through here, so init, the probes and the offline harness are the mode
    everywhere and only training samples."""
    if rng is None:
        return jnp.argmax(log_probs, axis=-1)
    return jax.random.categorical(rng, log_probs, axis=-1)


def straight_through_sample(probs: jax.Array, rng: jax.Array | None) -> jax.Array:
    """One-hot forward, the probabilities' gradient backward. With a key the
    forward is a categorical DRAW from `probs` (the unimix floor is what
    makes every class reachable), so every class the posterior gives mass
    to is decoded and trained through the decode; without one it is the
    argmax -- the mode decode, which under straight-through only ever
    decodes each group's leading class and lets the rest die (usage
    perplexity pinned ~2.6 of 16 for 400k steps)."""
    if rng is None:
        index = draw_or_mode(probs, None)
    else:
        index = draw_or_mode(jnp.log(probs), rng)
    hard = jax.nn.one_hot(index, probs.shape[-1], dtype=probs.dtype)
    return hard + probs - jax.lax.stop_gradient(probs)


def masked_log_softmax(logits: jax.Array, allowed: jax.Array) -> jax.Array:
    """log-softmax over the allowed entries (f32); the others read the
    floor logit's log-probability, which a zero target multiplies to 0."""
    masked = jnp.where(allowed, logits.astype(jnp.float32), NEGATIVE_INFINITY_LOGIT)
    return jax.nn.log_softmax(masked, axis=-1)


def support_set(probs: jax.Array, mass_threshold: float) -> jax.Array:
    """The smallest descending-probability prefix whose cumulative mass
    reaches `mass_threshold`, as a boolean mask over the alphabet; ties
    resolve by index (stable sort)."""
    order = jnp.argsort(-probs, stable=True)
    sorted_probs = probs[order]
    exclusive = jnp.cumsum(sorted_probs) - sorted_probs
    keep_sorted = exclusive < mass_threshold
    return jnp.zeros_like(probs, dtype=bool).at[order].set(keep_sorted)


def gumbel_top_k(log_probs: jax.Array, num_draws: int, rng: jax.Array | None):
    """Kool et al. 2019: the top-k of log p + Gumbel noise is k sequential
    draws without replacement from p (the Plackett-Luce ordering);
    without a key it is the deterministic top-k. -inf entries are never
    drawn ahead of finite ones. Not `draw_or_mode`: this is an ORDER over
    k slots, not one index."""
    perturbed = log_probs
    if rng is not None:
        perturbed = log_probs + jax.random.gumbel(rng, log_probs.shape)
    return jax.lax.top_k(perturbed, num_draws)[1]
