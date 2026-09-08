"""The transition model's LEARNER-ONLY objectives: the exact
action-discrimination read on the action encoder, and the teacher-forced
targets and loss for the candidate generator.

Nothing in `rl/model/transition.py` calls these -- they are scored in
`rl/online/training/train_step.py` against the model's outputs, so they
are objectives, not architecture, and they live beside neither the model
nor the update but on their own.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rl.model.categoricals import (
    NEGATIVE_INFINITY_LOGIT,
    masked_log_softmax,
    unimix_probs,
)


class DecodeRead(NamedTuple):
    """The exact action-discrimination objective (plan §4) at one state:
    `loss` = H_w(A | U, h) under uniform reference weights over the legal
    cells, `mutual_information` = log(n) - loss, `accuracy` the expected
    top-1 decode of the taken cell over u ~ q(. | h, a), `num_valid` n."""

    loss: jax.Array
    mutual_information: jax.Array
    accuracy: jax.Array
    num_valid: jax.Array


def exact_decode_loss(encoder_logits: jax.Array, cell_valid: jax.Array) -> DecodeRead:
    """L(h) = -sum_a w(a) sum_u q_a(u) log D_w(a | u), D_w(a | u) =
    w(a) q_a(u) / sum_a' w(a') q_a'(u), w uniform over the valid cells,
    q the unimix'd encoder distribution -- summed EXACTLY over every code
    (no sampled straight-through term: that drops the derivative of the
    sampling distribution). Stable log-space throughout; an empty legal
    set reads 0 on every field."""
    num_valid = cell_valid.sum()
    log_q = jnp.log(unimix_probs(encoder_logits))
    log_weight = jnp.where(
        cell_valid,
        -jnp.log(jnp.maximum(num_valid, 1).astype(jnp.float32)),
        NEGATIVE_INFINITY_LOGIT,
    )
    joint = log_weight[:, None] + log_q
    log_mixture = jax.nn.logsumexp(joint, axis=0)
    log_decode = joint - log_mixture[None]
    weight = jnp.exp(log_weight)
    probs = jnp.exp(log_q)
    loss = -jnp.sum(weight[:, None] * probs * log_decode, where=cell_valid[:, None])
    decoded = jnp.argmax(jnp.where(cell_valid[:, None], joint, -jnp.inf), axis=0)
    hit = decoded[None, :] == jnp.arange(cell_valid.shape[0])[:, None]
    accuracy = jnp.sum(weight[:, None] * probs * hit, where=cell_valid[:, None])
    log_num = jnp.log(jnp.maximum(num_valid, 1).astype(jnp.float32))
    nonempty = num_valid > 0
    return DecodeRead(
        loss=jnp.where(nonempty, loss, 0.0),
        mutual_information=jnp.where(nonempty, log_num - loss, 0.0),
        accuracy=jnp.where(nonempty, accuracy, 0.0),
        num_valid=num_valid,
    )


class CandidateTargets(NamedTuple):
    allowed: jax.Array
    targets: jax.Array
    occupied: jax.Array


def candidate_targets(
    p_target: jax.Array, teacher_codes: jax.Array, support_mask: jax.Array
) -> CandidateTargets:
    """Teacher-forced targets for the J candidate slots: slot 0 is trained
    on the FULL p_target so the first conditional stays a calibrated
    prior; slot j > 0 on p_target renormalised over the support minus the
    prefix drawn before it (exact Plackett-Luce conditionals). A slot is
    occupied when its teacher code lies in the support (slot 0 always)."""
    num_draws, num_codes = teacher_codes.shape[0], p_target.shape[0]
    one_hot = jax.nn.one_hot(teacher_codes, num_codes, dtype=jnp.float32)
    drawn_before = (jnp.cumsum(one_hot, axis=0) - one_hot) > 0.5
    first = jnp.arange(num_draws)[:, None] == 0
    allowed = first | (support_mask[None] & jnp.logical_not(drawn_before))
    targets = p_target[None] * allowed
    targets = targets / jnp.maximum(targets.sum(-1, keepdims=True), 1e-8)
    occupied = support_mask[teacher_codes].at[0].set(True)
    return CandidateTargets(allowed=allowed, targets=targets, occupied=occupied)


class CandidateLoss(NamedTuple):
    loss: jax.Array
    cross_entropy: jax.Array
    target_entropy: jax.Array


def candidate_loss(logits: jax.Array, targets: CandidateTargets) -> CandidateLoss:
    """Per node: the first slot's CE weighted once and the mean of the
    later occupied slots' CEs weighted once, averaged; one occupied slot
    reads the first CE alone. `cross_entropy - target_entropy` per slot
    is the KL, the read (CE < log C says nothing)."""
    log_probs = masked_log_softmax(logits, targets.allowed)
    cross_entropy = -(targets.targets * log_probs).sum(-1)
    safe_log = jnp.log(jnp.maximum(targets.targets, 1e-8))
    target_entropy = -(targets.targets * safe_log).sum(-1)
    later = targets.occupied & (jnp.arange(logits.shape[0]) > 0)
    later_count = later.sum()
    later_mean = jnp.sum(cross_entropy, where=later) / jnp.maximum(later_count, 1)
    loss = jnp.where(
        later_count > 0, 0.5 * (cross_entropy[0] + later_mean), cross_entropy[0]
    )
    return CandidateLoss(
        loss=loss, cross_entropy=cross_entropy, target_entropy=target_entropy
    )
