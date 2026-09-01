"""Representation-health probes — OFFLINE only.

Dormant-unit fraction and srank@0.99 over the trunk's embedding streams.
These measure capacity loss directly (dead units, rank collapse), and are
what caught the 1e-4 learning-rate collapse: action-embedding srank fell to
0.27 by 13k steps while actor-KL sat quietly at 0.002, so KL headroom was
never evidence the LR could rise (CLAUDE.md 5).

Deliberately NOT wired into the training loop. Everything here is a pure
function of (params, batch), so it can be run against any saved checkpoint
after the fact — which is where it belongs: an extra encoder forward plus an
eigendecomposition per probe is real GPU time spent on something no training
decision reads. See tests/test_checkpoint_collapse.py for the offline
harness, and scripts/attn_probe.py for the same pattern on attention.
"""

import jax
import jax.numpy as jnp

from rl.environment.interfaces import PlayerActorInput


def embedding_stats(emb: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Representation-health stats over one batch of trunk embeddings.

    Returns the dormant-unit fraction (ReDo criterion: units whose mean
    |activation| over valid steps is <= 0.025x the layer mean) and the
    srank@0.99 fraction (smallest number of singular values holding 99% of
    the spectrum mass, over the feature dim). emb is (T, B, ..., d), valid
    is (T, B); padded rows are zeroed, which leaves the Gram spectrum
    unchanged versus dropping them.
    """
    d = emb.shape[-1]
    lead = valid.reshape(valid.shape + (1,) * (emb.ndim - valid.ndim - 1))
    mask = jnp.broadcast_to(lead, emb.shape[:-1]).reshape(-1).astype(jnp.float32)
    flat = emb.reshape(-1, d).astype(jnp.float32) * mask[:, None]
    denom = mask.sum() + 1e-8

    unit_score = jnp.abs(flat).sum(axis=0) / denom
    dormant_frac = (unit_score <= 0.025 * unit_score.mean()).mean()

    gram = flat.T @ flat / denom
    singular_values = jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram), 0.0))
    singular_values = jnp.sort(singular_values)[::-1]
    srank = (jnp.cumsum(singular_values) < 0.99 * singular_values.sum()).sum() + 1
    return dormant_frac, srank / d


def make_capacity_probe(network):
    """Jitted encoder-only forward returning dormant-frac and srank@0.99 for
    both trunk embedding streams, keyed capacity_{action,value}_emb_*."""

    def encoder_only(module, actor_input: PlayerActorInput):
        return module.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

    encode = jax.vmap(
        lambda params, actor_input: network.apply(
            params, actor_input, method=encoder_only
        ),
        in_axes=(None, 1),
        out_axes=1,
    )

    def probe(params, batch) -> dict[str, jax.Array]:
        actor_input = PlayerActorInput(
            env=batch.player_transitions.env_output,
            packed_history=batch.player_packed_history,
            history=batch.player_history,
        )
        # The flat-trunk encoder returns ONE (T, B, 61, D) sequence
        # (2026-08-29); the "action" stream is the contiguous
        # private|move|target row block the readout consumes, the "value"
        # stream is the CLS row the critic reads.
        from rl.model.constants import CLS_ROW, PRIVATE_ROWS, TARGET_ROWS

        sequence = encode(params, actor_input)
        action_emb = sequence[:, :, PRIVATE_ROWS.start : TARGET_ROWS.stop]
        value_emb = sequence[:, :, CLS_ROW]
        dones = batch.player_transitions.env_output.done
        valid = (jnp.cumsum(dones, axis=0) - dones) == 0
        logs = {}
        for name, emb in (("action", action_emb), ("value", value_emb)):
            dormant_frac, srank_frac = embedding_stats(emb, valid)
            logs[f"capacity_{name}_emb_dormant_frac"] = dormant_frac
            logs[f"capacity_{name}_emb_srank_frac"] = srank_frac
        return logs

    return jax.jit(probe)
