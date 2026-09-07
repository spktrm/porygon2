"""Detached per-chunk replay feedback; no changes to loss reductions."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.online.training.loss import approx_forward_kl


def chunk_policy_mismatch(policy_ratio, log_policy_ratio, policy_mask):
    """Return taken-action k3 KL sums and eligible counts over time.

    The caller supplies the existing policy mask, excluding padding,
    bootstrap-only rows and non-policy decisions. This is a sampled mismatch
    estimate, not exact full-support KL or a learning-utility score.
    """
    terms = approx_forward_kl(
        policy_ratio=policy_ratio.astype(jnp.float32),
        log_policy_ratio=log_policy_ratio.astype(jnp.float32),
    )
    totals = jnp.where(policy_mask, terms, 0.0).sum(axis=0)
    counts = policy_mask.astype(jnp.float32).sum(axis=0)
    return jax.lax.stop_gradient(totals), jax.lax.stop_gradient(counts)


def consume_replay_feedback(store, host_logs):
    """Consume private feedback before scalar logs are sent to W&B."""
    feedback = host_logs.pop("_player_replay_feedback", None)
    if feedback is None:
        return
    if host_logs.get("player_update_skipped", 0) == 0:
        store.apply_feedback(*feedback)
        kl_sums = np.asarray(feedback[3]).reshape(-1)
        counts = np.asarray(feedback[4]).reshape(-1)
        valid = (
            np.isfinite(kl_sums) & (kl_sums >= 0) & np.isfinite(counts) & (counts > 0)
        )
        if valid.any():
            scores = kl_sums[valid] / counts[valid]
            host_logs["player_replay_chunk_kl_mean"] = float(scores.mean())
            host_logs["player_replay_chunk_kl_max"] = float(scores.max())
            host_logs["player_replay_chunk_policy_rows_mean"] = float(
                counts[valid].mean()
            )
            host_logs["player_replay_chunk_policy_rows_min"] = float(
                counts[valid].min()
            )
            host_logs["player_replay_chunk_above_threshold_frac"] = float(
                (scores > store.kl_threshold).mean()
            )
    host_logs.update(store.feedback_logs())
