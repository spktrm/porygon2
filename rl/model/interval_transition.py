"""Offline unilateral-interval experiment; never an opponent policy.

The matched arms share every parameter and operation except the successor rows
visible to the first posterior. ``combined`` sees the full successor; ``history``
sees history-row movement there and leaves other movement to the residual code.
Post-trunk history is contextual information, not labelled opponent decisions.
The conditional prior samples both codes ancestrally; no factorised-marginal
shortcut is valid for evaluation. The existing decoder and own-action alphabet
are reused. Production transition/search configuration is untouched.
"""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.model.categoricals import straight_through_sample, unimix_probs
from rl.model.constants import (
    MOVE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    SEQUENCE_GROUP_IDS,
    TARGET_ROWS,
    SequenceGroup,
)
from rl.model.heads import chosen_bank_rows
from rl.model.modules import MLP
from rl.model.transition import RowRead, TransitionModel


class DirectIntervalValue(nn.Module):
    """Offline copy-initialised value probe; no successor enters prediction."""

    cfg: ConfigDict
    categories: int
    dtype: jnp.dtype
    action_conditioned: bool
    action_representation: str = "latent"

    @nn.compact
    def __call__(self, rows, action, root_log_probs):
        read = RowRead(self.cfg.row_read_width, self.dtype, name="row_read")(rows)
        if self.action_representation == "latent":
            table = self.param(
                "action_table",
                nn.initializers.lecun_normal(),
                (self.cfg.action_classes, self.cfg.block.model_size),
            )
            embedding = action.astype(self.dtype) @ table.astype(self.dtype)
        elif self.action_representation == "rows":
            source, target = chosen_bank_rows(
                rows[PRIVATE_ROWS], rows[MOVE_ROWS], rows[TARGET_ROWS], action
            )
            embedding = nn.Dense(
                self.cfg.block.model_size, dtype=self.dtype, name="action_projection"
            )(jnp.concatenate((source, target)))
        else:
            raise ValueError("Action representation must be latent or rows")
        if not self.action_conditioned:
            embedding = jnp.zeros_like(embedding)
        read = jnp.broadcast_to(read, embedding.shape[:-1] + read.shape)
        features = jnp.concatenate((read, embedding), axis=-1)
        widths = tuple(self.cfg.prior.mlp.layer_sizes[:-1]) + (self.categories,)
        residual = MLP(
            widths, final_kernel_init=nn.initializers.zeros_init(), name="delta"
        )(features).astype(jnp.float32)
        residual = residual - residual.mean(axis=-1, keepdims=True)
        return jax.lax.stop_gradient(root_log_probs) + residual


class IntervalPrediction(NamedTuple):
    rows: jax.Array
    prior_logits: jax.Array
    posterior_logits: jax.Array
    code: jax.Array


def posterior_movement(rows, next_rows, evidence):
    movement = next_rows - rows
    if evidence == "history":
        groups = jnp.asarray(SEQUENCE_GROUP_IDS[POLICY_READABLE_ROWS])
        visible = (groups == SequenceGroup.HISTORY_ENTITY) | (
            groups == SequenceGroup.HISTORY_FIELD
        )
        movement = jnp.where(visible[:, None], movement, 0)
    elif evidence not in ("combined", "legacy"):
        raise ValueError("evidence must be legacy, combined or history")
    return movement


class IntervalTransition(nn.Module):
    cfg: ConfigDict
    dtype: jnp.dtype
    evidence: str

    def setup(self):
        if self.cfg.code_groups != 2:
            raise ValueError("Interval experiment requires two categorical code groups")
        if self.evidence not in ("legacy", "combined", "history"):
            raise ValueError("Unknown interval posterior evidence")
        self.dynamics = TransitionModel(self.cfg, self.dtype)
        if self.evidence != "legacy":
            widths = tuple(self.cfg.prior.mlp.layer_sizes[:-1]) + (
                self.cfg.code_classes,
            )
            self.behaviour_prior = MLP(widths)
            self.behaviour_posterior = MLP(widths)
            self.residual_prior = MLP(widths)
            self.residual_posterior = MLP(widths)

    def first_logits(self, features, movement=None):
        if movement is None:
            return self.behaviour_prior(features).astype(jnp.float32)
        return self.behaviour_posterior(
            jnp.concatenate((features, self.dynamics.row_read(movement)))
        ).astype(jnp.float32)

    def residual_logits(self, features, behaviour, movement=None):
        features = jnp.concatenate((features, behaviour.astype(self.dtype)))
        if movement is None:
            return self.residual_prior(features).astype(jnp.float32)
        return self.residual_posterior(
            jnp.concatenate((features, self.dynamics.row_read(movement)))
        ).astype(jnp.float32)

    def __call__(self, rows, action, next_rows, rng):
        if self.evidence == "legacy":
            prior = self.dynamics.prior(rows, action)
            posterior = self.dynamics.posterior(rows, action, next_rows)
            code = straight_through_sample(unimix_probs(posterior), rng)
        else:
            behaviour_key, residual_key = jax.random.split(rng)
            features = self.dynamics.prior_features(rows, action)
            movement = posterior_movement(rows, next_rows, self.evidence)
            behaviour_prior = self.first_logits(features)
            behaviour_posterior = self.first_logits(features, movement)
            behaviour = straight_through_sample(
                unimix_probs(behaviour_posterior), behaviour_key
            )
            # KL trains p(residual | posterior behaviour); deployment uses
            # a prior behaviour draw instead, in sample_prior below.
            residual_prior = self.residual_logits(
                features, jax.lax.stop_gradient(behaviour)
            )
            residual_posterior = self.residual_logits(
                features, behaviour, next_rows - rows
            )
            residual = straight_through_sample(
                unimix_probs(residual_posterior), residual_key
            )
            prior = jnp.stack((behaviour_prior, residual_prior))
            posterior = jnp.stack((behaviour_posterior, residual_posterior))
            code = jnp.stack((behaviour, residual))
        return IntervalPrediction(
            self.dynamics.imagine(rows, action, code), prior, posterior, code
        )

    def sample_prior(self, rows, action, rng):
        """Only current information; the successor is not an argument."""
        if self.evidence == "legacy":
            logits = self.dynamics.prior(rows, action)
            code = straight_through_sample(unimix_probs(logits), rng)
        else:
            behaviour_key, residual_key = jax.random.split(rng)
            features = self.dynamics.prior_features(rows, action)
            behaviour = straight_through_sample(
                unimix_probs(self.first_logits(features)), behaviour_key
            )
            residual = straight_through_sample(
                unimix_probs(self.residual_logits(features, behaviour)), residual_key
            )
            code = jnp.stack((behaviour, residual))
        return self.dynamics.imagine(rows, action, code)


def warm_start_decoder(initial_params, source_params):
    """Copy only equal-shape existing decoder leaves; return explicit provenance.

    New conditional networks retain their seeded initialisation. Never merge
    unrelated policy parameters or silently accept a changed decoder shape.
    """
    from flax import traverse_util
    from flax.core import unfreeze

    target = traverse_util.flatten_dict(unfreeze(initial_params))
    source = traverse_util.flatten_dict(unfreeze(source_params))
    copied = []
    for path, value in target.items():
        if path[0] != "dynamics":
            continue
        source_path = path[1:]
        if source_path not in source or source[source_path].shape != value.shape:
            raise ValueError(f"Missing or incompatible decoder parameter: {path}")
        target[path] = jnp.array(source[source_path], copy=True)
        copied.append("/".join(path))
    return traverse_util.unflatten_dict(target), copied
