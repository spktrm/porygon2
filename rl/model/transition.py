"""The latent transition model (2026-09-05): g(h_t, a, z) -> h_{t+1}.

`h_t` is the trunk's post-trunk POLICY-READABLE sequence (73 rows), exactly
what the heads read, so the imagined `h_{t+1}` keeps the sequence layout and
`V`, the action readout, this model itself and the heads below apply to it
unchanged. Between two of my requests the opponent decides, the engine
rolls and information is revealed, none of it observed as a choice
(memory: the opponent's action is NEVER a label); the chance code `z` --
`code_groups` categoricals of `code_classes` -- is the one latent that
covers all three, inferred through a bottleneck (LAPO/Genie's shape) from
the real next rows by the posterior and predicted from `h_t` and my
action by the prior. Every head here is trained on an OBSERVED label:

- grounding: per imagined row -> the CHANGE in the DYNAMICS_TARGET_ROWS'
  pre-trunk content, t -> t+1, in the next step's layout (the old delta
  head's label and its zero-init output: the head starts AT the copy
  predictor, gain 0, and its loss is on the delta's own scale -- a
  content target was tried first and started at loss ~18, the static
  tokens' reconstruction error against a delta-sized normaliser, with
  the head's gradient alone at 2x the global clip);
- the next action mask: the action readout's own form instantiated a
  second time (`mask_head`), so a rollout knows its next legal set;
- one cls head off the imagined CLS row: the next request kind (a
  force-switch node is MY decision again with no opponent move inside
  it) and done.

Init contract: `out_proj` is the ONE zero factor -- g is exactly the copy
predictor at step 0 (consistency loss 1.0 = gain 0, the old head's own
contract) and its gradient is the outer product of the live block output
with the residual, so it moves at step 1 and `code_proj` / `action_proj`
unfreeze at step 2 (the readout's query/key rule, not the two-factor
stall). Only the policy-readable rows exist here, so nothing a rollout
sees is privileged (`tests/test_transition_model.py` pins it with the
posterior as the positive control).
"""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.model.constants import (
    CLS_ROW,
    DYNAMICS_TARGET_ROWS,
    MOVE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    SEQUENCE_READ_MASK,
    TARGET_ROWS,
)
from rl.model.heads import FlatActionReadout, chosen_bank_rows
from rl.model.modules import MLP
from rl.model.trunk import Trunk

# The 1% unimix floor DreamerV3 puts under every categorical: the KL can
# never see a zero, and the straight-through sample keeps a gradient.
UNIMIX = 0.01


class TransitionOutput(NamedTuple):
    """Per-step outputs, all leading (T, ...). `pred` is the imagined
    next sequence decoded from the POSTERIOR sample; `pred_prior` is the
    no-gradient decode from the prior MODE and `ground_prior` its grounding
    read, the honest rollout-side numbers. Logits are f32; the code arrays are (T, G, K).
    """

    pred: jax.Array
    pred_prior: jax.Array
    prior_logits: jax.Array
    post_logits: jax.Array
    post_one_hot: jax.Array
    ground: jax.Array
    ground_prior: jax.Array
    mask_logits: jax.Array
    kind_logits: jax.Array
    done_logit: jax.Array


def unimix_probs(logits: jax.Array) -> jax.Array:
    """f32 softmax over the last axis with the unimix floor."""
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return (1.0 - UNIMIX) * probs + UNIMIX / logits.shape[-1]


def straight_through_sample(probs: jax.Array) -> jax.Array:
    """Argmax one-hot forward, the probabilities' gradient backward."""
    hard = jax.nn.one_hot(
        jnp.argmax(probs, axis=-1), probs.shape[-1], dtype=probs.dtype
    )
    return hard + probs - jax.lax.stop_gradient(probs)


class RowRead(nn.Module):
    """ONE Dense(D -> width) applied to every row, masked by validity and
    flattened in row order, so a reader sees WHICH row carries what. The
    mean pool it replaces cancelled a row's identity (an additive bias
    present at t and t+1 alike) and diluted a one-row change by the row
    count -- "my active lost 40%" and "theirs did" pooled to the same
    vector, and the posterior could not see the branch it was meant to
    code (irqeetfg 1266k-1312k: kl_long < kl_short, reveal margin 0.04).
    Position is the identity: the layout is named rows."""

    width: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, rows: jax.Array, row_valid: jax.Array) -> jax.Array:
        read = nn.Dense(self.width, dtype=self.dtype, name="read")(rows)
        return jnp.where(row_valid[:, None], read, 0).reshape(-1)


class TransitionModel(nn.Module):
    cfg: ConfigDict
    dtype: jnp.dtype

    @property
    def has_code(self) -> bool:
        return self.cfg.code_groups > 0

    def setup(self):
        model_size = self.cfg.block.model_size
        self.blocks = Trunk(self.cfg.block)
        self.action_proj = nn.Dense(model_size, dtype=self.dtype, name="action_proj")
        self.out_proj = nn.Dense(
            model_size,
            kernel_init=nn.initializers.zeros_init(),
            use_bias=False,
            dtype=self.dtype,
            name="out_proj",
        )
        if self.has_code:
            code_groups = self.cfg.code_groups
            assert model_size % code_groups == 0
            self.code_table = self.param(
                "code_table",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (code_groups, self.cfg.code_classes, model_size // code_groups),
            )
            self.code_proj = nn.Dense(model_size, dtype=self.dtype, name="code_proj")
            self.row_read = RowRead(
                self.cfg.row_read_width, self.dtype, name="row_read"
            )
            # `_read_net`, not `_net`: the nets that read the mean pool were
            # renamed so the by-path checkpoint merge inits them fresh --
            # the later layers' shapes are unchanged and would otherwise
            # resume weights trained on features that no longer exist.
            self.prior_read_net = MLP(**self.cfg.prior.mlp.to_dict())
            self.posterior_read_net = MLP(**self.cfg.posterior.mlp.to_dict())
        self.ground_delta_head = MLP(
            **self.cfg.ground.mlp.to_dict(), final_kernel_init=nn.initializers.zeros
        )
        self.mask_head = FlatActionReadout(self.cfg.action_head, name="mask_head")
        self.cls_head = MLP(**self.cfg.cls_head.mlp.to_dict())

    def code_embedding(self, code_one_hot: jax.Array) -> jax.Array:
        """(G, K) one-hot -> the concatenated code-table vector (D,)."""
        table = self.code_table.astype(self.dtype)
        return jnp.einsum("gk,gkd->gd", code_one_hot.astype(self.dtype), table).reshape(
            -1
        )

    def code_logits(self, net: MLP, features: jax.Array) -> jax.Array:
        return (
            net(features)
            .astype(jnp.float32)
            .reshape(self.cfg.code_groups, self.cfg.code_classes)
        )

    def imagine(
        self,
        rows: jax.Array,
        row_valid: jax.Array,
        src_row: jax.Array,
        tgt_row: jax.Array,
        code_one_hot: jax.Array | None,
    ) -> jax.Array:
        """One step of g over one (73, D) sequence. The conditioning -- the
        taken cell's readout rows and the code -- is ONE vector added to
        every row; the blocks route it. Rows the trunk zeroed stay zero."""
        conditioning = self.action_proj(jnp.concatenate((src_row, tgt_row), axis=-1))
        if code_one_hot is not None:
            conditioning = conditioning + self.code_proj(
                self.code_embedding(code_one_hot)
            )
        read_mask = SEQUENCE_READ_MASK[
            np.ix_(POLICY_READABLE_ROWS, POLICY_READABLE_ROWS)
        ]
        hidden = self.blocks(rows + conditioning[None], row_valid, read_mask)
        pred = rows + self.out_proj(hidden)
        return jnp.where(row_valid[:, None], pred, 0)

    def action_rows(self, rows: jax.Array, action_cell: jax.Array):
        return chosen_bank_rows(
            rows[PRIVATE_ROWS],
            rows[MOVE_ROWS],
            rows[TARGET_ROWS],
            action_cell.reshape(()),
        )

    def prior_features(
        self,
        rows: jax.Array,
        row_valid: jax.Array,
        src_row: jax.Array,
        tgt_row: jax.Array,
    ) -> jax.Array:
        return jnp.concatenate(
            (self.row_read(rows, row_valid), src_row, tgt_row), axis=-1
        )

    def prior(self, rows: jax.Array, row_valid: jax.Array, action_cell: jax.Array):
        """The rollout-side code distribution, (G, K) f32 logits."""
        src_row, tgt_row = self.action_rows(rows, action_cell)
        return self.code_logits(
            self.prior_read_net, self.prior_features(rows, row_valid, src_row, tgt_row)
        )

    def _step(
        self,
        rows: jax.Array,
        row_valid: jax.Array,
        action_cell: jax.Array,
        next_rows: jax.Array,
        next_valid: jax.Array,
    ) -> TransitionOutput:
        src_row, tgt_row = self.action_rows(rows, action_cell)
        code_shape = (self.cfg.code_groups, self.cfg.code_classes)
        if self.has_code:
            features = self.prior_features(rows, row_valid, src_row, tgt_row)
            prior_logits = self.code_logits(self.prior_read_net, features)
            # The posterior is LEARNER-ONLY: it reads the real next rows
            # (stop-gradient at the call site), which no rollout has -- as
            # the per-row CHANGE, the thing the code must explain, through
            # the same read the prior applies to the state.
            post_features = jnp.concatenate(
                (
                    features,
                    self.row_read(next_rows - rows, row_valid & next_valid),
                ),
                axis=-1,
            )
            post_logits = self.code_logits(self.posterior_read_net, post_features)
            post_one_hot = straight_through_sample(unimix_probs(post_logits))
            prior_mode = jax.nn.one_hot(
                jnp.argmax(prior_logits, axis=-1), code_shape[1], dtype=jnp.float32
            )
            pred = self.imagine(rows, row_valid, src_row, tgt_row, post_one_hot)
            pred_prior = jax.lax.stop_gradient(
                self.imagine(rows, row_valid, src_row, tgt_row, prior_mode)
            )
        else:
            prior_logits = jnp.zeros(code_shape, jnp.float32)
            post_logits = jnp.zeros(code_shape, jnp.float32)
            post_one_hot = jnp.zeros(code_shape, jnp.float32)
            pred = self.imagine(rows, row_valid, src_row, tgt_row, None)
            pred_prior = jax.lax.stop_gradient(pred)
        ground = self.ground_delta_head(pred[DYNAMICS_TARGET_ROWS])
        ground_prior = jax.lax.stop_gradient(
            self.ground_delta_head(pred_prior[DYNAMICS_TARGET_ROWS])
        )
        mask_logits = self.mask_head(
            pred[PRIVATE_ROWS], pred[MOVE_ROWS], pred[TARGET_ROWS]
        )
        cls_logits = self.cls_head(pred[CLS_ROW]).astype(jnp.float32)
        return TransitionOutput(
            pred=pred,
            pred_prior=pred_prior,
            prior_logits=prior_logits,
            post_logits=post_logits,
            post_one_hot=post_one_hot,
            ground=ground,
            ground_prior=ground_prior,
            mask_logits=mask_logits,
            kind_logits=cls_logits[:-1],
            done_logit=cls_logits[-1],
        )

    def __call__(
        self,
        rows: jax.Array,
        row_valid: jax.Array,
        action_cell: jax.Array,
        next_rows: jax.Array,
        next_valid: jax.Array,
    ) -> TransitionOutput:
        """Leading axis T on every input: rows (T, 73, D), row_valid (T,
        73), action_cell (T,), next_rows / next_valid the same rows one
        step on (the caller pairs them; the last step is self-paired and
        masked by the loss)."""
        return jax.vmap(self._step)(rows, row_valid, action_cell, next_rows, next_valid)
