"""The trunk: N standard pre-RMSNorm blocks over one sequence.

Replaces `RoundBlock` (2026-08-29), which carried three separate residual
streams -- 48 Perceiver latents, 41 action slots, 4 value queries -- wired
together by five individually-gated, block-masked attentions per round, four
rounds deep, at 3.69M parameters a round. Every route those masks encoded is
a subset of one all-pairs attention over the 61 rows the sequence now has,
and at 61 rows the trunk can simply carry them: 61 x 61 is 3.7k attention
cells against the 24k the old routing plus its two feeding cross-attention
reads paid, so the masks were buying nothing but their own complexity.

No gates. `RMSNorm` is `normed * (1 + scale)` with `scale` zeros-init, i.e.
exactly identity at step 0, and the residual adds are ungated -- so the
trunk is live at init by construction and an "is it wired" test needs no
gate opening. That also retires the 2026-08-24 gate-contribution finding
structurally rather than by tuning it.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from ml_collections import ConfigDict

from rl.model.constants import SEQUENCE_READ_MASK
from rl.model.modules import (
    COLLECT_INTERMEDIATES,
    FFWMLP,
    MultiHeadAttention,
    RMSNorm,
    create_attention_mask,
)


class TrunkBlock(nn.Module):
    """Pre-norm self-attention, pre-norm SwiGLU MLP, both plain residual.

    ONE MLP for every row -- deliberately not a per-token-type expert. Where
    a genuinely per-modality parameter is wanted it lives in the action
    readouts, on the axis it belongs to, not smeared across the trunk.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, carry: tuple[jax.Array, jax.Array], _):
        sequence, row_valid = carry
        # Validity AND the static leak partition (rl/model/constants.py
        # SEQUENCE_READ_MASK): policy-readable rows have no in-edge from the
        # learner-only partition at ANY block, so leak-freedom is transitive
        # across depth by induction -- see the matrix's own comment.
        mask = create_attention_mask(row_valid, row_valid) & jnp.asarray(
            SEQUENCE_READ_MASK
        )
        attended = MultiHeadAttention(
            name="attention",
            num_heads=self.cfg.num_heads,
            qk_size=self.cfg.qk_size,
            v_size=self.cfg.v_size,
            model_size=self.cfg.model_size,
            qk_layer_norm=self.cfg.qk_layer_norm,
            use_bias=self.cfg.use_bias,
            dtype=sequence.dtype,
            collect_intermediates=COLLECT_INTERMEDIATES,
        )(q=RMSNorm()(sequence), kv=RMSNorm()(sequence), mask=mask)
        sequence = sequence + attended

        sequence = sequence + FFWMLP(
            hidden_size=self.cfg.hidden_size, use_bias=self.cfg.use_bias, name="ffw"
        )(RMSNorm()(sequence))

        # Hard-zero invalid rows so a padded row never accumulates content.
        sequence = jnp.where(row_valid[..., None], sequence, 0)
        return (sequence, row_valid), None


class Trunk(nn.Module):
    """`num_blocks` unshared `TrunkBlock`s, scanned and rematted.

    `nothing_saveable`, not the house `checkpoint_dots`: the latter saves
    exactly the wide SwiGLU hidden activations that dominate the backward
    pass's memory, which is what OOM'd the train step when it was tried.

    MEASURED, not assumed (2026-09-01 sweep, full train_step compiled at the
    largest lattice entry (64, 256) x batch 4; XLA memory_analysis temp +
    15-step timing): the step is memory-bandwidth-bound, so recomputing is
    genuinely cheaper than storing -- NO remat is both 3.8x the memory AND
    ~10% SLOWER. Full table (trunk x entity pool, temp MiB / steps per sec):
    nothing+nothing 796/12.26 (this), nothing+dots 1071/12.65, dots+nothing
    1244/12.32, dots+dots 1508/12.60, none+nothing 3019/11.03, none+dots
    3402/11.29. The fastest fitting variant buys +3.2% for +275MiB, landing
    on the >=1.5GB headroom boundary (the 12GB box peaked ~10.5GB all-in),
    so the cheapest policy stays.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(self, sequence: jax.Array, row_valid: jax.Array) -> jax.Array:
        block = nn.remat(TrunkBlock, policy=jax.checkpoint_policies.nothing_saveable)
        (sequence, _), _ = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            length=self.cfg.num_blocks,
        )(self.cfg, name="blocks")((sequence, row_valid), None)
        return sequence
