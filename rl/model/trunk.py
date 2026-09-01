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
        # The block's output residual stream, for the offline row-homogeneity
        # read (rl/offline/trunk_homogeneity.py). Same gate as the attention
        # sow; training never allocates it.
        if COLLECT_INTERMEDIATES:
            self.sow("intermediates", "residual", sequence.astype(jnp.float32))
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
        # A sow is a silent no-op unless its collection is lifted through
        # every transform above it, and this scan lifted only params from
        # a1c18ed to 2026-09-02 -- so the block's attention sow captured
        # NOTHING and scripts/attn_probe.py never saw a trunk attention.
        # Stacked along the block axis, as the old round trunk's scan did.
        variable_axes = {"params": 0}
        if COLLECT_INTERMEDIATES:
            variable_axes["intermediates"] = 0
        (sequence, _), _ = nn.scan(
            block,
            variable_axes=variable_axes,
            split_rngs={"params": True},
            length=self.cfg.num_blocks,
        )(self.cfg, name="blocks")((sequence, row_valid), None)
        return sequence


def row_homogeneity(sequence: jax.Array) -> tuple[jax.Array, jax.Array]:
    """How alike a sequence's rows are: (mean off-diagonal cosine,
    participation ratio), each over the trailing (rows, dim) axes, batched
    over any leading ones. The over-smoothing instrument for a stack of
    ungated pre-norm blocks whose readouts are all PER ROW (Noci et al.
    2022, rank collapse): rows converging to one direction reads on the
    existing panels as "entropy at ceiling while the pointer params grow",
    which is indistinguishable from the phase-1 support-anchor shape.

    Cosine is UNCENTRED: q and k pass through RMSNorm, not LayerNorm, so a
    shared direction is exactly what every attention in the trunk sees.
    Participation is of the CENTRED Gram -- tr(G)^2 / ||G||_F^2, the number
    of equal-variance directions that would give the same spectrum, no
    eigendecomposition -- so it reads the spread AROUND the shared direction
    and the two disagree precisely when a common offset carries a live
    residual. NaN when there is no spread at all (an all-identical set).
    A valid row is one with nonzero norm: the trunk hard-zeroes invalid
    rows every block, so that is its own invariant.
    """
    rows = sequence.shape[-2]
    values = sequence.astype(jnp.float32)
    norms = jnp.linalg.norm(values, axis=-1)
    valid = norms > 0
    unit = values / jnp.maximum(norms, 1e-12)[..., None]
    pair = valid[..., :, None] & valid[..., None, :] & ~jnp.eye(rows, dtype=bool)
    cosines = jnp.einsum("...id,...jd->...ij", unit, unit)
    cosine = (cosines * pair).sum((-2, -1)) / jnp.maximum(pair.sum((-2, -1)), 1)

    num_valid = jnp.maximum(valid.sum(-1), 1)
    mean_row = (values * valid[..., None]).sum(-2) / num_valid[..., None]
    centred = (values - mean_row[..., None, :]) * valid[..., None]
    gram = jnp.einsum("...id,...jd->...ij", centred, centred)
    trace = jnp.trace(gram, axis1=-2, axis2=-1)
    frobenius_sq = jnp.sum(jnp.square(gram), axis=(-2, -1))
    participation = jnp.where(
        trace > 0, jnp.square(trace) / jnp.maximum(frobenius_sq, 1e-30), jnp.nan
    )
    return cosine, participation
