"""The trunk: N standard pre-RMSNorm blocks over one sequence.

No gates. `RMSNorm` is `normed * (1 + scale)` with `scale` zeros-init, i.e.
exactly identity at step 0, and the residual adds are ungated -- so the
trunk is live at init by construction and an "is it wired" test needs no
gate opening.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from ml_collections import ConfigDict

from rl.model.modules import (
    COLLECT_INTERMEDIATES,
    FFWMLP,
    MultiHeadAttention,
    RMSNorm,
    SequenceNormalisation,
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
    def __call__(self, carry: tuple[jax.Array, jax.Array], read_mask: jax.Array):
        sequence, row_valid = carry
        # Validity AND the static leak partition (rl/model/constants.py
        # SEQUENCE_READ_MASK, or its policy-readable sub-block on the actor's
        # shorter sequence): policy-readable rows have no in-edge from the
        # learner-only partition at ANY block, so leak-freedom is transitive
        # across depth by induction -- see the matrix's own comment. It is a
        # scan BROADCAST input rather than a module-level import so the trunk
        # is agnostic to which rows it was handed.
        mask = create_attention_mask(row_valid, row_valid) & read_mask
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
    The step is memory-bandwidth-bound, so recomputing is genuinely cheaper
    than storing, and the cheapest policy is the one that stays.
    """

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self, sequence: jax.Array, row_valid: jax.Array, read_mask: jax.Array
    ) -> jax.Array:
        input_rows = sequence.shape[-2]
        num_registers = self.cfg.get("num_registers", 0)
        if num_registers < 0:
            raise ValueError("num_registers must be nonnegative")
        if num_registers:
            # ViT-style workspace: learned initial tokens, refreshed each
            # forward, mixed through every block and discarded at the output.
            registers = self.param(
                "register_embeddings",
                nn.initializers.normal(stddev=0.02),
                (num_registers, sequence.shape[-1]),
            ).astype(sequence.dtype)
            # The same input norm every row passes through: registers
            # enter at RMS 1 like everything else.
            registers = SequenceNormalisation(num_groups=1, name="register_norm")(
                registers,
                jnp.ones(num_registers, dtype=jnp.bool_),
                jnp.zeros(num_registers, dtype=jnp.int32),
            )
            registers = jnp.broadcast_to(
                registers, (*sequence.shape[:-2], *registers.shape)
            )
            sequence = jnp.concatenate([sequence, registers], axis=-2)
            row_valid = jnp.concatenate(
                [
                    row_valid,
                    jnp.ones((*row_valid.shape[:-1], num_registers), dtype=jnp.bool_),
                ],
                axis=-1,
            )
            # A shared register may read only sources readable by EVERY
            # original query. This preserves the original partition
            # transitively while letting all original rows read registers.
            shared_sources = jnp.all(read_mask, axis=-2, keepdims=True)
            register_reads = jnp.broadcast_to(
                shared_sources, (*read_mask.shape[:-2], num_registers, input_rows)
            )
            read_mask = jnp.concatenate([read_mask, register_reads], axis=-2)
            read_mask = jnp.concatenate(
                [
                    read_mask,
                    jnp.ones((*read_mask.shape[:-1], num_registers), dtype=jnp.bool_),
                ],
                axis=-1,
            )
        block = nn.remat(TrunkBlock, policy=jax.checkpoint_policies.nothing_saveable)
        # A sow is a silent no-op unless its collection is lifted through
        # every transform above it -- the intermediates collection must be
        # in `variable_axes` or the block's attention sow captures NOTHING.
        # Stacked along the block axis.
        variable_axes = {"params": 0}
        if COLLECT_INTERMEDIATES:
            variable_axes["intermediates"] = 0
        (sequence, _), _ = nn.scan(
            block,
            variable_axes=variable_axes,
            split_rngs={"params": True},
            in_axes=nn.broadcast,
            length=self.cfg.num_blocks,
        )(self.cfg, name="blocks")((sequence, row_valid), jnp.asarray(read_mask))
        return sequence[..., :input_rows, :]


def group_row_l2(
    sequence: jax.Array, row_valid: jax.Array, group_ids: jax.Array, num_groups: int
) -> tuple[jax.Array, jax.Array]:
    """Per-group residual magnitude: (sum of valid-row L2 norms, valid-row
    count), each (..., num_groups) over the trailing (rows, dim) axes. Every
    row ENTERS at RMS 1, so the trunk's output norm per group is the read of
    which rows the blocks write to. Summed rather than averaged so the
    caller's mean weights every valid row once."""
    l2 = jnp.linalg.norm(sequence.astype(jnp.float32), axis=-1)
    membership = jax.nn.one_hot(group_ids, num_groups, dtype=jnp.float32)
    valid = row_valid.astype(jnp.float32)
    # HIGHEST: the default f32 einsum runs at TF32 on the GPU (~1e-3
    # relative), and this is a norm read, not a matmul worth the speed.
    highest = jax.lax.Precision.HIGHEST
    l2_sum = jnp.einsum("...r,rg->...g", l2 * valid, membership, precision=highest)
    rows = jnp.einsum("...r,rg->...g", valid, membership, precision=highest)
    return l2_sum, rows


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
    residual. Participation is NaN when there is no spread at all (an
    all-identical set); cosine is NaN with fewer than two valid rows.
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
    num_pairs = pair.sum((-2, -1))
    cosine = jnp.where(
        num_pairs > 0,
        (cosines * pair).sum((-2, -1)) / jnp.maximum(num_pairs, 1),
        jnp.nan,
    )

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
