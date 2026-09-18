"""The trunk: N standard pre-RMSNorm blocks over one sequence.

No gates. `RMSNorm` is `normed * (1 + scale)` with `scale` zeros-init, i.e.
exactly identity at step 0, and the residual adds are ungated -- so the
trunk is live at init by construction and an "is it wired" test needs no
gate opening. Under `cfg.normalised_residual` the adds become nGPT's step on
the RMS-1 sphere (Loshchilov et al. 2024, eq. 10-11) with a per-block,
per-channel alpha at 0.05: not identity at init, but bounded -- the plain
residual stream grew 20x over run ijk4nyi4 (LESSONS 2026-09-17) and the
input's direction was gone from the output by 514k steps.
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
    create_attention_mask,
)


def unit_rms(x: jax.Array) -> jax.Array:
    """Each row rescaled to RMS 1 in `modules.RMSNorm`'s arithmetic (variance
    in f32, the rsqrt cast back to the row's dtype) without its scale: the
    normalised residual stream then shares the RMS-1 convention every row
    enters at (modules.SequenceNormalisation), so the zeros-init pre-norms
    stay identity at init. A zero row stays zero."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + 1e-6).astype(x.dtype)


class TrunkBlock(nn.Module):
    """Pre-norm self-attention, pre-norm SwiGLU MLP, both plain residual --
    or, under `cfg.normalised_residual`, both a step on the RMS-1 sphere.

    ONE MLP for every row -- deliberately not a per-token-type expert. Where
    a genuinely per-modality parameter is wanted it lives in the action
    readouts, on the axis it belongs to, not smeared across the trunk.
    """

    cfg: ConfigDict

    def _normalised_update(
        self, sequence: jax.Array, sublayer_out: jax.Array, alpha_name: str
    ) -> jax.Array:
        # nGPT eq. 10-11: stream and sub-layer output both at RMS 1, alpha
        # the per-channel step. Stored f32 like every leaf, cast at use.
        alpha = self.param(
            alpha_name,
            nn.initializers.constant(self.cfg.get("residual_alpha_init", 0.05)),
            (sequence.shape[-1],),
            jnp.float32,
        ).astype(sequence.dtype)
        return unit_rms(sequence + alpha * (unit_rms(sublayer_out) - sequence))

    @nn.compact
    def __call__(self, carry: tuple[jax.Array, jax.Array], read_mask: jax.Array):
        sequence, row_valid = carry
        # .get: the world-model flow block and the hand-rolled test trunks
        # build this ConfigDict without the key.
        normalised_residual = self.cfg.get("normalised_residual", False)
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
        )(
            q=RMSNorm(name="attention_query_norm")(sequence),
            kv=RMSNorm(name="attention_key_value_norm")(sequence),
            mask=mask,
        )
        if normalised_residual:
            sequence = self._normalised_update(sequence, attended, "attention_alpha")
        else:
            sequence = sequence + attended

        ffw_out = FFWMLP(
            hidden_size=self.cfg.hidden_size, use_bias=self.cfg.use_bias, name="ffw"
        )(RMSNorm(name="ffw_norm")(sequence))
        if normalised_residual:
            sequence = self._normalised_update(sequence, ffw_out, "ffw_alpha")
        else:
            sequence = sequence + ffw_out

        # Hard-zero invalid rows so a padded row never accumulates content.
        sequence = jnp.where(row_valid[..., None], sequence, 0)
        # The block's output residual stream, for the offline row-homogeneity
        # read (rl/probes/trunk_homogeneity.py). Same gate as the attention
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
        return sequence


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


def group_row_cosine(
    sequence_in: jax.Array,
    sequence_out: jax.Array,
    row_valid: jax.Array,
    group_ids: jax.Array,
    num_groups: int,
) -> jax.Array:
    """Per-group sum over valid rows of the cosine between a row as it
    enters the trunk and the same row as it leaves, (..., num_groups) over
    the trailing (rows, dim) axes. Under the normalised residual every row
    leaves at RMS 1, so the L2 read (group_row_l2) is blind by construction
    and this is what the blocks turned each row by. Summed for
    group_row_l2's reason; the caller divides by its row count."""
    inputs = sequence_in.astype(jnp.float32)
    outputs = sequence_out.astype(jnp.float32)
    dot = jnp.sum(inputs * outputs, axis=-1)
    norms = jnp.linalg.norm(inputs, axis=-1) * jnp.linalg.norm(outputs, axis=-1)
    cosine = dot / jnp.maximum(norms, 1e-12)
    membership = jax.nn.one_hot(group_ids, num_groups, dtype=jnp.float32)
    valid = row_valid.astype(jnp.float32)
    highest = jax.lax.Precision.HIGHEST
    return jnp.einsum("...r,rg->...g", cosine * valid, membership, precision=highest)


# nGPT's normalize_matrices (train.py): every vector that lives in embedding
# space is unit-L2 after each optimiser step. Flax kernels are (in, out),
# stacked (blocks, in, out) under the scan: the projections INTO a block's
# heads or hidden layer hold one embedding-space vector per column (the
# input axis), the projections back out hold one per row (the output axis).
_TRUNK_BLOCKS_PATH = ("params", "encoder", "trunk", "blocks")
_INPUT_AXIS_KERNELS = frozenset(
    {
        ("attention", "q_proj", "kernel"),
        ("attention", "k_proj", "kernel"),
        ("attention", "v_proj", "kernel"),
        ("ffw", "gate_up", "kernel"),
    }
)
_OUTPUT_AXIS_KERNELS = frozenset(
    {("attention", "out_proj", "kernel"), ("ffw", "down", "kernel")}
)


def project_trunk_kernels(variables):
    """The trunk block kernels with their embedding-space vectors rescaled to
    unit L2; every other leaf (biases, norm scales, alphas, every module
    outside the trunk) returned untouched."""

    def project(path, leaf):
        keys = tuple(getattr(entry, "key", None) for entry in path)
        if keys[: len(_TRUNK_BLOCKS_PATH)] != _TRUNK_BLOCKS_PATH:
            return leaf
        tail = keys[-3:]
        if tail in _INPUT_AXIS_KERNELS:
            axis = -2
        elif tail in _OUTPUT_AXIS_KERNELS:
            axis = -1
        else:
            return leaf
        kernel = leaf.astype(jnp.float32)
        norm = jnp.linalg.norm(kernel, axis=axis, keepdims=True)
        return (kernel / jnp.maximum(norm, 1e-12)).astype(leaf.dtype)

    return jax.tree_util.tree_map_with_path(project, variables)


def row_homogeneity(sequence: jax.Array) -> tuple[jax.Array, jax.Array]:
    """How alike a sequence's rows are: (mean off-diagonal cosine,
    participation ratio), each over the trailing (rows, dim) axes, batched
    over any leading ones. The over-smoothing instrument for a stack of
    ungated pre-norm blocks whose readouts are all PER ROW (Noci et al.
    2022, rank collapse): rows converging to one direction reads on the
    existing panels as "entropy at ceiling while the pointer params grow",
    which is indistinguishable from the phase-1 support-anchor shape.

    Every channel is first scaled to unit RMS over the valid rows, so the
    reading is about how many channels the rows agree on, not about which
    channels are large: a handful of high-magnitude channels shared by
    every row (a common offset, a massive activation) would otherwise carry
    the raw cosine to 1 while the rest of the features disagree.
    Cosine is UNCENTRED after that scaling: q and k pass through RMSNorm,
    not LayerNorm, so a shared direction is still what every attention in
    the trunk sees. Participation is of the CENTRED Gram of the same scaled
    rows -- tr(G)^2 / ||G||_F^2, the number of equal-variance directions
    that would give the same spectrum, no eigendecomposition -- so it reads
    the spread AROUND the shared direction. Participation is NaN when there
    is no spread at all (an all-identical set); cosine is NaN with fewer
    than two valid rows. A valid row is one with nonzero norm: the trunk
    hard-zeroes invalid rows every block, so that is its own invariant.
    """
    rows = sequence.shape[-2]
    raw = sequence.astype(jnp.float32)
    valid = jnp.linalg.norm(raw, axis=-1) > 0
    num_valid = jnp.maximum(valid.sum(-1), 1)
    channel_rms = jnp.sqrt(
        (jnp.square(raw) * valid[..., None]).sum(-2) / num_valid[..., None]
    )
    values = raw / jnp.maximum(channel_rms, 1e-12)[..., None, :]
    norms = jnp.linalg.norm(values, axis=-1)
    unit = values / jnp.maximum(norms, 1e-12)[..., None]
    pair = valid[..., :, None] & valid[..., None, :] & ~jnp.eye(rows, dtype=bool)
    cosines = jnp.einsum("...id,...jd->...ij", unit, unit)
    num_pairs = pair.sum((-2, -1))
    cosine = jnp.where(
        num_pairs > 0,
        (cosines * pair).sum((-2, -1)) / jnp.maximum(num_pairs, 1),
        jnp.nan,
    )

    mean_row = (values * valid[..., None]).sum(-2) / num_valid[..., None]
    centred = (values - mean_row[..., None, :]) * valid[..., None]
    gram = jnp.einsum("...id,...jd->...ij", centred, centred)
    trace = jnp.trace(gram, axis1=-2, axis2=-1)
    frobenius_sq = jnp.sum(jnp.square(gram), axis=(-2, -1))
    participation = jnp.where(
        trace > 0, jnp.square(trace) / jnp.maximum(frobenius_sq, 1e-30), jnp.nan
    )
    return cosine, participation
