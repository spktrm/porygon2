import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from rl.model.utils import LARGE_NEGATIVE_BIAS

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


import jax
import jax.numpy as jnp
from flax import linen as nn


class GatLayer(nn.Module):
    out_dim: int
    num_heads: int = 1
    negative_slope: float = 0.2
    max_edges: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, e: jax.Array, f: jax.Array, valid_mask: jax.Array):
        """
        x: (N, D) node embeddings
        e: (N, D) edge embeddings, fully connected to valid nodes
        f: (D) global embedding, affects all valid nodes/edges
        valid_mask: (N,) bool
        returns: (N, out_dim)
        """
        N, D = x.shape
        # Choose head dim to cover out_dim across heads
        head_dim = math.ceil(self.out_dim / self.num_heads)

        # Projectors
        node_proj = nn.Dense(
            self.num_heads * head_dim,
            use_bias=False,
            name="node_proj",
            dtype=self.dtype,
        )
        edge_proj = nn.Dense(
            self.num_heads * head_dim,
            use_bias=False,
            name="edge_proj",
            dtype=self.dtype,
        )
        glob_proj = nn.Dense(
            self.num_heads * head_dim,
            use_bias=False,
            name="glob_proj",
            dtype=self.dtype,
        )
        msg_proj = nn.Dense(
            self.num_heads * head_dim, use_bias=False, name="msg_proj", dtype=self.dtype
        )
        out_proj = nn.Dense(
            self.out_dim, use_bias=True, name="out_proj", dtype=self.dtype
        )

        # Attention vector per head over [h_src || h_tgt || h_edge || h_global]
        attn_vec = self.param(
            "attn_vec",
            nn.initializers.lecun_normal(),
            (self.num_heads, 3 * head_dim + head_dim),
            self.dtype,
        )

        # Precompute projected representations
        H_nodes = node_proj(x).reshape(N, self.num_heads, head_dim)  # (N, H, Hd)
        H_edges = edge_proj(e).reshape(N, self.num_heads, head_dim)  # (N, H, Hd)
        H_glob = glob_proj(f[None, :]).reshape(
            1, self.num_heads, head_dim
        )  # (1, H, Hd), broadcastable

        # Build fully-connected edges among valid nodes, truncated to max_edges
        adj = jnp.outer(valid_mask, valid_mask)  # (N, N) bool
        num_edges = jnp.count_nonzero(adj)
        src_idx, tgt_idx = jnp.nonzero(adj, size=self.max_edges, fill_value=0)
        edge_mask = (jnp.arange(self.max_edges) < num_edges).astype(jnp.bool_)

        # Gather features for (src, tgt) edges
        h_src = jnp.take(H_nodes, src_idx, axis=0)  # (E, H, Hd)
        h_tgt = jnp.take(H_nodes, tgt_idx, axis=0)  # (E, H, Hd)
        h_edg = jnp.take(H_edges, src_idx, axis=0)  # (E, H, Hd)  (edge tied to source)
        h_glb = jnp.broadcast_to(H_glob, h_src.shape)  # (E, H, Hd)

        # Attention logits
        # concat along feature dim: (E, H, 4*Hd)
        cat = jnp.concatenate([h_src, h_tgt, h_edg, h_glb], axis=-1)
        # (E, H)
        logits = jnp.einsum("ehf,hf->eh", cat, attn_vec)
        logits = nn.leaky_relu(logits, negative_slope=self.negative_slope)

        # Mask out padded edges
        logits = jnp.where(edge_mask[:, None], logits, LARGE_NEGATIVE_BIAS)

        # Softmax over incoming edges per target node, per head
        # We'll compute per-head softmax with segment-wise operations.

        def segment_softmax_per_head(
            log_e: jax.Array, segments: jax.Array, num_segments: int
        ) -> jax.Array:
            # log_e: (E,)
            # segments: (E,)
            # returns: (E,)
            max_per_seg = jnp.full(
                (num_segments,), LARGE_NEGATIVE_BIAS, dtype=self.dtype
            )
            max_per_seg = max_per_seg.at[segments].max(log_e)
            log_e_centered = log_e - max_per_seg[segments]
            exp_e = jnp.exp(log_e_centered) * (segments >= 0)  # guard
            denom = jnp.zeros((num_segments,), dtype=self.dtype)
            denom = denom.at[segments].add(exp_e)
            return exp_e / (denom[segments] + 1e-9)

        # Compute attention weights α_{src->tgt}^{(h)}
        def per_head_softmax(h: int) -> jax.Array:
            return segment_softmax_per_head(logits[:, h], tgt_idx, N)

        alphas = jax.vmap(per_head_softmax)(jnp.arange(self.num_heads))  # (H, E)

        # Messages: function of source node, edge, and global
        msg_in = jnp.concatenate([h_src, h_edg, h_glb], axis=-1)  # (E, H, 3*Hd)
        msgs = msg_proj(
            msg_in.reshape(self.max_edges, self.num_heads * 3 * head_dim)
        ).reshape(
            self.max_edges, self.num_heads, head_dim
        )  # (E, H, Hd)

        # Weight by attention and aggregate to targets
        weighted = msgs * alphas[:, :, None]  # (E, H, Hd)
        # Scatter-add over targets
        agg = (
            jnp.zeros((N, self.num_heads, head_dim), dtype=self.dtype)
            .at[tgt_idx]
            .add(jnp.where(edge_mask[:, None, None], weighted, 0.0))
        )

        # Zero out invalid nodes (in case any padding crept in)
        agg = agg * valid_mask[:, None, None]

        # Combine heads and project out
        agg = agg.reshape(N, self.num_heads * head_dim)
        out = out_proj(agg)  # (N, out_dim)

        # Keep output zero for invalid nodes to be safe
        out = jnp.where(valid_mask[:, None], out, 0.0)
        return out


class GatNet(nn.Module):
    out_dim: int
    num_layers: int
    num_heads: int = 1
    negative_slope: float = 0.2
    max_edges: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, e: jax.Array, f: jax.Array, valid_mask: jax.Array):
        """
        x: (N, D) node embeddings
        e: (N, D) edge embeddings, fully connected to valid nodes
        f: (D) global embedding, affects all valid nodes/edges
        valid_mask: (N,) bool
        returns: (N, out_dim)
        """
        for _ in range(self.num_layers):
            mha = GatLayer(
                out_dim=self.out_dim,
                num_heads=self.num_heads,
                negative_slope=self.negative_slope,
                max_edges=self.max_edges,
                dtype=self.dtype,
            )(x, e, f, valid_mask)
            mha = layer_norm(mha, self.dtype)
            x = x + x
            qkv_ln = layer_norm(x, self.dtype)
            ffn = FeedForward(x.shape[-1], dtype=self.dtype)(qkv_ln)
            ffn = layer_norm(ffn, self.dtype)
            x = x + ffn
            x = jnp.where(valid_mask[..., None], x, 0)
        return x


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param(
            "scale", nn.initializers.zeros_init(), (x.shape[-1],), self.dtype
        )
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs


def activation_fn(array: jax.Array) -> jax.Array:
    """
    Apply activation function.
    """
    return nn.gelu(array)


def layer_norm(array: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """
    Apply layer normalization with RMS Norm.
    """
    return RMSNorm(dtype=dtype)(array)


def softcap(array: jax.Array, max_value: int = 50) -> jax.Array:
    """
    Apply softcap function.
    """
    return max_value * nn.tanh(array / max_value)


class Logits(nn.Module):
    """Logits for scalar heads."""

    num_logits: int = None
    num_linear_layers: int = 3
    use_layer_norm: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Logits for scalar heads."""

        for i in range(self.num_linear_layers):
            if i == self.num_linear_layers - 1:
                output_size = self.num_logits or x.shape[-1]
            else:
                output_size = x.shape[-1]

            # Optionally apply LayerNorm
            if self.use_layer_norm:
                x = layer_norm(x, self.dtype)

            # Apply activation and dense layer with custom kernel initializer
            x = activation_fn(x)
            x = nn.Dense(features=output_size, dtype=self.dtype)(x)
        return x


def apply_rope(
    inputs: jax.Array, positions: jax.Array, max_wavelength: int = 10_000
) -> jax.Array:
    """
    Get rotary position embeddings.

    Args:
        x (jax.Array): Input array.

    Returns:
        jax.Array: Rotary position embeddings.
    """
    *_, seq_len, num_heads, head_dim = inputs.shape
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.int32) / head_dim
    timescale = max_wavelength**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


def escort_transform(x: jax.Array, m: jax.Array = None, p: int = 2, axis: int = -1):
    if m is None:
        m = jnp.ones_like(x, dtype=bool)
    abs_x_p = jnp.where(m, jnp.abs(x) ** p, 0)
    denom = abs_x_p.sum(axis=axis, keepdims=True)
    return abs_x_p / (denom + (denom == 0))


class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA) module.

    This module is intended for attending over sequences of vectors.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    num_heads: int
    qk_size: int
    v_size: int | None = None
    model_size: int | None = None
    qk_layer_norm: bool = False
    need_pos: bool = False
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def _linear_projection(
        self, x: jax.Array, head_size: int, use_layer_norm: bool = False
    ) -> jax.Array:
        y = nn.Dense(
            self.num_heads * head_size,
            use_bias=self.use_bias,
            dtype=self.dtype,
        )(x)
        *leading_dims, _ = x.shape
        y = y.reshape((*leading_dims, self.num_heads, head_size))
        if use_layer_norm:
            y = layer_norm(y, self.dtype)
        return y

    @nn.compact
    def __call__(
        self,
        q: jax.Array,
        kv: jax.Array,
        mask: jax.Array,
        q_positions: jax.Array | None = None,
        kv_positions: jax.Array | None = None,
    ) -> jax.Array:
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, s1_length, _ = q.shape

        qk_size = self.qk_size
        v_size = self.v_size or self.qk_size
        model_size = self.model_size or self.qk_size * self.num_heads

        query_heads = self._linear_projection(q, qk_size, self.qk_layer_norm)
        key_heads = self._linear_projection(kv, qk_size, self.qk_layer_norm)
        value_heads = self._linear_projection(kv, v_size)

        if self.need_pos:
            if q_positions is None or kv_positions is None:
                raise ValueError(
                    "Rotary position embeddings require positions argument."
                )
            # Get the positions for the sequence.
            query_heads = apply_rope(query_heads, q_positions)
            key_heads = apply_rope(key_heads, kv_positions)

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits / np.sqrt(qk_size).astype(q.dtype)

        attn_logits = jnp.where(mask, attn_logits, LARGE_NEGATIVE_BIAS)
        attn_weights = nn.softmax(attn_logits)
        attn_weights = jnp.where(mask, attn_weights, 0)

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, s1_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = nn.Dense(
            model_size, use_bias=self.use_bias, dtype=self.dtype
        )
        return final_projection(attn)  # [T', D']


def create_attention_mask(
    mask1: jax.Array | None = None, mask2: jax.Array | None = None
) -> jax.Array | None:
    """
    Create a combined attention mask for cross-attention.

    Args:
        mask1 (Optional[jax.Array], optional): First mask array. Defaults to None.
        mask2 (Optional[jax.Array], optional): Second mask array. Defaults to None.

    Returns:
        Optional[jax.Array]: Combined attention mask.
    """
    if mask1 is None:
        return None

    if mask2 is None:
        mask2 = mask1

    mask = jnp.einsum("...s,...t->...st", mask1, mask2)
    return jnp.expand_dims(mask, axis=-3)


class FeedForward(nn.Module):
    """Feed forward module."""

    hidden_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        ff_gate = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        gate_value = nn.gelu(ff_gate)

        ff1 = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        activations = gate_value * ff1

        outputs = nn.Dense(x.shape[-1], dtype=self.dtype)(activations)
        return outputs


class FeedForwardResidual(nn.Module):

    hidden_dim: int
    post_ffw_norm: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        ffw = FeedForward(self.hidden_dim, dtype=self.dtype)(layer_norm(x, self.dtype))
        if self.post_ffw_norm:
            ffw = layer_norm(ffw, self.dtype)
        return x + ffw


class TransformerEncoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    qk_size: int
    v_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_bias: bool = True
    need_pos: bool = False
    qk_layer_norm: bool = False
    resblocks_hidden_size: int | None = None
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    dtype: jnp.dtype = jnp.float32

    def layer(
        self,
        qkv: jax.Array,
        attn_mask: jax.Array,
        positionwise_mask: jax.Array,
        qkv_positions: jax.Array | None = None,
    ):
        qkv_ln = layer_norm(qkv, self.dtype)
        mha = MultiHeadAttention(
            num_heads=self.num_heads,
            qk_size=self.qk_size,
            v_size=self.v_size,
            model_size=self.model_size,
            qk_layer_norm=self.qk_layer_norm,
            use_bias=self.use_bias,
            need_pos=self.need_pos,
            dtype=self.dtype,
        )(
            q=qkv_ln,
            kv=qkv_ln,
            mask=attn_mask,
            q_positions=qkv_positions,
            kv_positions=qkv_positions,
        )
        if self.use_post_attn_norm:
            mha = layer_norm(mha, self.dtype)
        qkv = qkv + mha
        qkv_ln = layer_norm(qkv, self.dtype)
        ffn = FeedForward(self.resblocks_hidden_size, dtype=self.dtype)(qkv_ln)
        if self.use_post_ffw_norm:
            ffn = layer_norm(ffn, self.dtype)
        qkv = qkv + ffn
        return jnp.where(positionwise_mask, qkv, 0)

    @nn.compact
    def __call__(
        self,
        qkv: jax.Array,
        attn_mask: jax.Array | None = None,
        qkv_positions: jax.Array | None = None,
    ) -> jax.Array:
        """
        Apply unit-wise resblocks, and transformer layers, to the units.

        Args:
            x (jax.Array): Input array.
            mask (Optional[jax.Array], optional): Mask array. Defaults to None.
            ca_mask (Optional[jax.Array], optional): Cross-attention mask array. Defaults to None.

        Returns:
            jax.Array: Output array.
        """
        if attn_mask is None:
            qkv_mask = jnp.ones_like(qkv[..., 0], dtype=jnp.bool)
            attn_mask = create_attention_mask(qkv_mask, qkv_mask)

        positionwise_mask = attn_mask.any(axis=-1, keepdims=True).squeeze(0)

        if self.need_pos and qkv_positions is None:
            qkv_positions = jnp.arange(qkv.shape[0], dtype=jnp.int32)

        for _ in range(self.num_layers):
            qkv = self.layer(qkv, attn_mask, positionwise_mask, qkv_positions)

        return qkv


class TransformerDecoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    qk_size: int
    v_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_bias: bool = True
    need_pos: bool = False
    qk_layer_norm: bool = False
    resblocks_hidden_size: int | None = None
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    dtype: jnp.dtype = jnp.float32

    def layer(
        self,
        q: jax.Array,
        kv: jax.Array,
        attn_mask: jax.Array,
        positionwise_mask: jax.Array,
        q_positions: jax.Array | None = None,
        kv_positions: jax.Array | None = None,
    ):
        q_ln = layer_norm(q, self.dtype)
        kv_ln = layer_norm(kv, self.dtype)
        mha = MultiHeadAttention(
            num_heads=self.num_heads,
            qk_size=self.qk_size,
            v_size=self.v_size,
            model_size=self.model_size,
            use_bias=self.use_bias,
            qk_layer_norm=self.qk_layer_norm,
            need_pos=self.need_pos,
            dtype=self.dtype,
        )(
            q=q_ln,
            kv=kv_ln,
            mask=attn_mask,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )
        if self.use_post_attn_norm:
            mha = layer_norm(mha, self.dtype)
        q = q + mha
        q_ln = layer_norm(q, self.dtype)
        ffn = FeedForward(self.resblocks_hidden_size, dtype=self.dtype)(q_ln)
        if self.use_post_ffw_norm:
            ffn = layer_norm(ffn, self.dtype)
        q = q + ffn
        return jnp.where(positionwise_mask, q, 0)

    @nn.compact
    def __call__(
        self,
        q: jax.Array,
        kv: jax.Array,
        attn_mask: jax.Array | None = None,
        q_positions: jax.Array | None = None,
        kv_positions: jax.Array | None = None,
    ) -> jax.Array:
        """
        Apply unit-wise resblocks, and transformer layers, to the units.

        Args:
            x (jax.Array): Input array.
            mask (Optional[jax.Array], optional): Mask array. Defaults to None.
            ca_mask (Optional[jax.Array], optional): Cross-attention mask array. Defaults to None.

        Returns:
            jax.Array: Output array.
        """
        if attn_mask is None:
            q_mask = jnp.ones_like(q[..., 0], dtype=jnp.bool)
            kv_mask = jnp.ones_like(kv[..., 0], dtype=jnp.bool)
            attn_mask = create_attention_mask(q_mask, kv_mask)

        positionwise_mask = attn_mask.any(axis=-1, keepdims=True).squeeze(0)

        if self.need_pos:
            if q_positions is None:
                q_positions = jnp.arange(q.shape[0], dtype=jnp.int32)
            if kv_positions is None:
                kv_positions = jnp.arange(kv.shape[0], dtype=jnp.int32)

        for _ in range(self.num_layers):
            q = self.layer(
                q, kv, attn_mask, positionwise_mask, q_positions, kv_positions
            )

        return q


class MLP(nn.Module):
    """Apply unit-wise linear layers to the units."""

    layer_sizes: int | tuple[int] | list[int]
    use_layer_norm: bool = True
    kernel_init: callable = nn.initializers.lecun_normal()
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply unit-wise linear layers to the units.

        Args:
            x (jax.Array): Input array.

        Returns:
            jax.Array: Output array.
        """
        if isinstance(self.layer_sizes, int):
            layer_sizes = (self.layer_sizes,)
        else:
            layer_sizes = self.layer_sizes

        for size in layer_sizes:
            if self.use_layer_norm:
                x = layer_norm(x, self.dtype)
            x = activation_fn(x)
            x = nn.Dense(size, kernel_init=self.kernel_init, dtype=self.dtype)(x)
        return x


class OutputLayer(nn.Module):
    layer_size: int
    gain: float = 1e-2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return MLP(
            self.layer_size,
            use_layer_norm=True,
            kernel_init=nn.initializers.orthogonal(self.gain),
            dtype=self.dtype,
        )(x)


class PretrainedEmbedding:
    def __init__(self, fpath: str, dtype: jnp.dtype = jnp.float32):
        """
        Initialize the PretrainedEmbedding with a specified file path.

        Args:
            fpath (str): File path to the pretrained embeddings.
        """
        with open(fpath, "rb") as f:
            arr = np.load(f)
        self.embeddings = jnp.asarray(arr, dtype=dtype)

    def __call__(self, indices: jax.Array) -> jax.Array:
        """
        Get embeddings for the given indices.

        Args:
            indices (jax.Array): Indices array.

        Returns:
            jax.Array: Embeddings array.
        """
        return jnp.take(self.embeddings, indices, axis=0)


class SumEmbeddings(nn.Module):
    output_size: int
    hidden_size: int | None = None
    dtype: jnp.dtype = jnp.float32
    norm_output: bool = True

    @nn.compact
    def __call__(self, *embeddings: list[jax.Array] | tuple[jax.Array]) -> jax.Array:
        """Sum embeddings."""
        embedding = sum(
            [
                nn.Dense(
                    self.hidden_size or self.output_size,
                    use_bias=False,
                    dtype=self.dtype,
                )(embedding)
                for embedding in embeddings
            ]
        ) + self.param(
            "bias",
            nn.initializers.zeros_init(),
            (self.hidden_size or self.output_size,),
            self.dtype,
        )
        output = MLP((self.output_size,), dtype=self.dtype)(embedding)
        if self.norm_output:
            output = layer_norm(output, self.dtype)
        return output


class MergeEmbeddings(nn.Module):
    output_size: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, *embeddings: list[jax.Array] | tuple[jax.Array]) -> jax.Array:
        """
        Sum embeddings.

        Args:
            encodings (List[jax.Array]): List of encoding arrays.
            embeddings (Optional[List[jax.Array]], optional): List of embedding arrays. Defaults to None.

        Returns:
            jax.Array: Summed embeddings array.
        """
        num_module_embeddings = len(embeddings)
        if num_module_embeddings == 0:
            raise ValueError("No embeddings or encodings provided")

        outputs = []
        gates = []

        def _gate_layer_fn():
            return nn.Dense(
                num_module_embeddings,
                kernel_init=nn.initializers.normal(5e-3),
                dtype=self.dtype,
            )

        def _output_layer_fn():
            return nn.Dense(self.output_size, dtype=self.dtype)

        for embedding in embeddings:
            feature = layer_norm(embedding, self.dtype)
            feature = activation_fn(feature)
            gates.append(_gate_layer_fn()(feature))
            outputs.append(_output_layer_fn()(feature))

        stacked_outputs = jnp.stack(outputs, axis=0)
        gate = sum(gates)
        weights = nn.softmax(gate.reshape(-1), axis=-1)

        scale = jax.lax.rsqrt((weights**2).sum())
        return (weights @ stacked_outputs).reshape(-1) * scale


def one_hot_concat_jax(
    one_hot_encoded: list[tuple[int, int]], dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """
    Concatenate one-hot encoded arrays.

    Args:
        one_hot_encoded (list[tuple[int, int]]): List of tuples containing indices and offsets.

    Returns:
        jax.Array: Concatenated one-hot encoded array.
    """
    sum_offsets = np.cumsum([0] + [offset for _, offset in one_hot_encoded])
    indices = jnp.stack(
        [idx + offset for (idx, _), offset in zip(one_hot_encoded, sum_offsets[:-1])]
    )
    return (
        (indices[:, jnp.newaxis] == np.arange(sum_offsets[-1]))
        .sum(axis=0)
        .astype(dtype)
    )
