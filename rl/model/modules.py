from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


class GatNet(nn.Module):
    out_dim: int
    num_layers: int
    num_heads: int = 1
    max_edges: int = 2
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
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)

        valid_idx = jnp.where(valid_mask, size=self.max_edges, fill_value=-1)[0]
        take_mask = valid_idx != -1  # True only for real indices

        # Safely gather with padding -> zeros (avoids accidental -1 indexing)
        valid_node = jnp.take(x, valid_idx, axis=0, mode="fill", fill_value=0)
        valid_edge = jnp.take(e, valid_idx, axis=0, mode="fill", fill_value=0)
        where_mask = take_mask  # use this everywhere downstream

        messages = jnp.concatenate(
            (valid_node, valid_edge, jnp.broadcast_to(f, valid_edge.shape)), axis=-1
        )

        output = TransformerDecoder(
            qk_size=self.out_dim // self.num_heads,
            v_size=self.out_dim // self.num_heads,
            model_size=self.out_dim,
            resblocks_hidden_size=self.out_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )(
            layer_norm(valid_node),
            messages,
            create_attention_mask(where_mask, where_mask),
        )

        safe_idx = valid_idx.clip(min=0)
        weights = take_mask.astype(output.dtype)[..., None]  # (max_edges, 1)
        return jnp.zeros_like(x).at[safe_idx].add(output * weights)


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param(
            "scale", nn.initializers.zeros_init(), (x.shape[-1],), self.param_dtype
        )
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + 1e-6)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        return (normed_inputs * (1 + scale)).astype(self.dtype)


def activation_fn(array: jax.Array) -> jax.Array:
    """
    Apply activation function.
    """
    return nn.gelu(array)


def layer_norm(array: jax.Array) -> jax.Array:
    """
    Apply layer normalization with RMS Norm.
    """
    return RMSNorm(dtype=array.dtype)(array)


def softcap(array: jax.Array, max_value: int = 50) -> jax.Array:
    """
    Apply softcap function.
    """
    return max_value * nn.tanh(array / max_value)


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
    param_dtype: jnp.dtype = jnp.float32

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
        *q_leading_dims, _ = q.shape
        *kv_leading_dims, _ = kv.shape

        qk_size = self.qk_size
        v_size = self.v_size or self.qk_size
        model_size = self.model_size or self.qk_size * self.num_heads

        query_heads = nn.Dense(
            self.num_heads * qk_size,
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="q_proj",
        )(q).reshape((*q_leading_dims, self.num_heads, qk_size))
        key_heads = nn.Dense(
            self.num_heads * qk_size,
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="k_proj",
        )(kv).reshape((*kv_leading_dims, self.num_heads, qk_size))
        value_heads = nn.Dense(
            self.num_heads * v_size,
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="v_proj",
        )(kv).reshape((*kv_leading_dims, self.num_heads, v_size))

        if self.qk_layer_norm:
            query_heads = layer_norm(query_heads)
            key_heads = layer_norm(key_heads)

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

        attn_logits = jnp.where(mask, attn_logits, jnp.finfo(attn_logits.dtype).min)
        attn_weights = nn.softmax(attn_logits)
        attn_weights = jnp.where(mask, attn_weights, 0)

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        head_scale = self.param(
            "head_scale",
            nn.initializers.zeros_init(),
            (1, self.num_heads, 1),
            self.param_dtype,
        )
        attn = attn * jnp.asarray(1 + head_scale, dtype=self.dtype)
        attn = jnp.reshape(attn, (*q_leading_dims, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = nn.Dense(
            model_size, use_bias=self.use_bias, dtype=self.dtype, name="out_proj"
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


class TransformerEncoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    qk_size: int
    v_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_bias: bool = True
    need_pos: bool = False
    qk_layer_norm: bool = True
    resblocks_hidden_size: int | None = None
    use_post_attn_norm: bool = False
    use_post_ffw_norm: bool = False

    def layer(
        self,
        qkv: jax.Array,
        attn_mask: jax.Array,
        positionwise_mask: jax.Array,
        qkv_positions: jax.Array | None = None,
    ):
        qkv_ln = layer_norm(qkv)
        mha = MultiHeadAttention(
            num_heads=self.num_heads,
            qk_size=self.qk_size,
            v_size=self.v_size,
            model_size=self.model_size,
            qk_layer_norm=self.qk_layer_norm,
            use_bias=self.use_bias,
            need_pos=self.need_pos,
            dtype=qkv.dtype,
        )(
            q=qkv_ln,
            kv=qkv_ln,
            mask=attn_mask,
            q_positions=qkv_positions,
            kv_positions=qkv_positions,
        )
        if self.use_post_attn_norm:
            mha = layer_norm(mha)
        qkv = qkv + mha
        qkv_ln = layer_norm(qkv)
        ffn = MLP(
            self.resblocks_hidden_size, input_activation=False, use_layer_norm=False
        )(qkv_ln)
        if self.use_post_ffw_norm:
            ffn = layer_norm(ffn)
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
    use_post_attn_norm: bool = False
    use_post_ffw_norm: bool = False

    def layer(
        self,
        q: jax.Array,
        kv: jax.Array,
        attn_mask: jax.Array,
        positionwise_mask: jax.Array,
        q_positions: jax.Array | None = None,
        kv_positions: jax.Array | None = None,
    ):
        q_ln = layer_norm(q)
        kv_ln = layer_norm(kv)
        mha = MultiHeadAttention(
            num_heads=self.num_heads,
            qk_size=self.qk_size,
            v_size=self.v_size,
            model_size=self.model_size,
            use_bias=self.use_bias,
            qk_layer_norm=self.qk_layer_norm,
            need_pos=self.need_pos,
            dtype=q.dtype,
        )(
            q=q_ln,
            kv=kv_ln,
            mask=attn_mask,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )
        if self.use_post_attn_norm:
            mha = layer_norm(mha)
        q = q + mha
        q_ln = layer_norm(q)
        ffn = MLP(
            self.resblocks_hidden_size, input_activation=False, use_layer_norm=False
        )(q_ln)
        if self.use_post_ffw_norm:
            ffn = layer_norm(ffn)
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
    input_activation: bool = True
    final_kernel_init: Optional[nn.initializers.Initializer] = None

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

        for i, size in enumerate(layer_sizes):
            if self.use_layer_norm:
                x = layer_norm(x)
            if (i > 0) or (i == 0 and self.input_activation):
                x = activation_fn(x)
            dense_kwargs = dict()
            if (i == len(layer_sizes) - 1) and (self.final_kernel_init is not None):
                dense_kwargs["kernel_init"] = self.final_kernel_init
            x = nn.Dense(size, dtype=x.dtype, **dense_kwargs)(x)
        return x


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


class ZeroEmbedding:
    def __init__(self, dtype: jnp.dtype = jnp.float32):
        self.embeddings = jnp.zeros((5, 1), dtype=dtype)

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
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, *embeddings: list[jax.Array] | tuple[jax.Array]) -> jax.Array:
        """Sum embeddings."""
        num_embeddings = len(embeddings)
        if num_embeddings == 0:
            raise ValueError("No embeddings provided")

        aggregated = sum(
            nn.Dense(
                self.hidden_size or self.output_size, use_bias=False, dtype=self.dtype
            )(embedding)
            for embedding in embeddings
        ) + self.param(
            "bias", nn.initializers.zeros_init(), (self.output_size,), self.param_dtype
        )
        return VectorResblock()(aggregated.astype(self.dtype))


class VectorResblock(nn.Module):
    num_layers: int = 2
    hidden_size: Optional[int] = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        shortcut = x
        input_size = x.shape[-1]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                output_size = self.hidden_size or input_size
                dense_kwargs = dict()
            else:
                output_size = input_size
                dense_kwargs = dict(
                    kernel_init=nn.initializers.normal(5e-3),
                    bias_init=nn.initializers.zeros_init(),
                )
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            x = nn.Dense(output_size, dtype=x.dtype, **dense_kwargs)(x)
        return x + shortcut


class Resnet(nn.Module):
    num_resblocks: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.num_resblocks):
            x = VectorResblock(use_layer_norm=self.use_layer_norm)(x)
        return x


class PointerLogits(nn.Module):
    num_layers_query: int = 2
    num_layers_keys: int = 2
    key_size: int = 64
    use_layer_norm: bool = True
    keys_final_kernel_init: Optional[nn.initializers.Initializer] = None

    @nn.compact
    def __call__(self, query: jax.Array, keys: jax.Array) -> jax.Array:
        for i in range(self.num_layers_query):
            if self.use_layer_norm:
                query = layer_norm(query)
            query = activation_fn(query)
            if i == self.num_layers_query - 1:
                query = nn.Dense(self.key_size, dtype=query.dtype)(query)
            else:
                query = nn.Dense(query.shape[-1], dtype=query.dtype)(query)

        for i in range(self.num_layers_keys):
            if self.use_layer_norm:
                keys = layer_norm(keys)
            keys = activation_fn(keys)
            if i == self.num_layers_keys - 1:
                dense_kwargs = dict()
                if self.keys_final_kernel_init is not None:
                    dense_kwargs["kernel_init"] = self.keys_final_kernel_init
                keys = nn.Dense(self.key_size, dtype=keys.dtype, **dense_kwargs)(keys)
            else:
                keys = nn.Dense(keys.shape[-1], dtype=keys.dtype)(keys)

        return jnp.einsum("d,jd->j", query, keys) / np.sqrt(self.key_size).astype(
            query.dtype
        )


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
