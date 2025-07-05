from typing import List, Literal, Optional, Sequence, Tuple, get_args

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

np.set_printoptions(precision=3, suppress=True)
jnp.set_printoptions(precision=3, suppress=True)


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
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


def activation_fn(array: chex.Array) -> chex.Array:
    """
    Apply activation function.

    Args:
        array (chex.Array): Input array.

    Returns:
        chex.Array: Activated array.
    """
    return nn.gelu(array)


def layer_norm(array: chex.Array) -> chex.Array:
    """
    Apply layer normalization.

    Args:
        array (chex.Array): Input array.

    Returns:
        chex.Array: Normalized array.
    """
    return RMSNorm()(array)


def softcap(array: chex.Array, max_value: int = 50) -> chex.Array:
    """
    Apply softcap function.

    Args:
        array (chex.Array): Input array.
        max_value (int, optional): Maximum value. Defaults to 50.

    Returns:
        chex.Array: Softcapped array.
    """
    return max_value * nn.tanh(array / max_value)


class Logits(nn.Module):
    """Logits for scalar heads."""

    num_logits: int = None
    num_linear_layers: int = 3
    use_layer_norm: bool = True
    kernel_init: Literal["zeros", "lecun", "small"] = "lecun"

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Logits for scalar heads.

        Args:
            x (chex.Array): Input array.

        Returns:
            chex.Array: Output array.
        """
        if self.kernel_init == "lecun":
            kernel_init_fn = nn.initializers.lecun_normal()
        elif self.kernel_init == "zeros":
            kernel_init_fn = nn.initializers.zeros_init()
        elif self.kernel_init == "small":
            kernel_init_fn = nn.initializers.uniform(1e-2)
        else:
            raise ValueError(
                f"{self.kernel_init} must be one of {get_args(self.kernel_init)}"
            )

        for i in range(self.num_linear_layers):
            if i == self.num_linear_layers - 1:
                output_size = self.num_logits or x.shape[-1]
            else:
                output_size = x.shape[-1]

            # Optionally apply LayerNorm
            if self.use_layer_norm:
                x = layer_norm(x)

            # Apply activation and dense layer with custom kernel initializer
            x = activation_fn(x)
            x = nn.Dense(features=output_size, kernel_init=kernel_init_fn)(x)
        return x


def get_freqs(
    seq_len: int, dim: int, base: int = 10000
) -> Tuple[chex.Array, chex.Array]:
    """
    Get frequency embeddings.

    Args:
        seq_len (int): Sequence length.
        dim (int): Dimension.
        base (int, optional): Base value. Defaults to 10000.

    Returns:
        Tuple[chex.Array, chex.Array]: Frequency embeddings.
    """
    theta = 1 / (base ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(seq_len)

    idx_theta = jnp.einsum("i,j->ij", t, theta)
    idx_theta = jnp.concatenate([idx_theta, idx_theta], axis=1)

    freqs_cos = jnp.cos(idx_theta)
    freqs_sin = jnp.sin(idx_theta)

    return freqs_cos, freqs_sin


def apply_rope(x: chex.Array, max_wavelength: int = 10_000) -> chex.Array:
    """
    Get rotary position embeddings.

    Args:
        x (chex.Array): Input array.

    Returns:
        chex.Array: Rotary position embeddings.
    """
    *_, seq_len, num_heads, head_dim = x.shape
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction

    positions = jnp.arange(seq_len, dtype=x.dtype)
    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(x, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(x.dtype)


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
    key_size: int
    value_size: Optional[int] = None
    model_size: Optional[int] = None
    qk_layer_norm: bool = False
    need_pos: bool = False

    @nn.compact
    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Computes (optionally masked) MHA with queries, keys & values.

        Args:
            query (chex.Array): Embeddings sequence used to compute queries.
            key (chex.Array): Embeddings sequence used to compute keys.
            value (chex.Array): Embeddings sequence used to compute values.
            mask (Optional[chex.Array], optional): Optional mask applied to attention weights.

        Returns:
            chex.Array: A new sequence of embeddings.
        """
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, s1_length, _ = query.shape

        key_size = self.key_size
        value_size = self.value_size or self.key_size
        model_size = self.model_size or self.key_size * self.num_heads

        def _linear_projection(
            x: chex.Array,
            head_size: int,
            use_layer_norm: bool = False,
            need_pos: bool = False,
        ) -> chex.Array:
            y = nn.Dense(self.num_heads * head_size, use_bias=False)(x)
            *leading_dims, _ = x.shape
            y = y.reshape((*leading_dims, self.num_heads, head_size))
            if use_layer_norm:
                y = layer_norm(y)
            if need_pos:
                y = apply_rope(y)
            return y

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = _linear_projection(
            query, key_size, self.qk_layer_norm, self.need_pos
        )  # [T', H, Q=K]
        key_heads = _linear_projection(
            key, key_size, self.qk_layer_norm, self.need_pos
        )  # [T, H, K]
        value_heads = _linear_projection(value, value_size)  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits / np.sqrt(key_size).astype(key.dtype)

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -2.3819763e38)

        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, 0)

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, s1_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = nn.Dense(model_size, use_bias=False)
        return final_projection(attn)  # [T', D']


def create_attention_mask(
    mask1: Optional[chex.Array] = None, mask2: Optional[chex.Array] = None
) -> Optional[chex.Array]:
    """
    Create a combined attention mask for cross-attention.

    Args:
        mask1 (Optional[chex.Array], optional): First mask array. Defaults to None.
        mask2 (Optional[chex.Array], optional): Second mask array. Defaults to None.

    Returns:
        Optional[chex.Array]: Combined attention mask.
    """
    if mask1 is None:
        return None

    if mask2 is None:
        mask2 = mask1

    return mask1[..., None, :, None] & mask2[..., None, None, :]


class FeedForward(nn.Module):
    """Feed forward module."""

    hidden_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        ff_gate = nn.Dense(self.hidden_dim, use_bias=False)(x)
        gate_value = nn.gelu(ff_gate)

        ff1 = nn.Dense(self.hidden_dim, use_bias=False)(x)
        activations = gate_value * ff1

        outputs = nn.Dense(x.shape[-1], use_bias=False)(activations)
        return outputs


class FeedForwardResidual(nn.Module):

    hidden_dim: int
    post_ffw_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        ffw = FeedForward(self.hidden_dim)(layer_norm(x))
        if self.post_ffw_norm:
            ffw = layer_norm(ffw)
        return x + ffw


class TransformerEncoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    key_size: int
    value_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_layer_norm: bool = True
    need_pos: bool = False
    qk_layer_norm: bool = False
    resblocks_hidden_size: Optional[int] = None
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
        ca_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Apply unit-wise resblocks, and transformer layers, to the units.

        Args:
            x (chex.Array): Input array.
            mask (Optional[chex.Array], optional): Mask array. Defaults to None.
            ca_mask (Optional[chex.Array], optional): Cross-attention mask array. Defaults to None.

        Returns:
            chex.Array: Output array.
        """
        if mask is None:
            mask = jnp.ones_like(x[..., 0], dtype=jnp.bool)

        self_attn_mask = create_attention_mask(mask)
        if ca_mask is not None:
            self_attn_mask = jnp.logical_and(self_attn_mask, ca_mask)
        positionwise_mask = self_attn_mask.any(axis=-1, keepdims=True).squeeze(0)

        for _ in range(self.num_layers):
            if self.use_layer_norm:
                x_ln = jax.vmap(layer_norm)(x)
            else:
                x_ln = x
            mha = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                value_size=self.value_size,
                model_size=self.model_size,
                qk_layer_norm=self.qk_layer_norm,
                need_pos=self.need_pos,
            )(query=x_ln, key=x_ln, value=x_ln, mask=self_attn_mask)
            if self.use_post_attn_norm:
                mha = jax.vmap(layer_norm)(mha)
            x = x + mha
            if self.use_layer_norm:
                x_ln = jax.vmap(layer_norm)(x)
            else:
                x_ln = x
            ffn = jax.vmap(FeedForward(self.resblocks_hidden_size))(x_ln)
            if self.use_post_ffw_norm:
                ffn = jax.vmap(layer_norm)(ffn)
            x = x + ffn
            x = jnp.where(positionwise_mask, x, 0)

        return x


class TransformerDecoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    key_size: int
    value_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_layer_norm: bool = True
    need_pos: bool = False
    qk_layer_norm: bool = False
    resblocks_hidden_size: Optional[int] = None
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        y: chex.Array,
        x_mask: Optional[chex.Array] = None,
        y_mask: Optional[chex.Array] = None,
        ca_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Apply unit-wise resblocks, and transformer layers, to the units.

        Args:
            x (chex.Array): Input array.
            y (chex.Array): Input array.
            x_mask (Optional[chex.Array], optional): Mask array for x. Defaults to None.
            y_mask (Optional[chex.Array], optional): Mask array for y. Defaults to None.
            ca_mask (Optional[chex.Array], optional): Cross-attention mask array. Defaults to None.

        Returns:
            chex.Array: Output array.
        """
        if x_mask is None:
            x_mask = jnp.ones_like(x[..., 0], dtype=jnp.bool)

        if y_mask is None:
            y_mask = jnp.ones_like(y[..., 0], dtype=jnp.bool)

        cross_attn_mask = create_attention_mask(x_mask, y_mask)
        if ca_mask is not None:
            cross_attn_mask = cross_attn_mask & ca_mask
        positionwise_mask = cross_attn_mask.any(axis=-1, keepdims=True).squeeze(0)

        if self.use_layer_norm:
            y_ln = jax.vmap(layer_norm)(y)
        else:
            y_ln = y

        for _ in range(self.num_layers):
            if self.use_layer_norm:
                x_ln = jax.vmap(layer_norm)(x)
            else:
                x_ln = x
            ca = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                value_size=self.value_size,
                model_size=self.model_size,
                qk_layer_norm=self.qk_layer_norm,
                need_pos=self.need_pos,
            )(query=x_ln, key=y_ln, value=y_ln, mask=cross_attn_mask)
            if self.use_post_attn_norm:
                ca = jax.vmap(layer_norm)(ca)
            x = x + ca
            if self.use_layer_norm:
                x_ln = jax.vmap(layer_norm)(x)
            else:
                x_ln = x
            ffn = jax.vmap(FeedForward(self.resblocks_hidden_size))(x_ln)
            if self.use_post_ffw_norm:
                ffn = jax.vmap(layer_norm)(ffn)
            x = x + ffn
            x = jnp.where(positionwise_mask, x, 0)

        return x


class MLP(nn.Module):
    """Apply unit-wise linear layers to the units."""

    layer_sizes: Sequence[int]
    use_layer_norm: bool = True
    activate_first: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Apply unit-wise linear layers to the units.

        Args:
            x (chex.Array): Input array.

        Returns:
            chex.Array: Output array.
        """
        for layer_index, size in enumerate(self.layer_sizes):
            if layer_index == 0 and not self.activate_first:
                # Skip layer normalization and activation for the first layer
                x = nn.Dense(size)(x)
            else:
                if self.use_layer_norm:
                    x = layer_norm(x)
                x = activation_fn(x)
                x = nn.Dense(size)(x)
        return x


class PretrainedEmbedding:
    def __init__(self, fpath: str):
        """
        Initialize the PretrainedEmbedding with a specified file path.

        Args:
            fpath (str): File path to the pretrained embeddings.
        """
        with open(fpath, "rb") as f:
            arr = np.load(f)
        self.embeddings = jnp.array(arr)

    def __call__(self, indices: chex.Array) -> chex.Array:
        """
        Get embeddings for the given indices.

        Args:
            indices (chex.Array): Indices array.

        Returns:
            chex.Array: Embeddings array.
        """
        return jnp.take(self.embeddings, indices, axis=0)


class BinaryEncoder:
    def __init__(self, num_bits: int, max_encodings: Optional[int] = None):
        """
        Initialize the BinaryEncoder with a specified number of bits.

        Args:
            num_bits (int): The length of the binary vector.
            max_encodings (Optional[int], optional): Maximum number of encodings. Defaults to None.
        """
        self.num_bits = num_bits
        self.max_encodings = max_encodings

        self.encodings = jnp.asarray(self._precompute_vectors())

    def _precompute_vectors(self) -> np.ndarray:
        """
        Precompute all binary vectors for values in the range [0, 2**num_bits - 1] using numpy operations.

        Returns:
            np.ndarray: A 2D array where each row is a binary vector representing a value.
        """
        total_values = self.max_encodings or 2**self.num_bits
        # Create a range of integers and convert to binary using numpy unpacking
        values = np.arange(total_values, dtype=int)[:, None]
        powers_of_two = 2 ** np.arange(self.num_bits - 1, -1, -1, dtype=int)
        vectors = (values & powers_of_two) > 0
        return vectors.astype(float)

    def __call__(self, indices: chex.Array) -> chex.Array:
        """
        Encode the given value into a binary vector.

        Args:
            indices (chex.Array): Indices array.

        Returns:
            chex.Array: Binary encoded array.
        """
        return jnp.take(self.encodings, indices, axis=0)


class SumEmbeddings(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, *embeddings: List[chex.Array]) -> chex.Array:
        """Sum embeddings."""
        return sum([nn.Dense(self.output_size)(embedding) for embedding in embeddings])


class MergeEmbeddings(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, *embeddings: List[chex.Array]) -> chex.Array:
        """
        Sum embeddings.

        Args:
            encodings (List[chex.Array]): List of encoding arrays.
            embeddings (Optional[List[chex.Array]], optional): List of embedding arrays. Defaults to None.

        Returns:
            chex.Array: Summed embeddings array.
        """
        num_module_embeddings = len(embeddings)
        if num_module_embeddings == 0:
            raise ValueError("No embeddings or encodings provided")

        outputs = []
        gates = []

        def _gate_layer_fn():
            return nn.Dense(
                num_module_embeddings, kernel_init=nn.initializers.normal(5e-3)
            )

        def _output_layer_fn():
            return nn.Dense(self.output_size)

        for embedding in embeddings:
            feature = layer_norm(embedding)
            feature = activation_fn(feature)
            gates.append(_gate_layer_fn()(feature))
            outputs.append(_output_layer_fn()(feature))

        stacked_outputs = jnp.stack(outputs, axis=0)
        gate = sum(gates)
        weights = nn.softmax(gate.reshape(-1), axis=-1)

        scale = jax.lax.rsqrt((weights**2).sum())
        return (weights @ stacked_outputs).reshape(-1) * scale


def one_hot_concat_jax(one_hot_encoded: List[Tuple[int, int]]) -> chex.Array:
    """
    Concatenate one-hot encoded arrays.

    Args:
        one_hot_encoded (List[Tuple[int, int]]): List of tuples containing indices and offsets.

    Returns:
        chex.Array: Concatenated one-hot encoded array.
    """
    sum_offsets = np.cumsum([0] + [offset for _, offset in one_hot_encoded])
    indices = jnp.stack(
        [idx + offset for (idx, _), offset in zip(one_hot_encoded, sum_offsets[:-1])]
    )
    return jnp.matmul(
        jnp.ones((len(indices),), jnp.float32),
        indices[:, jnp.newaxis] == jnp.arange(sum_offsets[-1]),
    )
