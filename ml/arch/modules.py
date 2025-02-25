from enum import Enum, auto
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, get_args

import chex
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from flax.linen.dtypes import promote_dtype

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


def astype(x: chex.Array, dtype: jnp.dtype) -> chex.Array:
    """
    Cast x if necessary.

    Args:
        x (chex.Array): Input array.
        dtype (jnp.dtype): Desired data type.

    Returns:
        chex.Array: Casted array.
    """
    if x.dtype != dtype:
        return x.astype(dtype)
    else:
        return x


def activation_fn(array: chex.Array) -> chex.Array:
    """
    Apply activation function.

    Args:
        array (chex.Array): Input array.

    Returns:
        chex.Array: Activated array.
    """
    return jax.nn.silu(array) * array


def layer_norm(array: chex.Array) -> chex.Array:
    """
    Apply layer normalization.

    Args:
        array (chex.Array): Input array.

    Returns:
        chex.Array: Normalized array.
    """
    return nn.RMSNorm()(array)


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


class SNDense(nn.Module):
    """A linear transformation applied over the last dimension of the input with sigmaReparam.
    Attributes:
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    precision: Any = None
    bias_init: Callable[[Any], Any] = jax.nn.initializers.zeros
    std_init: float = 0.1
    denom_backward: bool = True

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:
        """
        Applies a linear transformation to the inputs along the last dimension.

        Args:
            inputs (chex.Array): The nd-array to be transformed.

        Returns:
            chex.Array: The transformed input.
        """
        initializing = self.is_mutable_collection("params")

        kernel = self.param(
            "kernel",
            jax.nn.initializers.normal(self.std_init),
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        # fake init
        s = jnp.ones((1, 1))
        vh = jnp.ones((1))
        if initializing:
            _, s, vh = lax.stop_gradient(jnp.linalg.svd(kernel, full_matrices=False))
        sigma_param = self.param(
            "sigma", jax.nn.initializers.ones, (1,), self.param_dtype
        )
        spectral_u_var = self.variable(
            "spectral", "u", lambda shape: jnp.ones(shape) * vh[0], vh[0].shape
        )
        spectral_norm_var = self.variable(
            "spectral", "norm", lambda shape: jnp.ones(shape) * s[0], (1,)
        )
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        # power method to compute spectral norm
        u = spectral_u_var.value
        v = lax.stop_gradient(jnp.matmul(kernel, u))
        # l2 norm
        v = lax.stop_gradient(v / jnp.linalg.norm(v, ord=2))
        u = lax.stop_gradient(jnp.matmul(jnp.transpose(kernel), v))
        # l2 norm
        u = lax.stop_gradient(u / jnp.linalg.norm(u, ord=2))
        if spectral_u_var.is_mutable() and not initializing:
            spectral_u_var.value = u
        sigma = jnp.einsum("c,cd,d->", v, kernel, u)

        if spectral_norm_var.is_mutable() and not initializing:
            spectral_norm_var.value = sigma

        inputs, sigma_param, sigma = promote_dtype(
            inputs, sigma_param, sigma, dtype=self.dtype
        )
        y = lax.dot_general(
            inputs,
            (sigma_param / sigma) * kernel,
            (((inputs.ndim - 1,), (0,)), ((), (()))),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class GatingType(Enum):
    NONE = auto()
    GLOBAL = auto()
    POINTWISE = auto()


class PoolMethod(Enum):
    MAX = auto()
    MEAN = auto()
    ATTN = auto()


class VectorResblock(nn.Module):
    """Fully connected residual block."""

    num_layers: int = 2
    hidden_size: Optional[int] = None
    use_layer_norm: bool = True
    use_spectral_linear: bool = False

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Fully connected residual block.

        Args:
            x (chex.Array): Input array.

        Returns:
            chex.Array: Output array.
        """
        chex.assert_type(x, jnp.float32)
        shortcut = x
        input_size = x.shape[-1]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                kwargs = dict(features=self.hidden_size or input_size)
            else:
                kwargs = dict(
                    features=input_size,
                    kernel_init=nn.initializers.normal(stddev=0.005),
                )
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            if self.use_spectral_linear:
                x = SNDense(**kwargs)(x)
            else:
                x = nn.Dense(**kwargs)(x)
        return x + shortcut


class Resnet(nn.Module):
    """A fully-connected resnet."""

    num_resblocks: int
    hidden_size: int = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        A fully-connected resnet.

        Args:
            x (chex.Array): Input array.

        Returns:
            chex.Array: Output array.
        """
        for _ in range(self.num_resblocks):
            x = VectorResblock(
                hidden_size=self.hidden_size, use_layer_norm=self.use_layer_norm
            )(x)
        return x


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


def get_rope_embedding(x: chex.Array) -> chex.Array:
    """
    Get rotary position embeddings.

    Args:
        x (chex.Array): Input array.

    Returns:
        chex.Array: Rotary position embeddings.
    """

    def negate_half(x: chex.Array):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    seq_len, dim = x.shape
    cos, sin = get_freqs(seq_len, dim)
    neg_half_x = negate_half(x)
    x_rope = (x * cos) + (neg_half_x * sin)
    return x_rope


def l2_normalize(x: chex.Array, epsilon: float = 1e-6) -> chex.Array:
    """
    Apply L2 normalization.

    Args:
        x (chex.Array): Input array.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        chex.Array: Normalized array.
    """
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + epsilon)


def escort_transform(
    x: chex.Array, mask: chex.Array, p: int = 2, axis: int = -1, eps: float = 1e-8
) -> chex.Array:
    """
    Apply escort transformation.

    Args:
        x (chex.Array): Input array.
        mask (chex.Array): Mask array.
        p (int, optional): Power value. Defaults to 2.
        axis (int, optional): Axis value. Defaults to -1.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        chex.Array: Transformed array.
    """
    abs_x = jnp.power(jnp.abs(x), p)
    denom = abs_x.sum(axis=axis, where=mask, keepdims=True)
    return abs_x / (denom + eps)


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
    query_need_pos: bool = False
    key_need_pos: bool = False
    use_spectral_linear: bool = False

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

        linear_mod = SNDense if self.use_spectral_linear else nn.Dense

        def _linear_projection(
            x: chex.Array, head_size: int, need_pos: bool = False
        ) -> chex.Array:
            y = linear_mod(self.num_heads * head_size)(x)
            if need_pos:
                y = get_rope_embedding(y)
            *leading_dims, _ = x.shape
            return y.reshape((*leading_dims, self.num_heads, head_size))

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = _linear_projection(
            query, key_size, self.query_need_pos
        )  # [T', H, Q=K]
        key_heads = _linear_projection(key, key_size, self.key_need_pos)  # [T, H, K]
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
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, 0)

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, s1_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = linear_mod(model_size)
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


class TransformerEncoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    key_size: int
    value_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_layer_norm: bool = True
    x_need_pos: bool = False
    y_need_pos: bool = False
    use_spectral_linear: bool = False
    resblocks_hidden_size: Optional[int] = None

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

        for _ in range(self.num_layers):
            if self.use_layer_norm:
                x_ln = layer_norm(x)
            else:
                x_ln = x
            mha = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                value_size=self.value_size,
                model_size=self.model_size,
                query_need_pos=self.x_need_pos,
                key_need_pos=self.y_need_pos,
                use_spectral_linear=self.use_spectral_linear,
            )(query=x_ln, key=x_ln, value=x_ln, mask=self_attn_mask)
            x = x + mha
            if self.use_layer_norm:
                x_ln = layer_norm(x)
            else:
                x_ln = x
            ffn = MLP(
                (self.resblocks_hidden_size, self.model_size),
                use_layer_norm=False,
                activate_first=False,
                use_spectral_linear=self.use_spectral_linear,
            )(x_ln)
            x = x + ffn
            x = jnp.where(mask[..., jnp.newaxis], x, 0)

        return x


class TransformerDecoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    key_size: int
    value_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_layer_norm: bool = True
    x_need_pos: bool = False
    y_need_pos: bool = False
    use_spectral_linear: bool = False
    resblocks_hidden_size: Optional[int] = None

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

        if self.use_layer_norm:
            y_ln = layer_norm(y)
        else:
            y_ln = y

        for _ in range(self.num_layers):
            if self.use_layer_norm:
                x_ln = layer_norm(x)
            else:
                x_ln = x
            ca = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                value_size=self.value_size,
                model_size=self.model_size,
                query_need_pos=self.x_need_pos,
                key_need_pos=self.y_need_pos,
                use_spectral_linear=self.use_spectral_linear,
            )(query=x_ln, key=y_ln, value=y_ln, mask=cross_attn_mask)
            x = x + ca
            if self.use_layer_norm:
                x_ln = layer_norm(x)
            else:
                x_ln = x
            ffn = MLP(
                (self.resblocks_hidden_size, self.model_size),
                use_layer_norm=False,
                activate_first=False,
                use_spectral_linear=self.use_spectral_linear,
            )(x_ln)
            x = x + ffn
            x = jnp.where(x_mask[..., jnp.newaxis], x, 0)

        return x


class PercieverIO(nn.Module):
    num_latents: int
    latent_dim: int

    num_layers: int = 1
    num_heads: int = 2
    use_layer_norm: bool = True
    resblocks_hidden_size: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        y: chex.Array,
        x_mask: chex.Array,
        y_mask: chex.Array,
    ) -> chex.Array:
        """
        Apply Perceiver IO model.

        Args:
            x (chex.Array): Input array.
            y (chex.Array): Input array.
            x_mask (chex.Array): Mask array for x.
            y_mask (chex.Array): Mask array for y.

        Returns:
            chex.Array: Output array.
        """
        latents = self.param(
            "latent",
            nn.initializers.truncated_normal(0.02),
            (self.num_latents, self.latent_dim),
        )

        encoder = TransformerDecoder(
            key_size=x.shape[-1],
            value_size=self.latent_dim,
            model_size=self.latent_dim,
            num_layers=1,
            num_heads=self.num_heads,
            use_layer_norm=self.use_layer_norm,
            resblocks_hidden_size=self.resblocks_hidden_size or 4 * self.latent_dim,
        )

        process = TransformerEncoder(
            key_size=self.latent_dim,
            value_size=self.latent_dim,
            model_size=self.latent_dim,
            num_layers=1,
            num_heads=self.num_heads,
            use_layer_norm=self.use_layer_norm,
            resblocks_hidden_size=self.resblocks_hidden_size or 4 * self.latent_dim,
        )

        decoder = TransformerDecoder(
            key_size=self.latent_dim,
            value_size=y.shape[-1],
            model_size=y.shape[-1],
            num_layers=1,
            num_heads=self.num_heads,
            use_layer_norm=self.use_layer_norm,
            resblocks_hidden_size=self.resblocks_hidden_size or 4 * y.shape[-1],
        )

        latents = encoder(latents, x, None, x_mask)
        for _ in range(self.num_layers):
            latents = process(latents)
        return decoder(y, latents, y_mask, None)


class MLP(nn.Module):
    """Apply unit-wise linear layers to the units."""

    layer_sizes: Sequence[int]
    use_layer_norm: bool = True
    activate_first: bool = True
    use_spectral_linear: bool = False

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Apply unit-wise linear layers to the units.

        Args:
            x (chex.Array): Input array.

        Returns:
            chex.Array: Output array.
        """
        linear_mod = SNDense if self.use_spectral_linear else nn.Dense
        for layer_index, size in enumerate(self.layer_sizes):
            if layer_index == 0 and not self.activate_first:
                # Skip layer normalization and activation for the first layer
                x = linear_mod(size)(x)
            else:
                if self.use_layer_norm:
                    x = layer_norm(x)
                x = activation_fn(x)
                x = linear_mod(size)(x)
        return x


class PointerLogits(nn.Module):
    """Produce logits using a pointer network.

    This is basically an attention mechanism between keys, coming from the units
    stream, and a single key, coming from the vector stream.
    """

    num_layers_query: int = 2
    num_layers_keys: int = 2
    key_size: int = 64
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, query: chex.Array, keys: chex.Array) -> chex.Array:
        """
        Produce logits using a pointer network.

        Args:
            query (chex.Array): Query array.
            keys (chex.Array): Keys array.

        Returns:
            chex.Array: Logits array.
        """
        # Query.
        for i in range(self.num_layers_query):
            if self.use_layer_norm:
                query = layer_norm(query)
            query = activation_fn(query)
            if i == self.num_layers_query - 1:
                query = nn.Dense(features=self.key_size)(query)
            else:
                query = nn.Dense(features=query.shape[-1])(query)

        # Keys.
        for i in range(self.num_layers_keys):
            if self.use_layer_norm:
                keys = layer_norm(keys)
            keys = activation_fn(keys)
            if i == self.num_layers_keys - 1:
                keys = nn.Dense(features=self.key_size)(keys)
            else:
                keys = nn.Dense(features=keys.shape[-1])(keys)

        # Pointer
        logits = query @ keys.T  # ij,j->i
        return logits * jax.lax.rsqrt(jnp.array(self.key_size, dtype=jnp.float32))


class ToAvgVector(nn.Module):
    """Per-unit processing then average over the units dimension."""

    units_hidden_sizes: Sequence[int]
    output_stream_size: int = None
    use_layer_norm: bool = True
    pool_method: PoolMethod = PoolMethod.MEAN

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        """
        Per-unit processing then average over the units dimension.

        Args:
            x (chex.Array): Input array.
            mask (chex.Array): Mask array.

        Returns:
            chex.Array: Output array.
        """
        for size in self.units_hidden_sizes:
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            x = nn.Dense(features=size)(x)

        if self.pool_method is PoolMethod.MEAN:
            x = (mask.astype(jnp.float32) @ x) / mask.sum().clip(min=1)
        elif self.pool_method is PoolMethod.MAX:
            x = jnp.where(jnp.expand_dims(mask, axis=-1), x, -1e9).max(0)
        else:
            raise ValueError(f"Pool method {self.pool_method} is not supported")

        if self.use_layer_norm:
            x = layer_norm(x)
        x = activation_fn(x)
        x = nn.Dense(features=self.output_stream_size or x.shape[-1])(x)
        return x


class VectorMerge(nn.Module):
    """Merge vector streams.

    Streams are first transformed through layer normalization, relu and linear
    layers, then summed, so they don't need to have the same size.
    Gating can also be used before the sum.

    If gating_type is not none, the sum is weighted using a softmax
    of the intermediate activations labelled above.
    """

    output_size: int
    gating_type: GatingType = GatingType.NONE
    use_layer_norm: bool = True

    def _compute_gate(
        self, inputs_to_gate: List[chex.Array], init_gate: List[chex.Array]
    ) -> List[chex.Array]:
        """
        Compute gating values.

        Args:
            inputs_to_gate (List[chex.Array]): List of input arrays to gate.
            init_gate (List[chex.Array]): List of initial gate arrays.

        Returns:
            List[chex.Array]: List of gating values.
        """
        w_init = nn.initializers.normal(stddev=0.005)
        b_init = nn.initializers.constant(0.0)
        if self.gating_type is GatingType.GLOBAL:
            gate_size = 1
        elif self.gating_type is GatingType.POINTWISE:
            gate_size = self.output_size
        else:
            raise ValueError(f"Gating type {self.gating_type} is not supported")
        if len(inputs_to_gate) == 2:
            # more efficient than the general version below
            gate = [
                nn.Dense(gate_size, kernel_init=w_init, bias_init=b_init)(y)
                for y in init_gate
            ]
            gate = sum(gate)
            sigmoid = jax.nn.sigmoid(gate)
            gate = [sigmoid, 1.0 - sigmoid]
        else:
            gate = [
                nn.Dense(
                    len(inputs_to_gate) * gate_size,
                    kernel_init=w_init,
                    bias_init=b_init,
                )(y)
                for y in init_gate
            ]
            gate = sum(gate)
            gate = jnp.reshape(gate, [len(inputs_to_gate), gate_size])
            gate = jax.nn.softmax(gate, axis=0)
            gate = [gate[i] for i in range(gate.shape[0])]
        return gate

    def _encode(
        self, inputs: Sequence[chex.Array]
    ) -> Tuple[List[chex.Array], List[chex.Array]]:
        """
        Encode inputs.

        Args:
            inputs (Sequence[chex.Array]): Sequence of input arrays.

        Returns:
            Tuple[List[chex.Array], List[chex.Array]]: Tuple of gate and output arrays.
        """
        gate, outputs = [], []
        for feature in inputs:
            size = feature.shape
            if size is None:
                feature = feature[jnp.newaxis]
            feature = astype(feature, jnp.float32)
            if self.use_layer_norm:
                feature = layer_norm(feature)
            feature = activation_fn(feature)
            gate.append(feature)
            outputs.append(nn.Dense(self.output_size)(feature))
        return gate, outputs

    @nn.compact
    def __call__(self, *inputs: List[chex.Array]) -> chex.Array:
        """
        Merge vector streams.

        Args:
            inputs (List[chex.Array]): List of input arrays.

        Returns:
            chex.Array: Merged output array.
        """
        gate, outputs = self._encode(inputs)
        if len(outputs) == 1:
            # Special case of 1-D inputs that do not need any gating.
            output = outputs[0]
        elif self.gating_type is GatingType.NONE:
            output = sum(outputs)
        else:
            gate = self._compute_gate(outputs, gate)
            data = [g * d for g, d in zip(gate, outputs)]
            output = sum(data)
        outputs = output
        return outputs


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


class SwiGLU(nn.Module):
    hidden_size: int = None

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Apply SwiGLU activation.

        Args:
            x (chex.Array): Input array.

        Returns:
            chex.Array: Output array.
        """
        feature_dim = x.shape[-1]
        hidden_size = self.hidden_size or feature_dim

        w1 = nn.Dense(hidden_size, use_bias=False)
        w2 = nn.Dense(hidden_size, use_bias=False)
        w3 = nn.Dense(feature_dim, use_bias=False)

        x1 = w1(x)
        x2 = w2(x)
        h = nn.silu(x1) * x2

        return w3(h)


class GLU(nn.Module):
    hidden_size: int = None

    @nn.compact
    def __call__(self, a: chex.Array, b: chex.Array) -> chex.Array:
        """
        Apply GLU activation.

        Args:
            a (chex.Array): Input array a.
            b (chex.Array): Input array b.

        Returns:
            chex.Array: Output array.
        """
        feature_dim = a.shape[-1]
        hidden_size = self.hidden_size or feature_dim

        w1 = nn.Dense(hidden_size, use_bias=False)
        w2 = nn.Dense(hidden_size, use_bias=False)
        w3 = nn.Dense(feature_dim, use_bias=False)

        a = w1(a)
        b = w2(b)
        h = nn.silu(a) * b

        return w3(h)


class DenseMultiHeadProjection(nn.Module):
    num_heads: int
    embed_dim: int
    output_dim: int = None

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Apply dense multi-head projection.

        Args:
            x (chex.Array): Input array.

        Returns:
            chex.Array: Output array.
        """

        def _apply_projection(l):
            return MLP((self.embed_dim,))(l)

        outputs = [_apply_projection(x) for _ in range(self.num_heads)]
        output = jnp.concatenate(outputs, axis=-1)
        return MLP((self.output_dim or x.shape[-1],))(output)


class GatedResidualLayer(nn.Module):
    @nn.compact
    def __call__(self, x: chex.Array, y: chex.Array) -> chex.Array:
        """
        Apply gated residual layer.

        Args:
            x (chex.Array): Original input array.
            y (chex.Array): Transformed output array.

        Returns:
            chex.Array: Blended output array.
        """
        gate = MLP((1,))(jnp.concatenate([x, y], axis=-1))  # Learn to mix x and y
        gate = jax.nn.sigmoid(gate)  # Values between 0 and 1
        return x * gate + y * (1 - gate)  # Blended output


class SumEmbeddings(nn.Module):
    output_size: int
    use_layer_norm: bool = True
    scaling_type: str = "fixed"  # 'learned', 'fixed', or 'attention'

    @nn.compact
    def __call__(
        self, encodings: List[chex.Array], embeddings: Optional[List[chex.Array]] = None
    ) -> chex.Array:
        """
        Sum embeddings.

        Args:
            encodings (List[chex.Array]): List of encoding arrays.
            embeddings (Optional[List[chex.Array]], optional): List of embedding arrays. Defaults to None.

        Returns:
            chex.Array: Summed embeddings array.
        """
        module_embeddings = []
        for i, encoding in enumerate(encodings):
            transformed = nn.Dense(self.output_size, use_bias=False)(encoding)
            module_embeddings.append(layer_norm(activation_fn(transformed)))

        if embeddings is not None:
            for i, embedding in enumerate(embeddings):
                module_embeddings.append(layer_norm(activation_fn(embedding)))

        if self.scaling_type == "learned":
            # Learn weights for each encoding
            weights = self.param(
                "mixing_weights", nn.initializers.ones, (len(encodings),)
            )
            weights = jax.nn.softmax(weights)
            scaled = sum(w * e for w, e in zip(weights, module_embeddings))

        elif self.scaling_type == "attention":
            # Use attention-like scaling
            query = self.param(
                "query", nn.initializers.normal(0.02), (self.output_size,)
            )
            scores = [jnp.einsum("...d,d->...", e, query) for e in module_embeddings]
            weights = jax.nn.softmax(jnp.stack(scores, axis=-1))
            scaled = sum(w[..., None] * e for w, e in zip(weights, module_embeddings))

        else:  # 'fixed'
            scaled = sum(module_embeddings) / jnp.sqrt(len(module_embeddings))

        return scaled


class MergeEmbeddings(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, embeddings: List[chex.Array]) -> chex.Array:
        """
        Merge embeddings.

        Args:
            embeddings (List[chex.Array]): List of embedding arrays.

        Returns:
            chex.Array: Merged embeddings array.
        """
        embeddings = [activation_fn(layer_norm(embedding)) for embedding in embeddings]
        return SumEmbeddings(self.output_size)(embeddings)


class TimestepResblock(nn.Module):
    num_layers: int = 2
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        """
        Apply timestep residual block.

        Args:
            x (chex.Array): Input array.
            mask (chex.Array): Mask array.

        Returns:
            chex.Array: Output array.
        """
        res = x
        for _ in range(self.num_layers):
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            x = nn.Conv(
                features=x.shape[-1],
                kernel_size=(5,),
                padding="SAME",
            )(x)
        return jnp.where(mask[..., None], x + res, 0)


class TimestepResnet(nn.Module):
    num_layers: int = 2
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        """
        Apply timestep residual network.

        Args:
            x (chex.Array): Input array.
            mask (chex.Array): Mask array.

        Returns:
            chex.Array: Output array.
        """
        for _ in range(self.num_layers):
            x = TimestepResblock(
                num_layers=self.num_layers, use_layer_norm=self.use_layer_norm
            )(x, mask)
        return x


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
