import math
from typing import Optional

import flax.linen as nn
import jax
import jax.nn.initializers as initjax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact
from jax import lax

np.set_printoptions(precision=2, suppress=True)
jnp.set_printoptions(precision=2, suppress=True)


class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],))

        x_fp32 = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True)
        normed_inputs = x * jax.lax.rsqrt(var + 1e-6).astype(x.dtype)

        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1)).astype(x.dtype)
        return normed_inputs * (1 + scale)


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
    dtype: jnp.dtype | None = None
    std_init: float = 0.1
    denom_backward: bool = True

    @compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
            inputs: The nd-array to be transformed.
        Returns:
            The transformed input.
        """
        initializing = self.is_mutable_collection("params")

        kernel = self.param(
            "kernel",
            initjax.normal(self.std_init),
            (jnp.shape(inputs)[-1], self.features),
        )

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.features,))
        else:
            bias = None

        # fake init
        s = jnp.ones((1, 1))
        vh = jnp.ones((1))
        if initializing:
            _, s, vh = lax.stop_gradient(jnp.linalg.svd(kernel, full_matrices=False))
        sigma_param = self.param("sigma", nn.initializers.ones_init(), (1,))
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
            (((inputs.ndim - 1,), (0,)), ((), ())),
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


def activation_fn(array: jax.Array) -> jax.Array:
    """
    Apply activation function.
    """
    return nn.gelu(array)


def layer_norm(array: jax.Array) -> jax.Array:
    """
    Apply layer normalization with RMS Norm.
    """
    return RMSNorm()(array)


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
    implementation: Optional[str] = None  # "cudnn"

    @nn.compact
    def __call__(
        self,
        q: jax.Array,
        kv: jax.Array,
        mask: jax.Array,
        q_positions: jax.Array | None = None,
        kv_positions: jax.Array | None = None,
    ) -> jax.Array:
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
            if len(q_positions.shape) == 1:
                q_positions = q_positions[..., jnp.newaxis]
            if len(kv_positions.shape) == 1:
                kv_positions = kv_positions[..., jnp.newaxis]

            _apply_rope = jax.vmap(apply_rope, in_axes=(-1, -1), out_axes=-1)

            # Get the positions for the sequence.
            axial_query_heads = query_heads.reshape(
                *query_heads.shape[:-1], -1, q_positions.shape[-1]
            )
            query_heads = _apply_rope(axial_query_heads, q_positions).reshape(
                *query_heads.shape
            )

            axial_key_heads = key_heads.reshape(
                *key_heads.shape[:-1], -1, kv_positions.shape[-1]
            )
            key_heads = _apply_rope(axial_key_heads, kv_positions).reshape(
                *key_heads.shape
            )

        # target_shape = list(mask.shape)
        # target_shape[-3] = self.num_heads
        # mask = jnp.broadcast_to(mask, target_shape)

        # attn = jax.nn.dot_product_attention(
        #     query_heads,
        #     key_heads,
        #     value_heads,
        #     mask=mask,
        #     implementation=self.implementation,
        # )

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = softcap(attn_logits / np.sqrt(qk_size).astype(q.dtype))

        attn_logits = jnp.where(mask, attn_logits, -1e9)
        attn_log_probs = nn.log_softmax(attn_logits, axis=-1)
        attn_probs = jnp.exp(attn_log_probs)
        attn_probs = jnp.where(mask, attn_probs, 0)

        attn_entropy = -jnp.sum(attn_probs * attn_log_probs, axis=-1) / jnp.log(
            mask.sum(axis=-1)
        )

        # Capture attention weights and entropy as Flax intermediates so that
        # visualisation scripts can inspect them without modifying the forward pass.
        # self.sow(
        #     "intermediates",
        #     "attn_weights",
        #     attn_probs.astype(jnp.float32),
        # )
        # self.sow(
        #     "intermediates",
        #     "attn_entropy",
        #     jnp.nan_to_num(attn_entropy, nan=0.0).astype(jnp.float32),
        # )

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_probs, value_heads)
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

    return mask1[..., None, :, None] & mask2[..., None, None, :]


def norm_ratio(x: jax.Array, y: jax.Array, axis: int = -1) -> jax.Array:
    """
    Compute the ratio of the norms of two arrays.

    Args:
        x (jax.Array): First array.
        y (jax.Array): Second array.
        axis (int, optional): Axis along which to compute the norms. Defaults to -1.

    Returns:
        jax.Array: Ratio of the norms of the two arrays.
    """
    x_norm = jnp.linalg.norm(x, axis=axis)
    y_norm = jnp.linalg.norm(y, axis=axis)
    return jnp.where(x_norm == 0, 0, x_norm / y_norm)


class TransformerEncoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    qk_size: int
    v_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_bias: bool = False
    need_pos: bool = False
    qk_layer_norm: bool = True
    resblocks_hidden_size: int | None = None
    init_residual_scale: float = 1.0

    def layer(
        self,
        layer_idx: int,
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
        mha_a = self.param(
            f"mha_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        qkv = qkv + mha_a.astype(qkv.dtype) * mha
        qkv_ln = layer_norm(qkv)
        ffn = FFWMLP(
            hidden_size=self.resblocks_hidden_size,
            use_bias=self.use_bias,
        )(qkv_ln)
        ffn_a = self.param(
            f"ffn_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        qkv = qkv + ffn_a.astype(qkv.dtype) * ffn
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

        for layer_idx in range(self.num_layers):
            qkv = self.layer(
                layer_idx, qkv, attn_mask, positionwise_mask, qkv_positions
            )

        return layer_norm(qkv)


class TransformerDecoder(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    qk_size: int
    v_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_bias: bool = False
    need_pos: bool = False
    qk_layer_norm: bool = True
    resblocks_hidden_size: int | None = None
    init_residual_scale: float = 1.0

    def layer(
        self,
        layer_idx: int,
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
        mha_a = self.param(
            f"mha_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        q = q + mha_a.astype(q.dtype) * mha
        qkv_ln = layer_norm(q)
        ffn = FFWMLP(
            hidden_size=self.resblocks_hidden_size,
            use_bias=self.use_bias,
        )(qkv_ln)
        ffn_a = self.param(
            f"ffn_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        q = q + ffn_a.astype(q.dtype) * ffn
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

        positionwise_mask = attn_mask.any(axis=-1, keepdims=True).squeeze(-3)

        if self.need_pos:
            if q_positions is None:
                q_positions = jnp.arange(q.shape[0], dtype=jnp.int32)
            if kv_positions is None:
                kv_positions = jnp.arange(kv.shape[0], dtype=jnp.int32)

        for layer_idx in range(self.num_layers):
            q = self.layer(
                layer_idx,
                q,
                kv,
                attn_mask,
                positionwise_mask,
                q_positions,
                kv_positions,
            )

        return layer_norm(q)


class Transformer(nn.Module):

    qk_size: int
    v_size: int
    model_size: int
    num_layers: int
    num_heads: int
    use_bias: bool = False
    need_pos: bool = False
    qk_layer_norm: bool = True
    resblocks_hidden_size: int | None = None
    init_residual_scale: float = 1.0

    def decoder_layer(
        self,
        layer_idx: int,
        q: jax.Array,
        kv: jax.Array,
        self_attn_mask: jax.Array,
        cross_attn_mask: jax.Array,
        positionwise_mask: jax.Array,
        q_positions: jax.Array | None = None,
        kv_positions: jax.Array | None = None,
    ) -> jax.Array:
        q_ln = layer_norm(q)
        mhsa = MultiHeadAttention(
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
            kv=q_ln,
            mask=self_attn_mask,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )
        mhsa_a = self.param(
            f"decoder_mhsa_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        q = q + mhsa_a.astype(q.dtype) * mhsa
        q_ln = layer_norm(q)
        kv_ln = layer_norm(kv)
        mhca = MultiHeadAttention(
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
            mask=cross_attn_mask,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )
        mhca_a = self.param(
            f"decoder_mhca_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        q = q + mhca_a.astype(q.dtype) * mhca
        qkv_ln = layer_norm(q)
        ffn = FFWMLP(
            hidden_size=self.resblocks_hidden_size,
            use_bias=self.use_bias,
        )(qkv_ln)
        ffn_a = self.param(
            f"decoder_ffn_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        q = q + ffn_a.astype(q.dtype) * ffn
        return jnp.where(positionwise_mask, q, 0)

    def encoder_layer(
        self,
        layer_idx: int,
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
        mha_a = self.param(
            f"encoder_mha_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        qkv = qkv + mha_a.astype(qkv.dtype) * mha
        qkv_ln = layer_norm(qkv)
        ffn = FFWMLP(
            hidden_size=self.resblocks_hidden_size,
            use_bias=self.use_bias,
        )(qkv_ln)
        ffn_a = self.param(
            f"encoder_ffn_a_{layer_idx}",
            nn.initializers.constant(self.init_residual_scale),
            (1,),
        )
        qkv = qkv + ffn_a.astype(qkv.dtype) * ffn
        return jnp.where(positionwise_mask, qkv, 0)

    @nn.compact
    def __call__(
        self,
        q: jax.Array,
        kv: jax.Array,
        q_self_attn_mask: jax.Array | None = None,
        kv_self_attn_mask: jax.Array | None = None,
        cross_attn_mask: jax.Array | None = None,
    ) -> jax.Array:

        if q_self_attn_mask is None:
            q_mask = jnp.ones_like(q[..., 0], dtype=jnp.bool)
            q_self_attn_mask = create_attention_mask(q_mask, q_mask)

        if kv_self_attn_mask is None:
            kv_mask = jnp.ones_like(kv[..., 0], dtype=jnp.bool)
            kv_self_attn_mask = create_attention_mask(kv_mask, kv_mask)

        if cross_attn_mask is None:
            kv_mask = jnp.ones_like(kv[..., 0], dtype=jnp.bool)
            cross_attn_mask = create_attention_mask(q_mask, kv_mask)

        q_encoder_positionwise_mask = q_self_attn_mask.any(
            axis=-1, keepdims=True
        ).squeeze(-3)
        kv_encoder_positionwise_mask = kv_self_attn_mask.any(
            axis=-1, keepdims=True
        ).squeeze(-3)

        for layer_idx in range(self.num_layers):
            kv = self.encoder_layer(
                layer_idx,
                kv,
                kv_self_attn_mask,
                kv_encoder_positionwise_mask,
            )

        for layer_idx in range(self.num_layers, 2 * self.num_layers):
            q = self.decoder_layer(
                layer_idx,
                q,
                kv,
                q_self_attn_mask,
                cross_attn_mask,
                q_encoder_positionwise_mask,
            )

        q = layer_norm(q)

        return q


class MLP(nn.Module):
    """Apply unit-wise linear layers to the units."""

    layer_sizes: int | tuple[int] | list[int] = None
    use_layer_norm: bool = True
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply unit-wise linear layers to the units.

        Args:
            x (jax.Array): Input array.

        Returns:
            jax.Array: Output array.
        """
        layer_sizes = self.layer_sizes
        if layer_sizes is None:
            layer_sizes = x.shape[-1]

        if isinstance(layer_sizes, int):
            layer_sizes = (layer_sizes,)
        else:
            layer_sizes = self.layer_sizes

        for _, size in enumerate(layer_sizes):
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            x = nn.Dense(size, dtype=x.dtype, use_bias=self.use_bias)(x)
        return x


class FFWMLP(nn.Module):
    """Feed-Forward Network (FFN) MLP module."""

    hidden_size: int
    output_size: int = None
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply Feed-Forward Network (FFN) MLP to the inputs.

        Args:
            x (jax.Array): Input array.

        Returns:
            jax.Array: Output array.
        """
        inp_size = x.shape[-1]

        gating_layer = nn.Dense(
            2 * self.hidden_size, dtype=x.dtype, use_bias=self.use_bias
        )
        output_layer = nn.Dense(
            self.output_size or inp_size, dtype=x.dtype, use_bias=self.use_bias
        )

        gate = gating_layer(x)
        gate_a, gate_b = jnp.split(gate, 2, axis=-1)

        # SwiGLU: Apply Swish (SiLU) to the first half, multiply by the second half
        hidden = jax.nn.silu(gate_a) * gate_b

        return output_layer(hidden)


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


def simple_sum_embeddings(
    *embeddings: list[jax.Array], divisor: Optional[int] = None
) -> jax.Array:
    """
    Get the sum of the embeddings.

    Args:
        embeddings (list[jax.Array]): List of embedding arrays.

    Returns:
        jax.Array: Sum of the embeddings.
    """
    if len(embeddings) == 0:
        raise ValueError("No embeddings provided")
    if divisor is None:
        divisor = math.sqrt(len(embeddings))
    return sum(embeddings) / divisor


class SumEmbeddings(nn.Module):
    output_size: int
    hidden_size: int | None = None
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = True

    @nn.compact
    def __call__(self, *embeddings: list[jax.Array] | tuple[jax.Array]) -> jax.Array:
        """Sum embeddings."""
        num_embeddings = len(embeddings)
        if num_embeddings == 0:
            raise ValueError("No embeddings provided")

        embeddings = [
            nn.Dense(
                self.hidden_size or self.output_size, dtype=self.dtype, use_bias=False
            )(emb)
            for emb in embeddings
        ]

        # Sum and scale the variance by dividing by sqrt(N)
        aggregated = simple_sum_embeddings(*embeddings)

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.output_size,))
            return aggregated + bias.astype(self.dtype)
        else:
            return aggregated


class PointerLogits(nn.Module):
    qk_size: int = None
    num_heads: int = 1
    use_bias: bool = True
    qk_layer_norm: bool = True
    inverse_sqrt_normalisation: bool = True

    @nn.compact
    def __call__(self, q: jax.Array, k: jax.Array) -> jax.Array:
        *q_leading_dims, _ = q.shape
        *kv_leading_dims, _ = k.shape

        qk_size = self.qk_size or k.shape[-1] // self.num_heads

        query_heads = nn.Dense(
            self.num_heads * qk_size, use_bias=self.use_bias, dtype=q.dtype
        )(q).reshape((*q_leading_dims, self.num_heads, qk_size))

        key_heads = nn.Dense(
            self.num_heads * qk_size, use_bias=self.use_bias, dtype=k.dtype
        )(k).reshape((*kv_leading_dims, self.num_heads, qk_size))

        if self.qk_layer_norm:
            query_heads = layer_norm(query_heads)
            key_heads = layer_norm(key_heads)

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...tTh", query_heads, key_heads)

        if self.inverse_sqrt_normalisation:
            attn_logits = attn_logits / math.sqrt(qk_size)

        return attn_logits


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
