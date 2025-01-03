from enum import Enum, auto
from typing import Any, Callable, List, Optional, Sequence

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
    """Cast x if necessary."""
    if x.dtype != dtype:
        return x.astype(dtype)
    else:
        return x


def activation_fn(array: chex.Array):
    return jax.nn.relu(array)


def layer_norm(array: chex.Array):
    return nn.RMSNorm()(array)


def softcap(array: chex.Array, max_value: int = 50):
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
    def __call__(self, inputs: chex.Array) -> Any:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
            inputs: The nd-array to be transformed.
        Returns:
            The transformed input.
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
            (((inputs.ndim - 1,), (0,)), ((), ())),
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
    def __call__(self, x: chex.Array):
        for _ in range(self.num_resblocks):
            x = VectorResblock(
                hidden_size=self.hidden_size, use_layer_norm=self.use_layer_norm
            )(x)
        return x


class Logits(nn.Module):
    """Logits for scalar heads."""

    num_logits: int
    num_linear_layers: int = 3
    use_layer_norm: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for i in range(self.num_linear_layers):
            if i == self.num_linear_layers - 1:
                output_size = self.num_logits
            else:
                output_size = x.shape[-1]

            # Optionally apply LayerNorm
            if self.use_layer_norm:
                x = layer_norm(x)

            # Apply activation and dense layer with custom kernel initializer
            x = activation_fn(x)
            x = nn.Dense(features=output_size, kernel_init=self.kernel_init)(x)
        return x


def get_freqs(seq_len: int, dim: int, base: int = 10000):
    theta = 1 / (base ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(seq_len)

    idx_theta = jnp.einsum("i,j->ij", t, theta)
    idx_theta = jnp.concatenate([idx_theta, idx_theta], axis=1)

    freqs_cos = jnp.cos(idx_theta)
    freqs_sin = jnp.sin(idx_theta)

    return freqs_cos, freqs_sin


def get_rope_embedding(x: chex.Array):
    def negate_half(x: chex.Array):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    seq_len, dim = x.shape
    cos, sin = get_freqs(seq_len, dim)
    neg_half_x = negate_half(x)
    x_rope = (x * cos) + (neg_half_x * sin)
    return x_rope


def l2_normalize(x: chex.Array, epsilon: float = 1e-6):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + epsilon)


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
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [..., T', D_q].
          key: Embeddings sequence used to compute keys; shape [..., T, D_k].
          value: Embeddings sequence used to compute values; shape [..., T, D_v].
          mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """

        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, s1_length, _ = query.shape

        key_size = self.key_size
        value_size = self.value_size or self.key_size
        model_size = self.model_size or self.key_size * self.num_heads

        linear_mod = SNDense if self.use_spectral_linear else nn.Dense

        def _linear_projection(
            x: jax.Array, head_size: int, need_pos: bool = False
        ) -> jax.Array:
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
        attn_logits = softcap(attn_logits)

        attn_logits = attn_logits * jax.lax.rsqrt(
            jnp.array(key_size, dtype=jnp.float32)
        )
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


def create_attention_mask(mask1: jnp.ndarray, mask2: jnp.ndarray = None) -> jnp.ndarray:
    """Create a combined attention mask for cross-attention."""
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
        self, x: chex.Array, mask: chex.Array = None, ca_mask: chex.Array = None
    ):
        if mask is None:
            mask = jnp.ones_like(x[..., 0], dtype=jnp.bool)

        self_attn_mask = create_attention_mask(mask)
        if ca_mask is not None:
            self_attn_mask = self_attn_mask & ca_mask

        def encoder_layer(x: chex.Array):
            x_ln = layer_norm(x)
            mha = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                value_size=self.value_size,
                model_size=self.model_size,
                query_need_pos=self.x_need_pos,
                key_need_pos=self.y_need_pos,
                use_spectral_linear=self.use_spectral_linear,
            )(query=x_ln, key=x_ln, value=x_ln, mask=self_attn_mask)
            x, _ = nn.GRUCell(self.model_size)(x, nn.relu(mha))
            x_ln = layer_norm(x)
            mlp = MLP(
                (self.resblocks_hidden_size, self.model_size),
                use_layer_norm=False,
                activate_first=False,
                use_spectral_linear=self.use_spectral_linear,
            )(x_ln)
            x, _ = nn.GRUCell(self.model_size)(x, nn.relu(mlp))
            return x

        for _ in range(self.num_layers):
            x = encoder_layer(x)
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
        x_mask: chex.Array = None,
        y_mask: chex.Array = None,
        ca_mask: chex.Array = None,
    ):

        if x_mask is None:
            x_mask = jnp.ones_like(x[..., 0], dtype=jnp.bool)

        if y_mask is None:
            y_mask = jnp.ones_like(y[..., 0], dtype=jnp.bool)

        cross_attn_mask = create_attention_mask(x_mask, y_mask)
        if ca_mask is not None:
            cross_attn_mask = cross_attn_mask & ca_mask

        def decoder_layer(x: chex.Array, y: chex.Array):
            x_ln = layer_norm(x)
            ca = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                value_size=self.value_size,
                model_size=self.model_size,
                query_need_pos=self.x_need_pos,
                key_need_pos=self.y_need_pos,
                use_spectral_linear=self.use_spectral_linear,
            )(query=x_ln, key=y, value=y, mask=ca_mask)
            x, _ = nn.GRUCell(self.model_size)(x, nn.relu(ca))
            x_ln = layer_norm(x)
            mlp = MLP(
                (self.resblocks_hidden_size, self.model_size),
                use_layer_norm=False,
                activate_first=False,
                use_spectral_linear=self.use_spectral_linear,
            )(x_ln)
            x, _ = nn.GRUCell(self.model_size)(x, nn.relu(mlp))
            return x

        for _ in range(self.num_layers):
            x = decoder_layer(x, y)
            x = jnp.where(x_mask[..., jnp.newaxis], x, 0)

        return x


class MLP(nn.Module):
    """Apply unit-wise linear layers to the units."""

    layer_sizes: Sequence[int]
    use_layer_norm: bool = True
    activate_first: bool = True
    use_spectral_linear: bool = False

    @nn.compact
    def __call__(self, x: chex.Array):
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
    def __call__(self, query: chex.Array, keys: chex.Array):

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
    def __call__(self, x: chex.Array, mask: chex.Array):
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
    ):
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

    def _encode(self, inputs: Sequence[chex.Array]):
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
        with open(fpath, "rb") as f:
            arr = np.load(f)
        self.embeddings = jnp.array(arr)

    def __call__(self, indices):
        return jnp.take(self.embeddings, indices, axis=0)


class SwiGLU(nn.Module):
    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        feature_dim = x.shape[-1]

        w1 = nn.Dense(feature_dim, use_bias=False)
        w2 = nn.Dense(feature_dim, use_bias=False)
        w3 = nn.Dense(feature_dim, use_bias=False)

        x1 = w1(x)
        x2 = w2(x)
        h = nn.silu(x1) * x2

        return w3(h)


class GRUFeatureCombiner(nn.Module):
    """
    A Flax module that uses GRU-like gating to combine N feature vectors.

    Attributes:
        feature_dim: The target dimension of the combined feature vector.
    """

    feature_dim: int
    hidden_dim: int = None
    num_hidden_dims: int = 1

    @nn.compact
    def __call__(self, features: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Combine feature vectors using GRU-inspired gating.

        Args:
            features: A list of tensors of shape (feature_dim_i,), where feature_dim_i can vary.

        Returns:
            A tensor of shape (feature_dim,), representing the combined feature vector.
        """
        concatenated_features = jnp.concatenate(features, axis=-1)
        hidden_dim = self.hidden_dim or self.feature_dim
        layer_sizes = (hidden_dim,) * self.num_hidden_dims + (self.feature_dim,)

        embedding = nn.Dense(self.feature_dim)(concatenated_features)
        embedding = MLP(layer_sizes)(embedding)
        embedding = SwiGLU()(embedding)

        return layer_norm(embedding)
