from enum import Enum, auto
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import chex
import flax.linen as nn
import jax
import jax.experimental
import jax.experimental.host_callback
import jax.lax as lax
import jax.nn.initializers as initjax
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
    return jax.nn.leaky_relu(array)


class GatingType(Enum):
    NONE = auto()
    GLOBAL = auto()
    POINTWISE = auto()


class VectorResblock(nn.Module):
    """Fully connected residual block."""

    num_layers: int = 2
    hidden_size: Optional[int] = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        chex.assert_type(x, jnp.float32)
        shortcut = x
        input_size = x.shape[-1]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                output_size = self.hidden_size or input_size
            else:
                output_size = input_size
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = SNDense(features=output_size)(x)
        return x + shortcut


class UnitsResblock(VectorResblock):
    """Fully connected residual block, unit-wise."""

    def __call__(self, x: chex.Array) -> chex.Array:
        chex.assert_type(x, jnp.float32)
        return jax.vmap(super().__call__)(x)


class Resnet(nn.Module):
    """A fully-connected resnet."""

    num_resblocks: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array):
        for _ in range(self.num_resblocks):
            x = VectorResblock(use_layer_norm=self.use_layer_norm)(x)
        return x


class Logits(nn.Module):
    """Logits for scalar heads."""

    num_logits: int
    num_linear_layers: int = 2
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for i in range(self.num_linear_layers):
            if i == self.num_linear_layers - 1:
                output_size = self.num_logits
            else:
                output_size = x.shape[-1]
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = SNDense(features=output_size)(x)
        return x


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
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size)  # [T', H, Q=K]
        key_heads = projection(key, self.key_size)  # [T, H, K]
        value_heads = projection(value, self.value_size or self.key_size)  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        # attn_logits = utils._prenorm_softmax(attn_logits, mask)
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)

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

        jax.debug.print("x: {:.2f}", attn_weights)

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = SNDense(self.model_size or attn.shape[-1])
        return final_projection(attn)  # [T', D']

    def _linear_projection(self, x: jax.Array, head_size: int) -> jax.Array:
        y = SNDense(self.num_heads * head_size)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))


def create_attention_mask(mask1: jnp.ndarray, mask2: jnp.ndarray) -> jnp.ndarray:
    """Create a combined attention mask for cross-attention."""
    if mask1 is not None and mask2 is not None:
        # mask1: (batch_size, seq_len1), mask2: (batch_size, seq_len2)
        # We want a (batch_size, 1, seq_len1, seq_len2) mask for broadcasting.
        mask1 = mask1[..., None]  # Shape: (batch_size, seq_len1, 1)
        mask2 = mask2[..., None, :]  # Shape: (batch_size, 1, seq_len2)
        # Combine them with logical AND (only allow attending if both elements are not masked).
        combined_mask = mask1 & mask2  # Shape: (batch_size, seq_len1, seq_len2)
        combined_mask = combined_mask[
            ..., None, :, :
        ]  # Shape: (batch_size, 1, seq_len1, seq_len2)
    else:
        combined_mask = None  # No mask if both are None
    return combined_mask


class Transformer(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    stream_size: int
    num_layers: int
    num_heads: int
    key_size: int
    value_size: int
    resblocks_hidden_size: Optional[int] = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        q: chex.Array,
        kv: chex.Array = None,
        q_mask: chex.Array = None,
        kv_mask: chex.Array = None,
    ):

        if q_mask is None:
            q_mask = jnp.ones_like(q[..., 0], dtype=jnp.bool)

        if kv_mask is None:
            kv_mask = jnp.ones_like(q[..., 0], dtype=jnp.bool)

        logits_mask = create_attention_mask(q_mask, kv_mask)

        if kv is None:
            kv = q
        if self.use_layer_norm:
            kv = nn.LayerNorm()(kv)
        kv = activation_fn(kv)

        for _ in range(self.num_layers):
            # Apply pre-layer normalization before attention
            if self.use_layer_norm:
                q = nn.LayerNorm()(q)
            q = activation_fn(q)
            attn_output = MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                value_size=self.value_size,
                model_size=self.stream_size,
            )(query=q, key=kv, value=kv, mask=logits_mask)
            # attn_output = jnp.where(q_mask[..., jnp.newaxis], attn_output, 0)
            q = q + attn_output
            # Apply pre-layer normalization before the residual block
            if self.use_layer_norm:
                q = nn.LayerNorm()(q)
            dense_output = MLP(
                (self.resblocks_hidden_size, self.stream_size), use_layer_norm=False
            )(q)
            q = q + dense_output
            q = jnp.where(q_mask[..., jnp.newaxis], q, 0)

        return q


class MLP(nn.Module):
    """Apply unit-wise linear layers to the units."""

    layer_sizes: Sequence[int]
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array):
        for size in self.layer_sizes:
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = SNDense(size)(x)
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
                query = nn.LayerNorm()(query)
            query = activation_fn(query)
            if i == self.num_layers_query - 1:
                query = SNDense(features=self.key_size)(query)
            else:
                query = SNDense(features=query.shape[-1])(query)

        # Keys.
        for i in range(self.num_layers_keys):
            if self.use_layer_norm:
                keys = nn.LayerNorm()(keys)
            keys = activation_fn(keys)
            if i == self.num_layers_keys - 1:
                keys = SNDense(features=self.key_size)(keys)
            else:
                keys = SNDense(features=keys.shape[-1])(keys)

        # Pointer
        logits = query @ keys.T  # ij,j->i
        return logits


class ToAvgVector(nn.Module):
    """Per-unit processing then average over the units dimension."""

    units_hidden_sizes: Sequence[int]
    output_stream_size: int = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array = None):
        for size in self.units_hidden_sizes:
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = SNDense(features=size)(x)
        if mask is None:
            x = x.mean(-2)
        else:
            x = (mask.astype(jnp.float32) @ x) / mask.sum()
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = activation_fn(x)
        x = SNDense(features=self.output_stream_size or x.shape[-1])(x)
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
        if self.gating_type is GatingType.GLOBAL:
            gate_size = 1
        elif self.gating_type is GatingType.POINTWISE:
            gate_size = self.output_size
        else:
            raise ValueError(f"Gating type {self.gating_type} is not supported")
        if len(inputs_to_gate) == 2:
            # more efficient than the general version below
            gate = [SNDense(gate_size)(y) for y in init_gate]
            gate = jnp.stack(gate).sum(0)
            gate = jax.nn.sigmoid(gate)
        else:
            gate = [SNDense(len(inputs_to_gate) * gate_size)(y) for y in init_gate]
            gate = jnp.stack(gate).sum(0)
            gate = jnp.reshape(gate, [len(inputs_to_gate), gate_size])
            # gate = _prenorm_softmax(gate, axis=0)
            gate = jax.nn.softmax(gate, axis=0)
        return gate

    def _encode(self, inputs: List[chex.Array]):
        gate, outputs = [], []
        for feature in inputs:
            feature = astype(feature, jnp.float32)
            if self.use_layer_norm:
                feature = nn.LayerNorm()(feature)
            feature = activation_fn(feature)
            gate.append(feature)
            outputs.append(SNDense(self.output_size)(feature))
        return gate, outputs

    @nn.compact
    def __call__(self, *inputs: List[chex.Array]):
        gate, outputs = self._encode(inputs)
        if len(outputs) == 1:
            # Special case of 1-D inputs that do not need any gating.
            output = outputs[0]
        elif self.gating_type is GatingType.NONE:
            output = jnp.stack(outputs).sum(0)
        else:
            gate = self._compute_gate(outputs, gate)
            data = gate * jnp.stack(outputs)
            output = data.sum(0)
        return output


class GLU(nn.Module):
    """Gated Linear Unit.

    Helper Modules:
    Gating Linear Unit (GLU):
        Inputs: input, context, output_size

    # The gate value is a learnt function of the input.
    gate = sigmoid(linear(input.size)(context))

    # Gate the input and return an output of desired size.
    gated_input = gate * input
    output = linear(output_size)(gated_input)

    return output
    """

    output_size: int = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, input: chex.Array, context: chex.Array):
        if self.use_layer_norm:
            context = nn.LayerNorm()(context)
        context = activation_fn(context)
        gate = jax.nn.sigmoid(SNDense(input.shape[-1])(context))
        gated_input = gate * input
        if self.use_layer_norm:
            gated_input = nn.LayerNorm()(gated_input)
        gated_input = activation_fn(gated_input)
        output = SNDense(self.output_size or input.shape[-1])(gated_input)
        return output


class AutoEncoder(nn.Module):
    output_size: int
    latent_dim: int
    hidden_sizes: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        self.encoder = MLP(self.hidden_sizes, self.use_layer_norm)
        self.mean = MLP((self.latent_dim,), self.use_layer_norm)
        self.log_stddev = MLP((self.latent_dim,), self.use_layer_norm)
        self.decoder = MLP((self.output_size,), self.use_layer_norm)

    def __call__(self, x: chex.Array, key):
        h = self.encoder(x)
        mean = self.mean(h)
        log_stddev = self.log_stddev(h)
        stddev = jnp.exp(log_stddev)
        z = mean + stddev * jax.random.normal(key, mean.shape)
        return self.decoder(z)

    def encode(self, x: chex.Array):
        h = self.encoder(x)
        return self.mean(h)


class PretrainedEmbedding:
    def __init__(self, fpath: str):
        with open(fpath, "rb") as f:
            arr = np.load(f)
        self.embeddings = jnp.array(arr)

    def __call__(self, indices):
        return jnp.take(self.embeddings, indices, axis=0)


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]


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
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initjax.zeros
    std_init: float = 0.1
    denom_backward: bool = True

    @nn.compact
    def __call__(self, inputs: Any) -> Any:
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
        sigma_param = self.param("sigma", initjax.ones, (1,), self.param_dtype)
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
