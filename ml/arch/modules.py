import math
from enum import Enum, auto
from typing import Any, List, Optional, Sequence, Tuple, Union

import chex
import flax.linen as nn
import jax
import jax.experimental
import jax.experimental.host_callback
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from flax.linen.dtypes import promote_dtype
from flax.typing import Array, Dtype, Initializer, PrecisionLike
from flax.typing import PRNGKey as PRNGKey
from flax.typing import Shape as Shape

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


class Linear(nn.Dense):
    kernel_init: Initializer = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
        stddev = 1.0 / np.sqrt(inputs.shape[-1])
        kernel = self.param(
            "kernel",
            self.kernel_init or nn.initializers.truncated_normal(stddev=stddev),
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = lax.dot_general
        y = dot_general(
            inputs,
            kernel,
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
                kernel_init = None
            else:
                output_size = input_size
                kernel_init = nn.initializers.normal(stddev=0.005)
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            x = Linear(features=output_size, kernel_init=kernel_init)(x)
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
            x = Linear(features=output_size, kernel_init=self.kernel_init)(x)
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

        def _linear_projection(x: jax.Array, head_size: int) -> jax.Array:
            y = Linear(self.num_heads * head_size)(x)
            *leading_dims, _ = x.shape
            return y.reshape((*leading_dims, self.num_heads, head_size))

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = _linear_projection(query, self.key_size)  # [T', H, Q=K]
        key_heads = _linear_projection(key, self.key_size)  # [T, H, K]
        value_heads = _linear_projection(
            value, self.value_size or self.key_size
        )  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        # attn_logits = utils._prenorm_softmax(attn_logits, mask)
        attn_logits = attn_logits / jnp.sqrt(self.key_size).astype(key.dtype)

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
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = Linear(self.model_size or attn.shape[-1])
        return final_projection(attn)  # [T', D']


class MultiHeadDiffAttention(nn.Module):
    """
    Multi-head differential attention module with the core idea of subtracting the softmax.
    This module is modeled similar to the default Flax MultiHeadDotProductAttention module.
    """

    num_heads: int
    key_size: int
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

        def _query_key_projection(x: jax.Array, head_size: int) -> jax.Array:
            y = Linear(self.num_heads * 2 * head_size)(x)
            *leading_dims, _ = x.shape
            return y.reshape((*leading_dims, 2 * self.num_heads, head_size))

        def _value_projection(x: jax.Array, head_size: int) -> jax.Array:
            y = Linear(self.num_heads * 2 * head_size)(x)
            *leading_dims, _ = x.shape
            return y.reshape((*leading_dims, self.num_heads, 2 * head_size))

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = _query_key_projection(query, self.key_size)  # [T', H, Q=K]
        key_heads = _query_key_projection(key, self.key_size)  # [T, H, K]
        value_heads = _value_projection(value, self.key_size)  # [T, H, V]

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

        lambda_q1 = self.param(
            "lambda_q1", nn.initializers.normal(stddev=0.1), (self.key_size,)
        )
        lambda_k1 = self.param(
            "lambda_k1", nn.initializers.normal(stddev=0.1), (self.key_size,)
        )
        lambda_q2 = self.param(
            "lambda_q2", nn.initializers.normal(stddev=0.1), (self.key_size,)
        )
        lambda_k2 = self.param(
            "lambda_k2", nn.initializers.normal(stddev=0.1), (self.key_size,)
        )

        attn_weights = attn_weights.reshape(
            *leading_dims, self.num_heads, 2, sequence_length, -1
        )

        lambda_1 = jnp.exp(jnp.sum(lambda_q1 * lambda_k1, axis=-1))
        lambda_2 = jnp.exp(jnp.sum(lambda_q2 * lambda_k2, axis=-1))

        def lambda_init_fn(depth):
            return 0.8 - 0.6 * math.exp(-0.3 * depth)

        lambda_init = lambda_init_fn(2)
        lambda_full = lambda_1 - lambda_2 + lambda_init
        attn_weights = (
            attn_weights[..., 0, :, :] - lambda_full * attn_weights[..., 1, :, :]
        )

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]
        attn = layer_norm(attn)
        attn = attn * (1 - lambda_init)

        # Apply another projection to get the final embeddings.
        final_projection = Linear(self.model_size or attn.shape[-1])
        return final_projection(attn)  # [T', D']


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

    hidden_size: int
    num_layers: int
    num_heads: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array = None):

        if mask is None:
            mask = jnp.ones_like(x[..., 0], dtype=jnp.bool)

        logits_mask = create_attention_mask(mask, mask)

        for _ in range(self.num_layers):
            # Apply pre-layer normalization before attention
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            attn_output = MultiHeadDiffAttention(
                num_heads=self.num_heads,
                key_size=self.hidden_size // 2,
                model_size=self.hidden_size,
            )(query=x, key=x, value=x, mask=logits_mask)
            attn_output = jnp.where(mask[..., jnp.newaxis], attn_output, 0)
            x = x + attn_output
            # Apply pre-layer normalization before the residual block
            x = UnitsResblock(
                hidden_size=self.hidden_size // 2, use_layer_norm=self.use_layer_norm
            )(x)
            x = jnp.where(mask[..., jnp.newaxis], x, 0)

        return x


class CrossAttentionLayer(nn.Module):
    hidden_size: int
    num_heads: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        query_seq: chex.Array,
        key_value_seq: chex.Array,
        attention_mask: chex.Array = None,
    ):
        """
        Cross-attention layer where the query sequence attends to the key-value sequence.

        Args:
            query_seq: Array of shape (query_length, hidden_size)
            key_value_seq: Array of shape (key_value_length, hidden_size)
            query_mask: Optional boolean array of shape (query_length,)
            key_value_mask: Optional boolean array of shape (key_value_length,)

        Returns:
            attention_output: Array of shape (query_length, hidden_size)
        """

        attention_output = MultiHeadDiffAttention(
            num_heads=self.num_heads,
            key_size=self.hidden_size // 2,
            model_size=self.hidden_size,
        )(
            query=query_seq,
            key=key_value_seq,
            value=key_value_seq,
            mask=attention_mask,
        )

        return attention_output  # Shape: (query_length, hidden_size)


class CrossTransformer(nn.Module):
    num_layers: int
    hidden_size: int
    num_heads: int
    dtype: Any = jnp.float32
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        query_seq: chex.Array,
        key_value_seq: chex.Array,
        query_mask: chex.Array = None,
        key_value_mask: chex.Array = None,
    ):
        # Prepare masks
        if query_mask is not None:
            # Convert to float mask and add batch dimension
            query_mask = query_mask.astype(self.dtype)
        else:
            *_, query_length, __ = query_seq.shape
            query_mask = jnp.ones((1, query_length), dtype=self.dtype)

        if key_value_mask is not None:
            key_value_mask = key_value_mask.astype(self.dtype)
        else:
            *_, key_length, __ = key_value_seq.shape
            key_value_mask = jnp.ones((1, key_length), dtype=self.dtype)

        # Create attention bias
        attention_mask = self.make_attention_mask(query_mask, key_value_mask)
        transposed_attention_mask = jnp.swapaxes(attention_mask, -1, -2)

        expanded_query_mask = jnp.expand_dims(query_mask, axis=-1)
        expanded_key_value_mask = jnp.expand_dims(key_value_mask, axis=-1)

        for _ in range(self.num_layers):
            if self.use_layer_norm:
                query_seq = layer_norm(query_seq)
                key_value_seq = layer_norm(key_value_seq)

            query_seq = activation_fn(query_seq)
            key_value_seq = activation_fn(key_value_seq)

            contextualized_query = CrossAttentionLayer(
                hidden_size=self.hidden_size, num_heads=self.num_heads
            )(query_seq, key_value_seq, attention_mask)
            contextualized_key_value = CrossAttentionLayer(
                hidden_size=self.hidden_size, num_heads=self.num_heads
            )(key_value_seq, query_seq, transposed_attention_mask)

            contextualized_query = jnp.where(
                expanded_query_mask, contextualized_query, 0
            )
            contextualized_key_value = jnp.where(
                expanded_key_value_mask, contextualized_key_value, 0
            )

            query_seq = query_seq + contextualized_query
            key_value_seq = key_value_seq + contextualized_key_value

            query_seq = UnitsResblock(
                hidden_size=self.hidden_size // 2, use_layer_norm=self.use_layer_norm
            )(query_seq)
            key_value_seq = UnitsResblock(
                hidden_size=self.hidden_size // 2, use_layer_norm=self.use_layer_norm
            )(key_value_seq)

            query_seq = jnp.where(expanded_query_mask, query_seq, 0)
            key_value_seq = jnp.where(expanded_key_value_mask, key_value_seq, 0)

        return query_seq, key_value_seq

    def make_attention_mask(
        self, query_mask: chex.Array, key_value_mask: chex.Array
    ) -> chex.Array:
        """Creates an additive attention bias mask.

        Args:
            query_mask: (batch_size, query_length)
            key_value_mask: (batch_size, key_length)

        Returns:
            attention_bias: (batch_size, 1, query_length, key_length)
        """

        attention_mask = jnp.einsum("...i,...j->...ij", query_mask, key_value_mask)
        attention_mask = jnp.expand_dims(attention_mask, axis=-3)
        return attention_mask.astype(self.dtype)


class MLP(nn.Module):
    """Apply unit-wise linear layers to the units."""

    layer_sizes: Sequence[int]
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array):
        for size in self.layer_sizes:
            if self.use_layer_norm:
                x = layer_norm(x)
            x = activation_fn(x)
            x = Linear(size)(x)
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
                query = Linear(features=self.key_size)(query)
            else:
                query = Linear(features=query.shape[-1])(query)

        # Keys.
        for i in range(self.num_layers_keys):
            if self.use_layer_norm:
                keys = layer_norm(keys)
            keys = activation_fn(keys)
            if i == self.num_layers_keys - 1:
                keys = Linear(features=self.key_size)(keys)
            else:
                keys = Linear(features=keys.shape[-1])(keys)

        # Pointer
        logits = query @ keys.T  # ij,j->i
        return logits


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
            x = Linear(features=size)(x)

        if self.pool_method is PoolMethod.MEAN:
            x = (mask.astype(jnp.float32) @ x) / mask.sum().clip(min=1)
        elif self.pool_method is PoolMethod.MAX:
            x = jnp.where(jnp.expand_dims(mask, axis=-1), x, -1e9).max(0)
        else:
            raise ValueError(f"Pool method {self.pool_method} is not supported")

        if self.use_layer_norm:
            x = layer_norm(x)
        x = activation_fn(x)
        x = Linear(features=self.output_stream_size or x.shape[-1])(x)
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
                Linear(gate_size, kernel_init=w_init, bias_init=b_init)(y)
                for y in init_gate
            ]
            gate = sum(gate)
            sigmoid = jax.nn.sigmoid(gate)
            gate = [sigmoid, 1.0 - sigmoid]
        else:
            gate = [
                Linear(
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
            outputs.append(Linear(self.output_size)(feature))
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
            context = layer_norm(context)
        context = activation_fn(context)
        gate = jax.nn.sigmoid(Linear(input.shape[-1])(context))
        gated_input = gate * input
        if self.use_layer_norm:
            gated_input = layer_norm(gated_input)
        gated_input = activation_fn(gated_input)
        output = Linear(self.output_size or input.shape[-1])(gated_input)
        return output


class SSLReg(nn.Module):
    @nn.compact
    def __call__(self, st: chex.Array, stp1: chex.Array) -> chex.Array:

        latent_action = VectorMerge(st.shape[-1])(st, stp1)
        latent_action = MLP((st.shape[-1] // 4, st.shape[-1] // 8))(latent_action)
        pred_stpd1 = VectorMerge(stp1.shape[-1])(st, latent_action)

        return pred_stpd1


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
