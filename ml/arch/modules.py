import jax
import chex

import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from enum import Enum, auto
from typing import List, Optional, Sequence


def astype(x: chex.Array, dtype: jnp.dtype) -> chex.Array:
    """Cast x if necessary."""
    if x.dtype != dtype:
        return x.astype(dtype)
    else:
        return x


def activation_fn(array: chex.Array):
    return jax.nn.relu(array)


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
                kwargs = {}
            else:
                output_size = input_size
                kwargs = dict(
                    kernel_init=jax.nn.initializers.normal(stddev=0.005),
                    bias_init=jax.nn.initializers.constant(0.0),
                )
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = nn.Dense(features=output_size, **kwargs)(x)
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
            x = nn.Dense(features=output_size)(x)
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

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = nn.Dense(self.model_size or attn.shape[-1])
        return final_projection(attn)  # [T', D']

    def _linear_projection(self, x: jax.Array, head_size: int) -> jax.Array:
        y = nn.Dense(self.num_heads * head_size)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))


class Transformer(nn.Module):
    """Apply unit-wise resblocks, and transformer layers, to the units."""

    units_stream_size: int
    transformer_num_layers: int
    transformer_num_heads: int
    transformer_key_size: int
    transformer_value_size: int
    resblocks_hidden_size: Optional[int] = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self, x: chex.Array, mask: chex.Array, temporal_mask: chex.Array = None
    ):
        for _ in range(self.transformer_num_layers):
            x1 = x
            # The logits mask has shape [num_heads, num_units, num_units]:
            logits_mask = mask[jnp.newaxis, jnp.newaxis]
            if temporal_mask is not None:
                logits_mask = logits_mask * temporal_mask
            x1 = MultiHeadAttention(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                value_size=self.transformer_value_size,
                model_size=self.units_stream_size,
            )(query=x1, key=x1, value=x1, mask=logits_mask)
            # Mask here mostly for safety:
            x1 = jnp.where(mask[:, jnp.newaxis], x1, 0)
            x = x + x1
            x = UnitsResblock(
                hidden_size=self.resblocks_hidden_size,
                use_layer_norm=self.use_layer_norm,
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
        x = jnp.where(mask[:, jnp.newaxis], x, 0)
        return x


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
            x = nn.Dense(size)(x)
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
                query = nn.Dense(features=self.key_size)(query)
            else:
                query = nn.Dense(features=query.shape[-1])(query)

        # Keys.
        for i in range(self.num_layers_keys):
            if self.use_layer_norm:
                keys = nn.LayerNorm()(keys)
            keys = activation_fn(keys)
            if i == self.num_layers_keys - 1:
                keys = nn.Dense(features=self.key_size)(keys)
            else:
                keys = nn.Dense(features=keys.shape[-1])(keys)

        # Pointer
        logits = query @ keys.T  # ij,j->i
        return logits


class CNNEncoder(nn.Module):

    hidden_sizes: Sequence[int]
    kernel_size: Sequence[int] = (3,)
    output_size: int = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, valid_mask: jnp.ndarray):
        x = x * valid_mask[..., None]
        for size in self.hidden_sizes:
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = nn.Conv(
                features=size,
                kernel_size=self.kernel_size,
                strides=(1,),
                padding="VALID",
            )(x)
        x = x.reshape(-1)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = activation_fn(x)
        x = nn.Dense(features=self.output_size or x.shape[-1])(x)
        return x


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
            x = nn.Dense(features=size)(x)
        if mask is None:
            x = x.mean(-2)
        else:
            x = (mask.astype(jnp.float32) @ x) / mask.sum()
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = activation_fn(x)
        x = nn.Dense(features=self.output_stream_size or x.shape[-1])(x)
        return x


class ToScatter(nn.Module):
    units_stream_size: int
    units_hidden_sizes: Sequence[int]
    vector_stream_size: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array):
        for size in self.units_hidden_sizes:
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = nn.Dense(features=size)(x)

        mask_onehot = jax.nn.one_hot(mask, 6)
        mask_onehot = jnp.concatenate((mask_onehot, mask_onehot[..., -2:]), axis=-1)
        grouped = mask_onehot.swapaxes(-2, -1) @ x
        x = grouped.flatten()

        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = activation_fn(x)
        x = nn.Dense(features=self.vector_stream_size)(x)
        return x


class SumMergeLax(nn.Module):
    """Merge streams using iterative sum with LAX scan for efficiency.

    This class is designed to sum inputs (streams) more efficiently by leveraging JAX's LAX scan,
    suitable for large-scale operations. Inputs should have matching dimensions.
    Compatible with any stream type (vector, units, or visual).
    """

    @nn.compact
    def __call__(self, *inputs: Sequence[jnp.ndarray]) -> jnp.ndarray:
        def sum_scan(carry, x):
            return carry + x, None

        # Initialize carry with the first input, rest of the inputs are processed
        initial_carry = inputs[0]
        inputs_rest = jnp.stack(
            inputs[1:]
        )  # Stack the rest of inputs for proper scanning

        # Using lax.scan for efficient summation
        summed_output, _ = jax.lax.scan(sum_scan, initial_carry, inputs_rest)
        return summed_output


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
        kernel_init = jax.nn.initializers.lecun_normal()
        bias_init = jax.nn.initializers.constant(0.0)

        if self.gating_type is GatingType.GLOBAL:
            gate_size = 1
        elif self.gating_type is GatingType.POINTWISE:
            gate_size = self.output_size
        else:
            raise ValueError(f"Gating type {self.gating_type} is not supported")
        if len(inputs_to_gate) == 2:
            # more efficient than the general version below
            gate = [
                nn.Dense(gate_size, kernel_init=kernel_init, bias_init=bias_init)(y)
                for y in init_gate
            ]
            gate = jnp.stack(gate).sum(0)
            gate = jax.nn.sigmoid(gate)
        else:
            gate = [
                nn.Dense(
                    len(inputs_to_gate) * gate_size,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                )(y)
                for y in init_gate
            ]
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
            outputs.append(nn.Dense(self.output_size)(feature))
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
        gate = jax.nn.sigmoid(nn.Dense(input.shape[-1])(context))
        gated_input = gate * input
        if self.use_layer_norm:
            gated_input = nn.LayerNorm()(gated_input)
        gated_input = activation_fn(gated_input)
        output = nn.Dense(self.output_size or input.shape[-1])(gated_input)
        return output


class ToVisualScatter(nn.Module):
    """Scatter the units into their positions in the visual stream.

    This means that each element of the units stream will be embedded and placed
    in the visual stream, at the location corresponding to its (x, y) coordinate
    in the world map.
    """

    units_hidden_sizes: Sequence[int]
    kernel_size: int
    output_spatial_size_x: int
    output_spatial_size_y: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        z: chex.Array,
        unit_x: chex.Array,
        unit_y: chex.Array,
        non_empty_units: chex.Array,
    ) -> chex.Array:
        non_empty_units = non_empty_units[:, jnp.newaxis]

        for size in self.units_hidden_sizes:
            if self.use_layer_norm:
                z = nn.LayerNorm()(z)
            z = activation_fn(z)
            z = nn.Dense(features=size)(z)
        z = jnp.where(non_empty_units, z, 0)

        one_hot_x = jax.nn.one_hot(unit_x, self.output_spatial_size_x)
        one_hot_y = jax.nn.one_hot(unit_y, self.output_spatial_size_y)
        z = jnp.einsum("uy,uf->uyf", one_hot_y, z)
        z = jnp.einsum("ux,uyf->yxf", one_hot_x, z)

        if self.use_layer_norm:
            z = nn.LayerNorm()(z)
        z = activation_fn(z)
        z = nn.Conv(
            features=z.shape[-1],
            kernel_size=(self.kernel_size, self.kernel_size),
        )(z)
        return z


class Downscale(nn.Module):
    """Downscale the visual stream.."""

    output_features_size: int
    downscale_factor: int
    kernel_size: int
    ndims: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = activation_fn(x)
        x = nn.Conv(
            features=self.output_features_size,
            kernel_size=(self.kernel_size,) * self.ndims,
            strides=(self.downscale_factor,) * self.ndims,
        )(x)
        return x


class VisualResblock(nn.Module):
    """Convolutional (2d) residual block."""

    kernel_size: int
    ndims: int
    num_layers: int = 2
    hidden_size: Optional[int] = None
    use_layer_norm: bool = True

    def __call__(self, x: chex.Array) -> chex.Array:
        chex.assert_rank(x, self.ndims + 1)
        chex.assert_type(x, jnp.float32)
        shortcut = x
        input_size = x.shape[-1]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                output_size = self.hidden_size or input_size
                kwargs = {}
            else:
                output_size = input_size
                kwargs = dict(
                    kernel_init=jax.nn.initializers.normal(stddev=0.005),
                    bias_init=jax.nn.initializers.constant(0.0),
                )
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = activation_fn(x)
            x = nn.Conv(
                features=output_size,
                kernel_size=(self.kernel_size,) * self.ndims,
                **kwargs,
            )(x)
        return x + shortcut


class VisualResnet(nn.Module):
    """Resnet processing of the visual stream."""

    num_resblocks: int
    ndims: int
    kernel_size: int = 3
    use_layer_norm: bool = True
    num_hidden_feature_planes: Optional[int] = None

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for _ in range(self.num_resblocks):
            x = VisualResblock(
                kernel_size=self.kernel_size,
                ndims=self.ndims,
                hidden_size=self.num_hidden_feature_planes,
                use_layer_norm=self.use_layer_norm,
            )(x)
        return x


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
        return self.embeddings[indices]


class DoubleConv(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.filters, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, (3, 3), padding="SAME")(x)
        return nn.relu(x)


class Down(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x):
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        return DoubleConv(self.filters)(x)


class Up(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x, skip):
        x = jax.image.resize(
            x, (x.shape[0] * 2, x.shape[1] * 2, x.shape[2]), method="nearest"
        )
        x = jnp.concatenate([x, skip], axis=-1)
        return DoubleConv(self.filters)(x)


class UNet(nn.Module):
    num_classes: int
    scale: int = 1

    @nn.compact
    def __call__(self, x):
        # Encoder
        c1 = DoubleConv(int(8 * self.scale))(x)
        c2 = Down(int(16 * self.scale))(c1)
        c3 = Down(int(32 * self.scale))(c2)
        c4 = Down(int(64 * self.scale))(c3)

        # Bridge
        c5 = Down(int(128 * self.scale))(c4)

        # Decoder
        u6 = Up(int(64 * self.scale))(c5, c4)
        u7 = Up(int(32 * self.scale))(u6, c3)
        u8 = Up(int(16 * self.scale))(u7, c2)
        u9 = Up(int(8 * self.scale))(u8, c1)

        return nn.Conv(self.num_classes, (1, 1))(u9)


class CNN(nn.Module):
    output_size: int
    scale: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=int(8 * self.scale), kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=int(16 * self.scale), kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=int(32 * self.scale), kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=int(64 * self.scale), kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(-1)  # flatten
        x = nn.Dense(features=self.output_size)(x)
        return x
