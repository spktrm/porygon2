import jax
import chex

import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from ml_collections import ConfigDict
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
    resblocks_num_before: int
    resblocks_num_after: int
    resblocks_hidden_size: Optional[int] = None
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self, x: chex.Array, mask: chex.Array, temporal_mask: chex.Array = None
    ):
        for _ in range(self.resblocks_num_before):
            x = UnitsResblock(
                hidden_size=self.resblocks_hidden_size,
                use_layer_norm=self.use_layer_norm,
            )(x)
        for _ in range(self.transformer_num_layers):
            x1 = x
            if self.use_layer_norm:
                x1 = nn.LayerNorm()(x1)
            x1 = activation_fn(x1)
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
        for _ in range(self.resblocks_num_after):
            x = UnitsResblock(
                hidden_size=self.resblocks_hidden_size,
                use_layer_norm=self.use_layer_norm,
            )(x)
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
        logits = jnp.matmul(keys, query)  # ij,j->i
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


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def discretize(A: chex.Array, B: chex.Array, C: chex.Array, step: int):
    I = np.eye(A.shape[0])
    BL = jnp.linalg.inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def K_conv(Ab: chex.Array, Bb: chex.Array, Cb: chex.Array, L):
    return np.array(
        [(Cb @ jnp.linalg.matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


def causal_convolution(u: chex.Array, K: chex.Array, nofft: bool = False):
    if nofft:
        return jax.scipy.signal.convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


def scan_SSM(
    Ab: chex.Array, Bb: chex.Array, Cb: chex.Array, u: chex.Array, x0: chex.Array
):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


class SSMLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.A = self.param("A", nn.initializers.lecun_normal(), (self.N, self.N))
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)


def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ jnp.linalg.inv(I - jnp.linalg.matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


class S4Layer(nn.Module):
    N: int
    l_max: int

    def setup(self):
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", nn.initializers.normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        self.K = kernel_DPLR(
            self.Lambda,
            self.P,
            self.P,
            self.B,
            self.C,
            self.step,
            self.l_max,
        )

    def __call__(self, u):
        # This is identical to SSM Layer
        return causal_convolution(u, self.K) + self.D * u


class SequenceModel(nn.Module):
    num_layers: int

    @nn.compact
    def __call__(self, seq: chex.Array):
        for i in range(self.num_layers):
            seq = S4Layer(seq.shape[0], seq.shape[0])(seq)
        return seq


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


class PretrainedEmbedding(nn.Module):
    fpath: str

    def setup(self):
        with open(self.fpath, "rb") as f:
            arr = np.load(f)
        self.embeddings = jnp.array(arr)

    def __call__(self, indices):
        return self.embeddings[indices]
