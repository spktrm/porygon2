"""The nGPT normalised residual behind cfg.normalised_residual: off is today's
trunk bit for bit; on keeps every row on the RMS-1 sphere with a live alpha,
leaks nothing across the read mask, and the kernel projection puts every
embedding-space vector at unit L2 and touches nothing else."""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.model.constants import (
    NUM_SEQUENCE_ROWS,
    OPP_PRIVATE_ROWS,
    PRIVILEGED_REGISTER_ROWS,
    PUBLIC_CLS_ROW,
    SEQUENCE_READ_MASK,
    VALUE_CLS_ROW,
)
from rl.model.trunk import Trunk, project_trunk_kernels, unit_rms
from rl.online.training.telemetry import trunk_alpha_telemetry

READ_MASK = jnp.asarray(SEQUENCE_READ_MASK)
WIDTH = 32
NUM_BLOCKS = 3
INVALID_ROWS = (5, 40, 70)
ALPHA_NAMES = ("attention_alpha", "ffw_alpha")

_POLICY_READABLE = np.array(
    [
        row not in range(OPP_PRIVATE_ROWS.start, OPP_PRIVATE_ROWS.stop)
        and row
        not in range(PRIVILEGED_REGISTER_ROWS.start, PRIVILEGED_REGISTER_ROWS.stop)
        and row not in (VALUE_CLS_ROW, PUBLIC_CLS_ROW)
        for row in range(NUM_SEQUENCE_ROWS)
    ]
)


def _trunk_cfg(normalised_residual: bool | None) -> ConfigDict:
    cfg = ConfigDict()
    cfg.num_blocks = NUM_BLOCKS
    cfg.num_heads = 2
    cfg.qk_size = 16
    cfg.v_size = 16
    cfg.model_size = WIDTH
    cfg.hidden_size = 2 * WIDTH
    cfg.qk_layer_norm = True
    cfg.use_bias = True
    if normalised_residual is not None:
        cfg.normalised_residual = normalised_residual
    return cfg


def _inputs():
    sequence = jax.random.normal(jax.random.key(0), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool).at[np.array(INVALID_ROWS)].set(False)
    return sequence, valid


def _forward(trunk: Trunk, params, sequence, valid) -> np.ndarray:
    output = jax.jit(trunk.apply)(params, sequence, valid, READ_MASK)
    return np.asarray(output, dtype=np.float32)


def _init(trunk: Trunk, sequence, valid):
    return jax.jit(trunk.init)(jax.random.key(1), sequence, valid, READ_MASK)


def _with_alphas(params, value: float):
    def fill(path, leaf):
        if getattr(path[-1], "key", None) in ALPHA_NAMES:
            return jnp.full_like(leaf, value)
        return leaf

    return jax.tree_util.tree_map_with_path(fill, params)


def _without_alphas(params):
    blocks = {
        name: leaf
        for name, leaf in params["params"]["blocks"].items()
        if name not in ALPHA_NAMES
    }
    return {"params": {"blocks": blocks}}


def test_flag_off_is_bit_identical_to_the_trunk_without_the_key() -> None:
    sequence, valid = _inputs()
    absent = Trunk(_trunk_cfg(None))
    off = Trunk(_trunk_cfg(False))
    params_absent = _init(absent, sequence, valid)
    params_off = _init(off, sequence, valid)
    jax.tree.map(np.testing.assert_array_equal, params_absent, params_off)
    np.testing.assert_array_equal(
        _forward(absent, params_absent, sequence, valid),
        _forward(off, params_off, sequence, valid),
    )
    jaxpr_absent = str(
        jax.make_jaxpr(absent.apply)(params_absent, sequence, valid, READ_MASK)
    )
    jaxpr_off = str(jax.make_jaxpr(off.apply)(params_off, sequence, valid, READ_MASK))
    assert jaxpr_absent == jaxpr_off

    # Positive control: the flag on adds exactly the alpha leaves, at the
    # nGPT init, leaves every other leaf as the off init has it, and
    # changes the output.
    on = Trunk(_trunk_cfg(True))
    params_on = _init(on, sequence, valid)
    for name in ALPHA_NAMES:
        alpha = np.asarray(params_on["params"]["blocks"][name])
        assert alpha.shape == (NUM_BLOCKS, WIDTH)
        np.testing.assert_array_equal(
            alpha, np.full((NUM_BLOCKS, WIDTH), 0.05, np.float32)
        )
    jax.tree.map(np.testing.assert_array_equal, _without_alphas(params_on), params_off)
    assert not np.allclose(
        _forward(on, params_on, sequence, valid),
        _forward(off, params_off, sequence, valid),
    )


def test_flag_on_keeps_every_valid_row_at_unit_rms() -> None:
    sequence, valid = _inputs()
    on = Trunk(_trunk_cfg(True))
    output = _forward(on, _init(on, sequence, valid), sequence, valid)
    rms = np.sqrt(np.mean(np.square(output), axis=-1))
    np.testing.assert_allclose(rms[np.asarray(valid)], 1.0, atol=1e-4)
    np.testing.assert_array_equal(output[list(INVALID_ROWS)], 0.0)


def test_alpha_is_the_step_size() -> None:
    sequence, valid = _inputs()
    on = Trunk(_trunk_cfg(True))
    params_on = _init(on, sequence, valid)

    # alpha = 0: every sub-layer step is the identity up to the norm, so the
    # trunk is unit_rms applied once per sub-layer, invalid rows zeroed.
    reference = sequence
    for _ in range(2 * NUM_BLOCKS):
        reference = unit_rms(reference)
    reference = np.asarray(jnp.where(valid[:, None], reference, 0), np.float32)
    np.testing.assert_allclose(
        _forward(on, _with_alphas(params_on, 0.0), sequence, valid),
        reference,
        atol=1e-6,
    )

    # alpha = 1: the step replaces the row by its normalised sub-layer
    # output, and the pre-norms make that output blind to a row's
    # magnitude, so doubling a row's input leaves the output unchanged...
    doubled = sequence.at[3].multiply(2.0)
    params_one = _with_alphas(params_on, 1.0)
    np.testing.assert_allclose(
        _forward(on, params_one, sequence, valid),
        _forward(on, params_one, doubled, valid),
        atol=1e-5,
    )
    # ...whereas at the nGPT init the row's own direction still carries
    # through the residual term, so the same doubling moves the output --
    # the positive control that alpha is read.
    assert not np.allclose(
        _forward(on, params_on, sequence, valid),
        _forward(on, params_on, doubled, valid),
        atol=1e-5,
    )


def test_secret_rows_stay_invisible_under_the_normalised_residual() -> None:
    sequence = jax.random.normal(jax.random.key(0), (NUM_SEQUENCE_ROWS, WIDTH))
    valid = jnp.ones(NUM_SEQUENCE_ROWS, bool)
    on = Trunk(_trunk_cfg(True))
    params = _init(on, sequence, valid)

    perturbed = sequence.at[OPP_PRIVATE_ROWS].add(10.0)
    perturbed = perturbed.at[PRIVILEGED_REGISTER_ROWS].add(10.0)
    base = _forward(on, params, sequence, valid)
    moved = _forward(on, params, perturbed, valid)
    np.testing.assert_array_equal(base[_POLICY_READABLE], moved[_POLICY_READABLE])
    assert not np.allclose(base[VALUE_CLS_ROW], moved[VALUE_CLS_ROW])
    control = sequence.at[3].add(10.0)
    control_out = _forward(on, params, control, valid)
    assert not np.allclose(base[_POLICY_READABLE], control_out[_POLICY_READABLE])


def _as_player_tree(params):
    return {"params": {"encoder": {"trunk": params["params"]}}}


def _vector_norms(blocks, sublayer: str, layer: str, axis: int) -> np.ndarray:
    kernel = np.asarray(blocks[sublayer][layer]["kernel"], np.float32)
    return np.linalg.norm(kernel, axis=axis)


def test_projection_puts_every_embedding_space_vector_on_the_unit_sphere() -> None:
    sequence, valid = _inputs()
    on = Trunk(_trunk_cfg(True))
    fresh = _as_player_tree(_init(on, sequence, valid))
    fresh_blocks = fresh["params"]["encoder"]["trunk"]["blocks"]
    # Positive control: the fresh init is not on the sphere.
    assert not np.allclose(
        _vector_norms(fresh_blocks, "ffw", "Dense_0", axis=-2), 1.0, atol=1e-3
    )

    projected = project_trunk_kernels(fresh)
    blocks = projected["params"]["encoder"]["trunk"]["blocks"]
    for sublayer, layer in (
        ("attention", "q_proj"),
        ("attention", "k_proj"),
        ("attention", "v_proj"),
        ("ffw", "Dense_0"),
    ):
        np.testing.assert_allclose(
            _vector_norms(blocks, sublayer, layer, axis=-2), 1.0, atol=1e-5
        )
    for sublayer, layer in (("attention", "out_proj"), ("ffw", "Dense_1")):
        np.testing.assert_allclose(
            _vector_norms(blocks, sublayer, layer, axis=-1), 1.0, atol=1e-5
        )

    # Everything that is not one of those six kernels is bit-identical:
    # biases, norm scales, the alphas, the attention's qk norms.
    projected_kernels = {
        ("attention", "q_proj"),
        ("attention", "k_proj"),
        ("attention", "v_proj"),
        ("attention", "out_proj"),
        ("ffw", "Dense_0"),
        ("ffw", "Dense_1"),
    }
    for path, leaf in jax.tree_util.tree_leaves_with_path(fresh):
        keys = tuple(entry.key for entry in path)
        if keys[-3:-1] in projected_kernels and keys[-1] == "kernel":
            continue
        after = projected
        for key in keys:
            after = after[key]
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(after))

    # A tree without the trunk path is returned untouched.
    other = {"params": {"world_model": {"trunk": {"blocks": fresh_blocks}}}}
    jax.tree.map(np.testing.assert_array_equal, project_trunk_kernels(other), other)


def test_alpha_telemetry_exists_exactly_with_the_flag() -> None:
    sequence, valid = _inputs()
    on = Trunk(_trunk_cfg(True))
    off = Trunk(_trunk_cfg(False))
    logs_on = trunk_alpha_telemetry(
        {"encoder": {"trunk": _init(on, sequence, valid)["params"]}}
    )
    logs_off = trunk_alpha_telemetry(
        {"encoder": {"trunk": _init(off, sequence, valid)["params"]}}
    )
    alpha_keys = sorted(key for key in logs_on if key.startswith("player_trunk_alpha_"))
    assert alpha_keys == sorted(
        f"player_trunk_alpha_{sublayer}_b{block}"
        for sublayer in ("attention", "ffw")
        for block in range(NUM_BLOCKS)
    )
    for key in alpha_keys:
        np.testing.assert_allclose(np.asarray(logs_on[key]), 0.05, atol=1e-7)
    assert not [key for key in logs_off if key.startswith("player_trunk_alpha_")]
    # The kernel column norms read on both trees: they are the plain
    # trunk's spectral-growth panel.
    for logs in (logs_on, logs_off):
        assert "player_trunk_kernel_col_norm_attention_q" in logs
        assert "player_trunk_kernel_col_norm_ffw_up" in logs
