"""`row_homogeneity` -- the over-smoothing instrument -- on sets whose
answer is known in closed form, plus the control that an invalid (zeroed)
row is excluded rather than counted as a direction."""

import jax.numpy as jnp
import numpy as np

from rl.model.trunk import row_homogeneity


def _np(values: np.ndarray) -> tuple[float, float]:
    cosine, participation = row_homogeneity(jnp.asarray(values))
    return float(cosine), float(participation)


def test_orthonormal_rows_are_maximally_spread() -> None:
    rows = 8
    cosine, participation = _np(np.eye(rows, 16, dtype=np.float32))
    assert abs(cosine) < 1e-6
    # Centring removes the mean direction: an orthonormal set spans rows-1
    # equal-variance directions around its mean.
    assert abs(participation - (rows - 1)) < 1e-4


def test_identical_rows_are_fully_collapsed() -> None:
    values = np.tile(np.arange(1, 17, dtype=np.float32), (8, 1))
    cosine, participation = _np(values)
    assert abs(cosine - 1.0) < 1e-4  # tf32 matmul on the GPU
    assert np.isnan(participation)


def _raw_cosine(values: np.ndarray) -> float:
    unit = values / np.linalg.norm(values, axis=-1, keepdims=True)
    cosines = unit @ unit.T
    off_diagonal = ~np.eye(len(values), dtype=bool)
    return float(cosines[off_diagonal].mean())


def test_one_large_shared_channel_cannot_carry_the_cosine() -> None:
    """A shared large offset in ONE channel over an orthonormal spread: the
    raw cosine reads ~1 (the control, computed here), the channel-scaled
    reading counts it as one agreeing channel among nine live ones, and
    the centred participation is untouched either way."""
    rows = 8
    spread = np.eye(rows, 16, dtype=np.float32)
    offset = np.zeros(16, dtype=np.float32)
    offset[-1] = 30.0
    assert _raw_cosine(spread + offset) > 0.99
    cosine, participation = _np(spread + offset)
    # Scaled: the spread channels sit at sqrt(rows) in their one row and the
    # offset channel at 1 everywhere, so each pair agrees on 1 / (rows + 1).
    assert abs(cosine - 1 / (rows + 1)) < 1e-4
    assert abs(participation - (rows - 1)) < 1e-4


def test_agreement_across_many_channels_still_reads_collapsed() -> None:
    """The scaling must not erase real convergence: rows that share the same
    direction across every channel, at different magnitudes, read ~1."""
    rng = np.random.default_rng(2)
    direction = rng.normal(size=16).astype(np.float32)
    magnitudes = np.linspace(0.5, 4.0, 8, dtype=np.float32)
    cosine, _ = _np(magnitudes[:, None] * direction[None, :])
    assert abs(cosine - 1.0) < 1e-4


def test_zeroed_row_is_excluded() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(10, 32)).astype(np.float32)
    with_hole = values.copy()
    with_hole[3] = 0.0
    subset = np.delete(values, 3, axis=0)
    np.testing.assert_allclose(_np(with_hole), _np(subset), rtol=1e-5)
    # Control: the same slot carrying content changes both readings.
    assert not np.allclose(_np(values), _np(subset))


def test_batched_over_leading_axes() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, 5, 10, 32)).astype(np.float32)
    cosine, participation = row_homogeneity(jnp.asarray(values))
    assert cosine.shape == (3, 5) and participation.shape == (3, 5)
    single = _np(values[2, 4])
    np.testing.assert_allclose(
        (float(cosine[2, 4]), float(participation[2, 4])), single, rtol=1e-5
    )


def test_fewer_than_two_valid_rows_is_nan_not_zero() -> None:
    """A group with no pair (PREV_ACTION in singles: rows never valid) must
    read as no reading, not as "perfectly spread" -- a 0 cosine there would
    average into the table as if it were data."""
    single = np.zeros((4, 8), np.float32)
    single[0] = np.arange(8)
    cosine, participation = row_homogeneity(jnp.asarray(single))
    assert np.isnan(cosine)
    assert np.isnan(participation)
    # Control: two live rows do produce a number.
    single[1] = np.arange(8)[::-1]
    cosine, _ = row_homogeneity(jnp.asarray(single))
    assert np.isfinite(cosine)
