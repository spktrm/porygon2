"""`row_homogeneity` -- the over-smoothing instrument -- on sets whose
answer is known in closed form, plus the control that an invalid (zeroed)
row is excluded rather than counted as a direction."""

import jax.numpy as jnp
import numpy as np

from rl.model.trunk import row_homogeneity


def _np(values):
    cosine, participation = row_homogeneity(jnp.asarray(values))
    return float(cosine), float(participation)


def test_orthonormal_rows_are_maximally_spread():
    rows = 8
    cosine, participation = _np(np.eye(rows, 16, dtype=np.float32))
    assert abs(cosine) < 1e-6
    # Centring removes the mean direction: an orthonormal set spans rows-1
    # equal-variance directions around its mean.
    assert abs(participation - (rows - 1)) < 1e-4


def test_identical_rows_are_fully_collapsed():
    values = np.tile(np.arange(1, 17, dtype=np.float32), (8, 1))
    cosine, participation = _np(values)
    assert abs(cosine - 1.0) < 1e-4  # tf32 matmul on the GPU
    assert np.isnan(participation)


def test_common_offset_reads_on_cosine_not_participation():
    """A shared large offset over an orthonormal spread: cosine goes to ~1
    while the centred participation is untouched -- the two instruments
    disagree exactly when a common direction carries a live residual."""
    rows = 8
    spread = np.eye(rows, 16, dtype=np.float32)
    offset = np.zeros(16, dtype=np.float32)
    offset[-1] = 30.0
    cosine, participation = _np(spread + offset)
    assert cosine > 0.99
    assert abs(participation - (rows - 1)) < 1e-4


def test_zeroed_row_is_excluded():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(10, 32)).astype(np.float32)
    with_hole = values.copy()
    with_hole[3] = 0.0
    subset = np.delete(values, 3, axis=0)
    np.testing.assert_allclose(_np(with_hole), _np(subset), rtol=1e-5)
    # Control: the same slot carrying content changes both readings.
    assert not np.allclose(_np(values), _np(subset))


def test_batched_over_leading_axes():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, 5, 10, 32)).astype(np.float32)
    cosine, participation = row_homogeneity(jnp.asarray(values))
    assert cosine.shape == (3, 5) and participation.shape == (3, 5)
    single = _np(values[2, 4])
    np.testing.assert_allclose(
        (float(cosine[2, 4]), float(participation[2, 4])), single, rtol=1e-5
    )


def test_fewer_than_two_valid_rows_is_nan_not_zero():
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
