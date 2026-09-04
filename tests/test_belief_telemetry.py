"""The belief head's honest accuracy (2026-09-02): `player_belief_accuracy`
beside the majority rate a constant predictor scores on the same rows, so
a head that has learnt the batch marginal reads as ~0 above it.
"""

import jax.numpy as jnp
import numpy as np

from rl.online.training.telemetry import belief_accuracy_logs, code_usage_logs


def _skewed_labels(seed=0, time=5, batch=3, mons=6, groups=4, classes=8):
    rng = np.random.default_rng(seed)
    # A different skew per group so the majority class is not class 0 in
    # every group by accident.
    probs = rng.dirichlet(np.full(classes, 0.3), size=groups)
    classes_drawn = np.stack(
        [
            rng.choice(classes, size=(time, batch, mons), p=probs[group])
            for group in range(groups)
        ],
        axis=-1,
    )
    labels = np.eye(classes)[classes_drawn]
    mask = rng.random((time, batch, mons)) < 0.7
    return jnp.asarray(labels, jnp.float32), jnp.asarray(mask)


def test_constant_majority_predictor_is_zero_above_marginal():
    labels, mask = _skewed_labels()
    weights = mask[..., None, None].astype(jnp.float32)
    marginal = (labels * weights).sum((0, 1, 2)) / weights.sum((0, 1, 2))
    # Logits that always pick each group's majority class.
    logits = jnp.broadcast_to(marginal, labels.shape)
    logs = belief_accuracy_logs(logits, labels, mask)
    assert float(logs["player_belief_majority_rate"]) == float(marginal.max(-1).mean())
    np.testing.assert_allclose(
        float(logs["player_belief_accuracy"]), float(marginal.max(-1).mean())
    )
    np.testing.assert_allclose(
        float(logs["player_belief_accuracy_above_marginal"]), 0.0, atol=1e-6
    )


def test_perfect_predictor_is_one_minus_majority_above_marginal():
    """Positive control: the same labels read as logits score 1.0, so the
    test above is not passing because nothing can move the number."""
    labels, mask = _skewed_labels()
    logs = belief_accuracy_logs(labels, labels, mask)
    np.testing.assert_allclose(float(logs["player_belief_accuracy"]), 1.0)
    majority = float(logs["player_belief_majority_rate"])
    assert 0.0 < majority < 1.0
    np.testing.assert_allclose(
        float(logs["player_belief_accuracy_above_marginal"]), 1.0 - majority
    )


def test_majority_rate_reads_the_belief_mask_rows_only():
    """The baseline population is the SCORED rows: masking out every row of
    one class must remove that class from the marginal."""
    labels, mask = _skewed_labels()
    majority_class = jnp.argmax(labels, -1)[..., 0]
    top = jnp.argmax((labels[..., 0, :] * mask[..., None]).sum((0, 1, 2)))
    without_top = mask & (majority_class != top)
    logs = belief_accuracy_logs(labels, labels, without_top)
    weights = without_top[..., None, None].astype(jnp.float32)
    marginal = (labels * weights).sum((0, 1, 2)) / weights.sum((0, 1, 2))
    assert float(marginal[0, top]) == 0.0
    np.testing.assert_allclose(
        float(logs["player_belief_majority_rate"]),
        float(marginal.max(-1).mean()),
        rtol=1e-6,
    )


def test_code_usage_perplexity_reads_one_at_a_dead_group():
    """The factored marginal keeps the usage panel's meaning: a group using
    one class reads perplexity exactly 1, a uniform one reads K."""
    time, batch, mons, groups, classes = 4, 2, 6, 3, 8
    rng = np.random.default_rng(1)
    drawn = rng.integers(0, classes, size=(time, batch, mons, groups))
    drawn[..., 0] = 3
    code = jnp.asarray(np.eye(classes)[drawn], jnp.float32)
    team = jnp.ones((time, batch, mons, 40), jnp.int32)
    value_mask = jnp.ones((time, batch), bool)
    logs = code_usage_logs(code, team, value_mask)
    np.testing.assert_allclose(
        float(logs["player_code_perplexity_min"]), 1.0, rtol=1e-5
    )
    assert float(logs["player_code_perplexity_mean"]) > 1.0
    assert float(logs["player_code_row_frac"]) == 1.0


def test_code_usage_row_mask_narrows_the_population_and_renames():
    """The hidden-token label's panel (2026-09-05): `row_mask` restricts
    the marginal to the rows the belief loss scores, and `prefix` names
    the panel. Rows outside the mask use a second class in every group;
    inside it every row uses one, so the narrowed read is exactly 1 where
    the full read is not."""
    time, batch, mons, groups, classes = 3, 2, 6, 4, 5
    drawn = np.full((time, batch, mons, groups), 2)
    drawn[:, :, 3:] = 4
    code = jnp.asarray(np.eye(classes)[drawn], jnp.float32)
    team = jnp.ones((time, batch, mons, 40), jnp.int32)
    value_mask = jnp.ones((time, batch), bool)
    row_mask = (
        jnp.asarray(np.arange(mons) < 3)[None, None].repeat(time, 0).repeat(batch, 1)
    )
    full = code_usage_logs(code, team, value_mask)
    assert float(full["player_code_perplexity_min"]) > 1.0
    narrowed = code_usage_logs(
        code, team, value_mask, row_mask=row_mask, prefix="player_hidden_code"
    )
    assert set(narrowed) == {
        "player_hidden_code_perplexity_mean",
        "player_hidden_code_perplexity_min",
        "player_hidden_code_row_frac",
    }
    np.testing.assert_allclose(
        float(narrowed["player_hidden_code_perplexity_mean"]), 1.0, rtol=1e-5
    )
