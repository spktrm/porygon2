"""Flat legal-cell support telemetry: how close the policy is to pruning.

Over each real decision row's legal cells, read as flat complete actions
(a move x target, or one switch, is one cell — no hierarchy): the minimum
and median cell probability, the legal-cell count, and the fraction of
legal cells below each of three lines, then the minimum and the fractions
split over the switch and move cells so the switch floor reads on its
own. Observers only: no loss reads them. They are the exposure instrument
for the support hinge (what it must lift) and the v-trace threshold (what
it will discard), and the calibration input for both.
"""

import jax
import jax.numpy as jnp

from rl.utils import average

SUPPORT_LINES = (("p01", 0.01), ("p005", 0.005), ("p001", 0.001))


def _fraction_below(policy, cells, line):
    return (cells & (policy < line)).sum(-1) / jnp.maximum(cells.sum(-1), 1)


def _minimum(policy, cells):
    return jnp.where(cells, policy, jnp.inf).min(-1)


def _median(policy, cells):
    ordered = jnp.sort(jnp.where(cells, policy, jnp.inf), axis=-1)
    middle = (jnp.maximum(cells.sum(-1), 1) - 1) // 2
    return jnp.take_along_axis(ordered, middle[..., None], axis=-1)[..., 0]


def legal_support_telemetry(
    log_policy: jax.Array,
    legal_mask: jax.Array,
    switch_cells: jax.Array,
    policy_mask: jax.Array,
) -> dict[str, jax.Array]:
    """`log_policy` (..., cells) the learner's full log-policy, `legal_mask`
    the row's legal cells, `switch_cells` the (cells,) switch indicator,
    `policy_mask` the real decision rows. Split readouts average over the
    rows that have at least one cell of that kind."""
    log_policy = jax.lax.stop_gradient(log_policy.astype(jnp.float32))
    policy = jnp.where(legal_mask, jnp.exp(log_policy), 0.0)
    legal_count = legal_mask.sum(-1)
    logs = {
        "player_support_min_prob": average(_minimum(policy, legal_mask), policy_mask),
        "player_support_median_prob": average(_median(policy, legal_mask), policy_mask),
        "player_support_legal_count": average(legal_count, policy_mask),
    }
    for name, line in SUPPORT_LINES:
        logs[f"player_support_frac_below_{name}"] = average(
            _fraction_below(policy, legal_mask, line), policy_mask
        )
    for kind, kind_cells in (
        ("switch", legal_mask & switch_cells),
        ("move", legal_mask & jnp.logical_not(switch_cells)),
    ):
        rows = policy_mask & kind_cells.any(-1)
        logs[f"player_support_{kind}_min_prob"] = average(
            _minimum(policy, kind_cells), rows
        )
        for name, line in SUPPORT_LINES:
            logs[f"player_support_{kind}_frac_below_{name}"] = average(
                _fraction_below(policy, kind_cells, line), rows
            )
    return logs
