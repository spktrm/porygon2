"""Depth-1 expectimax over the transition model (rl/model/search.py,
stochastic-transition Step 3).

The operator is a free function over callables, so the unit half runs it
on stubs where the value of every imagined state is set by hand: the Q it
returns must rank the legal cells by that hand-set value (the positive
control), score illegal cells exactly 0 with no padding leak (the
`nonzero` fill lands on cell 0 -- a SET scatter would double-count a
legal cell 0), and the diagnostics must read 0 when the bonus is 0. The
real-model half runs the search-enabled actor network on the bundled
trajectory with the zero-init paths OPENED (fresh params make every logit
0, so "the bonus moves the policy" would pass or fail vacuously) and pins
that the searched arm's policy differs from the plain arm's while the
plain arm's output carries no search leaves at all.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import NUM_ACTION_CELLS, NUM_SWITCH_CELLS
from rl.environment.interfaces import (
    PlayerActorOutput,
    PlayerAgentOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PlayerTransition,
    SearchOutput,
    Trajectory,
)
from rl.model.constants import CLS_ROW, NUM_POLICY_READABLE_ROWS
from rl.model.search import depth_one_expectimax, search_diagnostics
from rl.model.utils import open_zero_init_paths

MODEL_SIZE = 8
MAX_CELLS = 16
NUM_SAMPLES = 4
TEMP = 0.1


class _Expectation:
    def __init__(self, expectation):
        self.expectation = expectation


def _stub_search(rows, legal, cell_value, code_groups=2, code_classes=16):
    """A search whose imagined state's value is `cell_value[cell]`
    regardless of the code: `imagine_fn` writes the taken cell's value
    into the CLS row, `value_fn` reads it back."""
    row_valid = jnp.ones(rows.shape[0], bool)

    def prior_fn(rows, row_valid, cell):
        return jnp.zeros((code_groups, code_classes), jnp.float32)

    def action_rows_fn(rows, cell):
        return jnp.full((MODEL_SIZE,), cell_value[cell]), rows[0]

    def imagine_fn(rows, row_valid, src_row, tgt_row, code_one_hot):
        return rows.at[CLS_ROW].set(src_row)

    def value_fn(cls_rows):
        return _Expectation(cls_rows[..., 0])

    return depth_one_expectimax(
        rows,
        row_valid,
        legal,
        jax.random.key(0),
        prior_fn=prior_fn,
        action_rows_fn=action_rows_fn,
        imagine_fn=imagine_fn,
        value_fn=value_fn,
        num_samples=NUM_SAMPLES,
        max_cells=MAX_CELLS,
        temp=TEMP,
    )


def test_root_q_ranks_legal_cells_by_the_imagined_value():
    rows = jnp.zeros((NUM_POLICY_READABLE_ROWS, MODEL_SIZE))
    legal_cells = np.array([3, 40, 120])
    legal = np.zeros(NUM_ACTION_CELLS, bool)
    legal[legal_cells] = True
    cell_value = np.zeros(NUM_ACTION_CELLS, np.float32)
    cell_value[legal_cells] = [0.2, -0.5, 0.9]
    root = jax.jit(_stub_search)(rows, jnp.asarray(legal), jnp.asarray(cell_value))
    q = np.asarray(root.q)
    np.testing.assert_allclose(q[legal_cells], cell_value[legal_cells], atol=1e-6)
    assert np.argmax(q) == 120 and q[40] == q[legal].min()
    np.testing.assert_allclose(
        np.asarray(root.bonus)[legal_cells], cell_value[legal_cells] / TEMP, atol=1e-5
    )
    assert int(root.num_legal) == 3 and not bool(root.legal_truncated)
    # Illegal cells: exactly 0 on both, no padding leak.
    assert np.all(q[~legal] == 0) and np.all(np.asarray(root.bonus)[~legal] == 0)


def test_padding_never_double_counts_cell_zero():
    # Cell 0 legal and the `nonzero` fill value is 0: with 13 padded
    # entries all pointing at cell 0, a set-scatter or an unmasked sum
    # would corrupt its Q. It must equal its own value exactly.
    rows = jnp.zeros((NUM_POLICY_READABLE_ROWS, MODEL_SIZE))
    legal = np.zeros(NUM_ACTION_CELLS, bool)
    legal[[0, 9, 17]] = True
    cell_value = np.zeros(NUM_ACTION_CELLS, np.float32)
    cell_value[[0, 9, 17]] = [0.7, 0.1, 0.3]
    root = jax.jit(_stub_search)(rows, jnp.asarray(legal), jnp.asarray(cell_value))
    np.testing.assert_allclose(float(root.q[0]), 0.7, atol=1e-6)


def test_more_legal_cells_than_max_is_counted_and_scored_on_the_first_max():
    rows = jnp.zeros((NUM_POLICY_READABLE_ROWS, MODEL_SIZE))
    legal = np.zeros(NUM_ACTION_CELLS, bool)
    legal[: MAX_CELLS + 2] = True
    cell_value = np.arange(NUM_ACTION_CELLS, dtype=np.float32) / NUM_ACTION_CELLS
    root = jax.jit(_stub_search)(rows, jnp.asarray(legal), jnp.asarray(cell_value))
    assert bool(root.legal_truncated) and int(root.num_legal) == MAX_CELLS + 2
    q = np.asarray(root.q)
    np.testing.assert_allclose(q[:MAX_CELLS], cell_value[:MAX_CELLS], atol=1e-6)
    # The two cells past the cap are legal but unscored: 0, and the bonus
    # there is 0 too (they keep the policy's own logit alone).
    assert np.all(q[MAX_CELLS : MAX_CELLS + 2] == 0)


def test_diagnostics_read_zero_at_zero_bonus_and_the_kl_of_a_tilt():
    legal = np.zeros(NUM_ACTION_CELLS, bool)
    legal[[2, 5, 8, 11]] = True
    legal = jnp.asarray(legal)
    base_logits = jnp.asarray(np.random.default_rng(0).normal(size=NUM_ACTION_CELLS))
    zero_root = _root_with(jnp.zeros(NUM_ACTION_CELLS), legal)
    diag = search_diagnostics(base_logits, zero_root, legal, jnp.float32(0.25))
    assert float(diag.root_kl) == 0.0
    assert float(diag.search_value) == 0.0 and float(diag.root_value_gap) == -0.25
    # A tilt onto one cell: KL > 0 and the search value is E_{pi_search}[q].
    q = jnp.zeros(NUM_ACTION_CELLS).at[8].set(1.0)
    root = _root_with(q, legal)
    diag = search_diagnostics(base_logits, root, legal, jnp.float32(0.0))
    assert float(diag.root_kl) > 0.0
    pi_search = jax.nn.softmax(jnp.where(legal, base_logits + root.bonus, -1e9))
    np.testing.assert_allclose(
        float(diag.search_value), float((pi_search * q).sum()), rtol=1e-5
    )


def _root_with(q, legal):
    from rl.model.search import SearchRoot

    return SearchRoot(
        q=q,
        bonus=jnp.where(legal, q / TEMP, 0.0),
        num_legal=legal.sum(),
        legal_truncated=jnp.asarray(False),
    )


def test_eval_game_logs_read_real_acted_rows_and_search_leaves_when_present():
    from rl.online.main import eval_game_logs

    num_rows = 6
    done = np.zeros(num_rows, bool)
    done[3:] = True  # row 3 is the terminal row; 4-5 copy it (padding)
    action_mask = np.zeros((num_rows, NUM_ACTION_CELLS), bool)
    action_mask[:, NUM_SWITCH_CELLS] = True  # a move is always legal
    action_mask[[0, 2], 0] = True  # rows 0 and 2 offer a switch
    action_index = np.array([0, NUM_SWITCH_CELLS, NUM_SWITCH_CELLS, 0, 0, 0])
    root_kl = np.array([0.1, 0.2, 0.3, 9.0, 9.0, 9.0], np.float32)

    def trajectory(search):
        return Trajectory(
            player_transitions=PlayerTransition(
                env_output=PlayerEnvOutput(done=done, action_mask=action_mask),
                agent_output=PlayerAgentOutput(
                    actor_output=PlayerActorOutput(
                        action_head=PlayerPolicyHeadOutput(action_index=action_index),
                        search=search,
                    )
                ),
            ),
            game_length=np.array([10]),
            game_step_offset=np.array([6]),  # 4 real rows: 0..3
        )

    logs = eval_game_logs(trajectory(SearchOutput()), 2.0, "s")
    # Two decisions offered a switch (rows 0, 2), one took it.
    assert logs["switch-frac-s"] == 0.5 and logs["ms-per-step-s"] == 500.0
    assert not any(key.startswith("search-") for key in logs)
    logs = eval_game_logs(
        trajectory(
            SearchOutput(
                root_kl=root_kl,
                search_value=root_kl,
                root_value_gap=root_kl,
                num_legal=root_kl,
                legal_truncated=np.ones(num_rows, bool),
            )
        ),
        2.0,
        "s",
    )
    # Acted rows are 0-2 only: the terminal row and the padding are out.
    np.testing.assert_allclose(logs["search-root-kl-s"], 0.2, rtol=1e-6)
    assert logs["search-legal-truncated-s"] == 1.0


@pytest.mark.gpu
@pytest.mark.slow
def test_search_moves_the_real_policy_and_the_plain_arm_carries_no_leaves(
    real_model_and_trajectory,
):
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model

    _, params, actor_input, actor_output = real_model_and_trajectory
    actor_input = jax.tree.map(lambda leaf: leaf[:4], actor_input)
    actor_output = jax.tree.map(lambda leaf: leaf[:4], actor_output)
    # Fresh params emit all-zero logits and an out_proj-zero transition
    # (the copy predictor): open both so the operator has something to
    # read and something to move.
    params = open_zero_init_paths(params, ["action_head", "out_proj"])
    plain_cfg = get_player_model_config(9, train=False)
    search_cfg = get_player_model_config(9, train=False)
    search_cfg.search.enabled = True
    rngs = {"sampling": jax.random.key(1)}
    plain = jax.jit(get_player_model(plain_cfg).apply)(
        params, actor_input, actor_output, HeadParams(), rngs=rngs
    )
    searched = jax.jit(get_player_model(search_cfg).apply)(
        params, actor_input, actor_output, HeadParams(), rngs=rngs
    )
    assert all(isinstance(leaf, tuple) for leaf in dataclasses.astuple(plain.search))
    assert searched.search.root_kl.shape == (4,)
    assert np.all(np.asarray(searched.search.root_kl) > 0)
    assert np.all(np.isfinite(np.asarray(searched.search.root_value_gap)))
    assert not np.allclose(
        np.asarray(plain.action_head.entropy), np.asarray(searched.action_head.entropy)
    )
