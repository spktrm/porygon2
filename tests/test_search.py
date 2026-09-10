"""Search over the transition model (rl/model/search.py; latent actions
and the recursive backup 2026-09-07).

The operator is a free function over callables, so the unit half runs it
on stubs where the value of every imagined state is set by hand: the
root Q must rank the legal cells by that hand-set value (the positive
control), score illegal cells exactly 0 with no padding leak (the
`nonzero` fill lands on cell 0 -- a SET scatter would double-count a
legal cell 0), return the base policy untouched on overflow, and the
diagnostics must read 0 when the bonus is 0. The depth-2 backup is
checked against a hand-enumerated tree: chance averaged (never
maximised), the candidates improved by the KL-regularised softmax, the
terminal outcome counted once through the continuation, the plan's
counterexamples (c = 0.5, T = +1, Q = -1 -> 0; c = 0.25, T = -1, Q = +1
-> -0.5) reproduced. The real-model half runs the search-enabled actor
network on the bundled trajectory with the zero-init paths OPENED (fresh
params make every logit 0, so "the bonus moves the policy" would pass or
fail vacuously) at both depths, and pins that the plain arm's output
carries no search leaves at all.
"""

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.data import NUM_ACTION_CELLS, NUM_SWITCH_CELLS
from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerAgentOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PlayerTransition,
    SearchOutput,
    Trajectory,
)
from rl.model.constants import CLS_ROW, NUM_POLICY_READABLE_ROWS
from rl.model.search import (
    SearchBudget,
    SearchFns,
    SearchRoot,
    chance_backup,
    sample_chance,
    search_diagnostics,
    search_root,
    subset_improvement,
)
from rl.model.transition import Candidates
from rl.model.utils import open_zero_init_paths

MODEL_SIZE = 8
MAX_CELLS = 16
NUM_SAMPLES = 4
NUM_CODES = 64
NUM_CANDIDATES = 8
TEMP = 0.1
CODE_GROUPS = 2
CODE_CLASSES = 16


def _budget(depth: int = 1, num_samples_inner: int = 2) -> SearchBudget:
    return SearchBudget(
        depth=depth,
        num_samples=NUM_SAMPLES,
        num_samples_inner=num_samples_inner,
        max_cells=MAX_CELLS,
        temp=TEMP,
    )


def _stub_fns(
    code_value: np.ndarray,
    candidate_codes: np.ndarray | None = None,
    candidate_rho: np.ndarray | None = None,
    occupied: np.ndarray | None = None,
    continue_prob: float = 1.0,
    terminal: float = 0.0,
) -> SearchFns:
    """A model whose imagined state's value is `code_value[u]` whatever
    the chance code: `encoder_fn` maps cell c to code c mod 64 exactly,
    `imagine_fn` writes the code's value into the CLS row, `value_fn`
    reads it back. The depth-2 stubs are constants set by hand."""
    code_value = jnp.asarray(code_value, jnp.float32)
    if candidate_codes is None:
        candidate_codes = np.arange(NUM_CANDIDATES)
    candidate_codes = jnp.asarray(candidate_codes, jnp.int32)
    if candidate_rho is None:
        candidate_rho = np.full(NUM_CANDIDATES, 1.0 / NUM_CANDIDATES)
    candidate_rho = jnp.asarray(candidate_rho, jnp.float32)
    if occupied is None:
        occupied = np.ones(NUM_CANDIDATES, bool)
    occupied = jnp.asarray(occupied)

    def encoder_fn(rows: jax.Array, cell: jax.Array) -> jax.Array:
        return jax.nn.one_hot(cell % NUM_CODES, NUM_CODES, dtype=jnp.float32)

    def prior_fn(rows: jax.Array, action_one_hot: jax.Array) -> jax.Array:
        return jnp.zeros((CODE_GROUPS, CODE_CLASSES), jnp.float32)

    def imagine_fn(
        rows: jax.Array, action_one_hot: jax.Array, code_one_hot: jax.Array
    ) -> jax.Array:
        return rows.at[CLS_ROW, 0].set(action_one_hot @ code_value)

    def value_fn(cls_row: jax.Array) -> jax.Array:
        return cls_row[0]

    def generate_fn(rows: jax.Array, rng: jax.Array) -> Candidates:
        return Candidates(
            codes=candidate_codes,
            occupied=occupied,
            log_draw_conditionals=jnp.zeros(NUM_CANDIDATES, jnp.float32),
            rho_at_codes=candidate_rho,
            support_mask=jnp.zeros(NUM_CODES, bool).at[candidate_codes].set(True),
            retained_mass=jnp.sum(candidate_rho, where=occupied),
        )

    def continue_fn(rows: jax.Array) -> jax.Array:
        return jnp.float32(continue_prob)

    def terminal_fn(rows: jax.Array) -> jax.Array:
        return jnp.float32(terminal)

    return SearchFns(
        encoder_fn=encoder_fn,
        prior_fn=prior_fn,
        imagine_fn=imagine_fn,
        value_fn=value_fn,
        generate_fn=generate_fn,
        continue_fn=continue_fn,
        terminal_fn=terminal_fn,
    )


def _legal(cells: np.ndarray | list[int]) -> jax.Array:
    legal = np.zeros(NUM_ACTION_CELLS, bool)
    legal[cells] = True
    return jnp.asarray(legal)


def _rows() -> jax.Array:
    return jnp.zeros((NUM_POLICY_READABLE_ROWS, MODEL_SIZE))


def test_root_q_ranks_legal_cells_by_the_imagined_value() -> None:
    legal_cells = np.array([3, 40, 120])
    code_value = np.zeros(NUM_CODES, np.float32)
    code_value[legal_cells % NUM_CODES] = [0.2, -0.5, 0.9]
    root = jax.jit(
        lambda rows, legal: search_root(
            rows, legal, jax.random.key(0), _stub_fns(code_value), _budget()
        )
    )(_rows(), _legal(legal_cells))
    q = np.asarray(root.cell_values)
    np.testing.assert_allclose(q[legal_cells], [0.2, -0.5, 0.9], atol=1e-6)
    assert np.argmax(q) == 120 and q[40] == q[np.asarray(_legal(legal_cells))].min()
    np.testing.assert_allclose(
        np.asarray(root.bonus)[legal_cells],
        np.array([0.2, -0.5, 0.9]) / TEMP,
        atol=1e-5,
    )
    assert int(root.num_legal) == 3 and not bool(root.legal_truncated)
    legal = np.asarray(_legal(legal_cells))
    # Illegal cells: exactly 0 on both, no padding leak.
    assert np.all(q[~legal] == 0) and np.all(np.asarray(root.bonus)[~legal] == 0)
    # Depth 1 leaves the deep reads at 0.
    assert float(root.deep_gain) == 0.0 and float(root.candidate_occupied) == 0.0


def test_padding_never_double_counts_cell_zero() -> None:
    # Cell 0 legal and the `nonzero` fill value is 0: with 13 padded
    # entries all pointing at cell 0, a set-scatter or an unmasked sum
    # would corrupt its Q. It must equal its own value exactly.
    code_value = np.zeros(NUM_CODES, np.float32)
    code_value[[0, 9, 17]] = [0.7, 0.1, 0.3]
    root = jax.jit(
        lambda rows, legal: search_root(
            rows, legal, jax.random.key(0), _stub_fns(code_value), _budget()
        )
    )(_rows(), _legal([0, 9, 17]))
    np.testing.assert_allclose(float(root.cell_values[0]), 0.7, atol=1e-6)


def test_overflow_returns_the_base_policy_and_is_counted() -> None:
    code_value = np.arange(NUM_CODES, dtype=np.float32) / NUM_CODES
    root = jax.jit(
        lambda rows, legal: search_root(
            rows, legal, jax.random.key(0), _stub_fns(code_value), _budget()
        )
    )(_rows(), _legal(np.arange(MAX_CELLS + 2)))
    assert bool(root.legal_truncated) and int(root.num_legal) == MAX_CELLS + 2
    # No bonus anywhere: the actor samples its own policy on this decision.
    assert np.all(np.asarray(root.bonus) == 0)
    # The scored cells keep their Q as a read.
    np.testing.assert_allclose(
        np.asarray(root.cell_values)[:MAX_CELLS], code_value[:MAX_CELLS], atol=1e-6
    )


def test_chance_is_averaged_over_the_prior_draws_never_maximised() -> None:
    """`chance_backup` is the MEAN of the leaf over the drawn codes: with
    a leaf that reads the first chance group's class, the value equals
    the mean over the same draws `sample_chance` produces from the same
    key -- and not their max."""
    rows = _rows()
    key = jax.random.key(3)
    prior_logits = jnp.zeros((CODE_GROUPS, CODE_CLASSES), jnp.float32)
    class_value = jnp.linspace(-1.0, 1.0, CODE_CLASSES)

    def prior_fn(rows: jax.Array, action_one_hot: jax.Array) -> jax.Array:
        return prior_logits

    def imagine_fn(
        rows: jax.Array, action_one_hot: jax.Array, code_one_hot: jax.Array
    ) -> jax.Array:
        return rows.at[CLS_ROW, 0].set(code_one_hot[0] @ class_value)

    fns = _stub_fns(np.zeros(NUM_CODES))._replace(
        prior_fn=prior_fn, imagine_fn=imagine_fn
    )
    num_samples = 6
    value = chance_backup(
        rows,
        jnp.zeros(NUM_CODES),
        key,
        fns,
        num_samples,
        lambda node: fns.value_fn(node[CLS_ROW]),
    )
    codes = sample_chance(prior_logits, key, num_samples)
    drawn_values = np.asarray(codes[:, 0] @ class_value)
    assert len(set(drawn_values.tolist())) > 1, "the draws must differ to test"
    np.testing.assert_allclose(float(value), drawn_values.mean(), rtol=1e-6)
    assert float(value) < drawn_values.max()


def test_subset_improvement_is_the_kl_regularised_softmax_over_the_set() -> None:
    rho = jnp.asarray([0.5, 0.3, 0.1, 0.1])
    q = jnp.asarray([0.0, 0.2, -0.3, 0.9])
    occupied = jnp.asarray([True, True, True, False])
    value = subset_improvement(rho, q, occupied, TEMP)
    rho_set = np.asarray(rho[:3]) / np.asarray(rho[:3]).sum()
    logits = np.log(rho_set) + np.asarray(q[:3]) / TEMP
    mu = np.exp(logits - logits.max())
    mu = mu / mu.sum()
    np.testing.assert_allclose(float(value), float(mu @ np.asarray(q[:3])), rtol=1e-5)
    # The unoccupied slot's high Q is never read.
    assert float(value) < 0.9
    # temp -> inf recovers E_rho[Q]; a very small temp the set's max.
    np.testing.assert_allclose(
        float(subset_improvement(rho, q, occupied, 1e6)),
        float(rho_set @ np.asarray(q[:3])),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        float(subset_improvement(rho, q, occupied, 1e-4)), 0.2, atol=1e-4
    )


def _depth_two_q(
    code_value: np.ndarray, **stub_kwargs: np.ndarray | float
) -> SearchRoot:
    fns = _stub_fns(code_value, **stub_kwargs)
    root = jax.jit(
        lambda rows, legal: search_root(
            rows, legal, jax.random.key(0), fns, _budget(depth=2)
        )
    )(_rows(), _legal([3]))
    return root


def test_depth_two_backs_up_the_improved_candidates_through_continuation() -> None:
    """Every depth-1 node has the same stub readers, so the root cell's
    value IS B_1 = (1 - c) T + c E_mu[Q] with Q the candidates' code
    values (chance-free here); `deep_gain` reads B_1 - V(h_1)."""
    code_value = np.zeros(NUM_CODES, np.float32)
    code_value[3] = 0.4  # V of the depth-1 node (the root cell's code)
    candidate_codes = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    candidate_values = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.2, 0.05])
    code_value[candidate_codes] = candidate_values
    rho = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
    logits = np.log(rho) + candidate_values / TEMP
    mu = np.exp(logits - logits.max())
    mu = mu / mu.sum()
    improved = float(mu @ candidate_values)
    # c = 1: the improved continuation alone, T never read.
    root = _depth_two_q(
        code_value, candidate_codes=candidate_codes, candidate_rho=rho, terminal=-1.0
    )
    np.testing.assert_allclose(float(root.cell_values[3]), improved, rtol=1e-5)
    np.testing.assert_allclose(float(root.deep_gain), improved - 0.4, rtol=1e-5)
    np.testing.assert_allclose(float(root.candidate_retained_mass), 1.0, rtol=1e-6)
    assert float(root.candidate_occupied) == NUM_CANDIDATES
    # c = 0: the conditional terminal outcome alone, counted once; the
    # node's own V (0.4) is NOT what comes back.
    root = _depth_two_q(
        code_value,
        candidate_codes=candidate_codes,
        candidate_rho=rho,
        continue_prob=0.0,
        terminal=-1.0,
    )
    np.testing.assert_allclose(float(root.cell_values[3]), -1.0, atol=1e-6)
    np.testing.assert_allclose(float(root.deep_continue), 0.0, atol=1e-6)
    # The plan's counterexamples: a node aliasing a 50% terminal win with
    # a 50% continuation worth -1 backs up 0, whatever its V; the
    # asymmetric case c = 0.25, T = -1, continuation +1 backs up -0.5.
    for continue_prob, terminal, continuation, expected in (
        (0.5, 1.0, -1.0, 0.0),
        (0.25, -1.0, 1.0, -0.5),
    ):
        flat = np.zeros(NUM_CODES, np.float32)
        flat[3] = 0.0
        flat[candidate_codes] = continuation
        root = _depth_two_q(
            flat,
            candidate_codes=candidate_codes,
            candidate_rho=rho,
            continue_prob=continue_prob,
            terminal=terminal,
        )
        np.testing.assert_allclose(float(root.cell_values[3]), expected, atol=1e-6)
    # Unoccupied candidates are never read, whatever their value.
    occupied = np.array([True, True, False, False, False, False, False, False])
    root = _depth_two_q(
        code_value,
        candidate_codes=candidate_codes,
        candidate_rho=rho,
        occupied=occupied,
    )
    logits = np.log(rho[:2] / rho[:2].sum()) + candidate_values[:2] / TEMP
    mu = np.exp(logits - logits.max())
    mu = mu / mu.sum()
    np.testing.assert_allclose(
        float(root.cell_values[3]), float(mu @ candidate_values[:2]), rtol=1e-5
    )
    assert float(root.candidate_occupied) == 2.0


def test_diagnostics_read_zero_at_zero_bonus_and_the_kl_of_a_tilt() -> None:
    legal = _legal([2, 5, 8, 11])
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


def _root_with(q: jax.Array, legal: jax.Array) -> SearchRoot:
    zero = jnp.zeros((), jnp.float32)
    return SearchRoot(
        cell_values=q,
        bonus=jnp.where(legal, q / TEMP, 0.0),
        num_legal=legal.sum(),
        legal_truncated=jnp.asarray(False),
        deep_gain=zero,
        deep_continue=zero,
        candidate_retained_mass=zero,
        candidate_occupied=zero,
    )


def test_eval_game_logs_read_real_acted_rows_and_never_search_leaves() -> None:
    from rl.online.main import eval_game_logs

    num_rows = 6
    done = np.zeros(num_rows, bool)
    done[3:] = True  # row 3 is the terminal row; 4-5 copy it (padding)
    action_mask = np.zeros((num_rows, NUM_ACTION_CELLS), bool)
    action_mask[:, NUM_SWITCH_CELLS] = True  # a move is always legal
    action_mask[[0, 2], 0] = True  # rows 0 and 2 offer a switch
    action_index = np.array([0, NUM_SWITCH_CELLS, NUM_SWITCH_CELLS, 0, 0, 0])
    root_kl = np.array([0.1, 0.2, 0.3, 9.0, 9.0, 9.0], np.float32)

    def trajectory(search: SearchOutput) -> Trajectory:
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
    # Two decisions offered a switch (rows 0, 2), one took it; the terminal
    # row and the padding are out.
    assert logs["switch-frac-s"] == 0.5 and logs["ms-per-step-s"] == 500.0
    assert not any(key.startswith("search-") for key in logs)
    # The search eval actor was deleted 2026-09-09: search leaves, even when
    # a trajectory carries them (the offline readers still produce them),
    # are not an eval read.
    searched = SearchOutput(
        root_kl=root_kl,
        search_value=root_kl,
        root_value_gap=root_kl,
        num_legal=root_kl,
        legal_truncated=np.ones(num_rows, bool),
    )
    assert eval_game_logs(trajectory(searched), 2.0, "s") == logs


@pytest.mark.gpu
@pytest.mark.slow
def test_search_moves_the_real_policy_and_the_plain_arm_carries_no_leaves(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
) -> None:
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model

    _, params, actor_input, actor_output = real_model_and_trajectory
    # Per-STEP leaves only: `history` / `packed_history` are per trajectory
    # and a step's gathers name their rows by absolute index.
    actor_input = actor_input.replace(
        env=jax.tree.map(lambda leaf: leaf[:4], actor_input.env)
    )
    actor_output = jax.tree.map(lambda leaf: leaf[:4], actor_output)
    # Fresh params emit all-zero logits and a zero dynamics_out_proj (the
    # copy predictor): open both so the operator has something to read
    # and something to move.
    params = open_zero_init_paths(params, ["action_head", "dynamics_out_proj"])
    plain_cfg = get_player_model_config(9, train=False)
    rngs = {"sampling": jax.random.key(1)}
    plain = jax.jit(get_player_model(plain_cfg).apply)(
        params, actor_input, actor_output, HeadParams(), rngs=rngs
    )
    assert all(isinstance(leaf, tuple) for leaf in dataclasses.astuple(plain.search))
    for depth in (1, 2):
        search_cfg = get_player_model_config(9, train=False)
        search_cfg.search.enabled = True
        search_cfg.search.depth = depth
        # This checks integration, not search strength. Bound both chance
        # axes so depth two fits alongside the shared full-model fixture.
        search_cfg.search.num_samples = 2
        search_cfg.search.num_samples_inner = 2
        searched = jax.jit(get_player_model(search_cfg).apply)(
            params, actor_input, actor_output, HeadParams(), rngs=rngs
        )
        assert searched.search.root_kl.shape == (4,)
        assert np.all(np.asarray(searched.search.root_kl) > 0)
        assert np.all(np.isfinite(np.asarray(searched.search.root_value_gap)))
        assert not np.allclose(
            np.asarray(plain.action_head.entropy),
            np.asarray(searched.action_head.entropy),
        )
        if depth == 1:
            assert isinstance(searched.search.deep_gain, tuple)
        else:
            assert searched.search.deep_gain.shape == (4,)
            assert np.all(np.isfinite(np.asarray(searched.search.deep_gain)))
            assert np.all(np.asarray(searched.search.candidate_occupied) >= 1)
