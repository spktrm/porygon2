"""Search over the latent transition model (stochastic-transition Step 3,
2026-09-06; recursive with latent actions 2026-09-07).

The root expands its EXACT legal cells: each cell's latent action is
drawn from the action encoder q(u | h_0, a), each draw's chance code from
the prior p(z | h_0, u), and the successor `g(h_0, u, z)` is valued by the
leaf operator. Every deeper node generates its own latent candidates from
its imagined state (the candidate generator, without replacement inside
its support), draws chance from the prior again and backs up the explicit
decision / chance recursion

    Q_d(h, u) = E_{z ~ p(. | h, u)} B_{d-1}(g(h, u, z)),
    B_d(h)    = (1 - c(h)) T(h) + c(h) sum_u mu_d(u | h) Q_d(h, u),
    B_0(h)    = V(h),

with c the predicted continuation, T the CONDITIONAL terminal outcome
(E[outcome | the node ends the game] -- V is unconditional and cannot
stand in for it at a node that may be terminal), and mu the KL-regularised
improvement of the generator's first-conditional prior restricted to the
occupied candidates: mu = softmax(log rho_C + Q / temp), the maximiser of
E_mu[Q] - temp KL(mu || rho_C) over the set. Chance is AVERAGED, never
maximised; the backup is E_mu[Q], the expected return. This is
conditional-on-subset planning: no Sampled-MuZero inclusion correction is
claimed. At an observed non-terminal root c = 1 and the immediate reward
is 0, so the root's per-cell score is E_{u ~ q(. | h_0, a)} Q_D(h_0, u)
and the bonus the readout's logits receive is Q(a) / temp on the legal
cells (Gumbel-MuZero's additive form; a bonus of zeros is exactly the
trained policy and `root_kl` measures the operator's size). A root with
more legal cells than the static width returns the base policy untouched
and is counted (`legal_truncated`), never silently truncated.

Free functions over callables (`SearchFns`) so the unit tests run the
operator on stubs whose values are set by hand; the parent model binds
the transition model's methods and the shared `v_head`.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from rl.model.constants import CLS_ROW
from rl.model.transition import Candidates, unimix_probs
from rl.model.utils import legal_log_policy


class SearchFns(NamedTuple):
    """The model's node interface. `encoder_fn(rows, cell) -> (C,)` the
    unimix'd action PROBABILITIES (so a stub can hand the root an exact
    code); `prior_fn(rows, action_one_hot) -> (G, K)` logits;
    `imagine_fn(rows, action_one_hot, code_one_hot) -> rows`;
    `value_fn(cls_row) -> ()` V; `generate_fn(rows, rng) -> Candidates`;
    `continue_fn(rows) -> ()` c; `terminal_fn(rows) -> ()` T."""

    encoder_fn: Callable
    prior_fn: Callable
    imagine_fn: Callable
    value_fn: Callable
    generate_fn: Callable
    continue_fn: Callable
    terminal_fn: Callable


class SearchBudget(NamedTuple):
    """The static shape budget: `depth` (1 = the root's successors are
    leaves), `num_samples` chance draws per root cell, `num_samples_inner`
    per deeper candidate, `max_cells` the legal set's static width, and
    `temp` for both the root bonus and the deeper improvement."""

    depth: int
    num_samples: int
    num_samples_inner: int
    max_cells: int
    temp: float


class SearchRoot(NamedTuple):
    """`q` and `bonus` are (NUM_ACTION_CELLS,): the per-cell value (0 off
    the legal set) and what the readout's logits receive (Q / temp on the
    legal cells; all zero when the root overflowed). The deep leaves are
    per-decision reads of the depth-1 nodes (0 at depth 1)."""

    q: jax.Array
    bonus: jax.Array
    num_legal: jax.Array
    legal_truncated: jax.Array
    deep_gain: jax.Array
    deep_continue: jax.Array
    candidate_retained_mass: jax.Array
    candidate_occupied: jax.Array


class SearchDiagnostics(NamedTuple):
    """`root_kl` KL(pi_search || pi) over the legal cells; `search_value`
    E_{pi_search}[q]; `root_value_gap` that minus V at the root."""

    root_kl: jax.Array
    search_value: jax.Array
    root_value_gap: jax.Array


class NodeBackup(NamedTuple):
    value: jax.Array
    leaf_value: jax.Array
    continue_prob: jax.Array
    retained_mass: jax.Array
    occupied: jax.Array


def sample_chance(
    prior_logits: jax.Array, rng: jax.Array, num_samples: int
) -> jax.Array:
    """`num_samples` one-hot chance codes (S, G, K) from the unimix'd
    prior."""
    probs = unimix_probs(prior_logits)
    samples = jax.random.categorical(
        rng, jnp.log(probs), axis=-1, shape=(num_samples, *probs.shape[:-1])
    )
    return jax.nn.one_hot(samples, probs.shape[-1], dtype=jnp.float32)


def chance_backup(
    rows: jax.Array,
    action_one_hot: jax.Array,
    rng: jax.Array,
    fns: SearchFns,
    num_samples: int,
    leaf_fn: Callable,
) -> jax.Array:
    """Q(h, u) = E_z leaf(g(h, u, z)) over `num_samples` prior draws."""
    codes = sample_chance(fns.prior_fn(rows, action_one_hot), rng, num_samples)
    successors = jax.vmap(fns.imagine_fn, in_axes=(None, None, 0))(
        rows, action_one_hot, codes
    )
    return jax.vmap(leaf_fn)(successors).mean()


def subset_improvement(
    rho_at_codes: jax.Array, q_values: jax.Array, occupied: jax.Array, temp: float
) -> jax.Array:
    """mu = softmax(log rho_C + Q / temp) over the occupied candidates,
    rho_C the first-conditional prior renormalised over the set; returns
    E_mu[Q] (the expected return, not the regularised value)."""
    log_rho = jnp.log(jnp.maximum(rho_at_codes, 1e-8))
    logits = jnp.where(occupied, log_rho + q_values / temp, -1e9)
    mu = jax.nn.softmax(logits)
    return jnp.sum(mu * q_values, where=occupied)


def node_backup(
    rows: jax.Array, rng: jax.Array, fns: SearchFns, budget: SearchBudget
) -> NodeBackup:
    """B_1(h) at a depth-1 node: the generator's candidates, each valued
    by the chance mean of V one step on, improved over the set and backed
    up through the continuation and the conditional terminal outcome."""
    generate_key, chance_key = jax.random.split(rng)
    candidates: Candidates = fns.generate_fn(rows, generate_key)
    num_codes = candidates.support_mask.shape[-1]
    action_one_hot = jax.nn.one_hot(candidates.codes, num_codes, dtype=jnp.float32)
    chance_keys = jax.random.split(chance_key, candidates.codes.shape[0])

    def leaf(successor):
        return fns.value_fn(successor[CLS_ROW]).astype(jnp.float32)

    q_values = jax.vmap(
        lambda action, key: chance_backup(
            rows, action, key, fns, budget.num_samples_inner, leaf
        )
    )(action_one_hot, chance_keys)
    improved = subset_improvement(
        candidates.rho_at_codes, q_values, candidates.occupied, budget.temp
    )
    continue_prob = fns.continue_fn(rows).astype(jnp.float32)
    terminal = fns.terminal_fn(rows).astype(jnp.float32)
    value = (1.0 - continue_prob) * terminal + continue_prob * improved
    return NodeBackup(
        value=value,
        leaf_value=leaf(rows),
        continue_prob=continue_prob,
        retained_mass=candidates.retained_mass,
        occupied=candidates.occupied.sum().astype(jnp.float32),
    )


def search_root(
    rows: jax.Array,
    legal: jax.Array,
    rng: jax.Array,
    fns: SearchFns,
    budget: SearchBudget,
) -> SearchRoot:
    """`rows` (R, D) the post-trunk policy-readable sequence at an
    observed non-terminal root, `legal` (NUM_ACTION_CELLS,) bool."""
    num_cells = legal.shape[-1]
    cells = jnp.nonzero(legal, size=budget.max_cells, fill_value=0)[0]
    num_legal = legal.sum()
    cell_valid = jnp.arange(budget.max_cells) < num_legal
    legal_truncated = num_legal > budget.max_cells
    action_key, chance_key, deep_key = jax.random.split(rng, 3)

    action_probs = jax.vmap(fns.encoder_fn, in_axes=(None, 0))(rows, cells)
    num_codes = action_probs.shape[-1]
    actions = jax.random.categorical(
        action_key,
        jnp.log(action_probs),
        axis=-1,
        shape=(budget.num_samples, budget.max_cells),
    )
    action_one_hot = jax.nn.one_hot(actions, num_codes, dtype=jnp.float32)
    prior_logits = jax.vmap(
        jax.vmap(fns.prior_fn, in_axes=(None, 0)), in_axes=(None, 0)
    )(rows, action_one_hot)
    chance_keys = jax.random.split(chance_key, (budget.num_samples, budget.max_cells))
    code_one_hot = jax.vmap(
        jax.vmap(lambda logits, key: sample_chance(logits, key, 1)[0])
    )(prior_logits, chance_keys)
    imagine_cells = jax.vmap(fns.imagine_fn, in_axes=(None, 0, 0))
    imagined = jax.vmap(imagine_cells, in_axes=(None, 0, 0))(
        rows, action_one_hot, code_one_hot
    )
    leaf_values = jax.vmap(jax.vmap(lambda node: fns.value_fn(node[CLS_ROW])))(
        imagined
    ).astype(jnp.float32)
    deep_gain = jnp.zeros((), jnp.float32)
    deep_continue = jnp.zeros((), jnp.float32)
    retained_mass = jnp.zeros((), jnp.float32)
    occupied = jnp.zeros((), jnp.float32)
    values = leaf_values
    if budget.depth >= 2:
        deep_keys = jax.random.split(deep_key, (budget.num_samples, budget.max_cells))
        backups = jax.vmap(
            jax.vmap(lambda node, key: node_backup(node, key, fns, budget))
        )(imagined, deep_keys)
        values = backups.value
        node_valid = jnp.broadcast_to(cell_valid[None], values.shape)
        deep_gain = jnp.mean(backups.value - backups.leaf_value, where=node_valid)
        deep_continue = jnp.mean(backups.continue_prob, where=node_valid)
        retained_mass = jnp.mean(backups.retained_mass, where=node_valid)
        occupied = jnp.mean(backups.occupied, where=node_valid)
    q_cells = jnp.where(cell_valid, values.mean(axis=0), 0.0)
    # Padded slots point at cell 0 with a zero value: a sum, never a set,
    # so a legal cell 0 keeps its own Q.
    q = jax.ops.segment_sum(q_cells, cells, num_segments=num_cells)
    bonus = jnp.where(legal & jnp.logical_not(legal_truncated), q / budget.temp, 0.0)
    return SearchRoot(
        q=q,
        bonus=bonus,
        num_legal=num_legal,
        legal_truncated=legal_truncated,
        deep_gain=deep_gain,
        deep_continue=deep_continue,
        candidate_retained_mass=retained_mass,
        candidate_occupied=occupied,
    )


def search_diagnostics(
    base_logits: jax.Array, root: SearchRoot, legal: jax.Array, root_value: jax.Array
) -> SearchDiagnostics:
    """The read of one decision. `base_logits` are the readout's own (pre-
    bonus, at the actor's temp) logits over the cell space; the search
    policy is the softmax after the bonus, exactly the one the actor
    samples from."""
    log_pi = legal_log_policy(base_logits.astype(jnp.float32), legal)
    log_pi_search = legal_log_policy(
        base_logits.astype(jnp.float32) + root.bonus, legal
    )
    pi_search = jnp.where(legal, jnp.exp(log_pi_search), 0.0)
    root_kl = jnp.sum(pi_search * (log_pi_search - log_pi), where=legal)
    search_value = jnp.sum(pi_search * root.q, where=legal)
    return SearchDiagnostics(
        root_kl=root_kl,
        search_value=search_value,
        root_value_gap=search_value - root_value.astype(jnp.float32),
    )


def configure_search(
    config, *, mode="plain", depth=2, simulations=64, chance_samples=4
):
    """Shared offline/interactive search configuration; plain is an exact off mode."""
    if mode not in ("plain", "expectimax", "mcts"):
        raise ValueError("search mode must be plain, expectimax or mcts")
    if depth < 1 or chance_samples < 1:
        raise ValueError("search depth and chance samples must be positive")
    if mode == "expectimax" and depth > 2:
        raise ValueError("expectimax supports depth one or two")
    if mode == "mcts" and simulations < config.search.max_cells:
        raise ValueError("MCTS simulations must cover every root action slot")
    config.search.enabled = mode != "plain"
    if mode != "plain":
        config.search.method = mode
    config.search.depth = depth
    config.search.mcts_simulations = simulations
    config.search.mcts_chance_samples = chance_samples
    return config
