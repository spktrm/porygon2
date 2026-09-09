"""Bounded stochastic PUCT over real root actions and imagined latent actions.

Reference: mctx's MuZero PUCT selection (c1=1.25, c2=19652).
Differences: no evaluation Dirichlet noise; raw bounded value units; generated
latent action subsets below the root; a lazy, fixed empirical chance bank per
edge, sampled uniformly on traversal. Opponent behaviour is part of that learned
chance distribution, not an explicit adversarial player. Values always refer to
the same player, so backups do not alternate signs. No equilibrium claim.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core

from rl.model.constants import CLS_ROW
from rl.model.search import SearchFns, SearchRoot, sample_chance


def _guarded_cond(predicate, true_fn, false_fn, operand):
    """Keep singleton-vmap guards lazy without changing batched model maths.

    Closure conversion makes captured model/state arrays explicit operands.
    The selected branch still runs under vmap, preserving its numerical layout.
    Mixed multi-root predicates retain JAX's ordinary masked batching rule.
    This helper is evaluation-only: custom_vmap has no reverse-mode rule.
    """

    def convert(branch):
        # closure_convert only hoists AD-perturbed captures. Constant value
        # supports can still carry a vmap axis, so hoist every capture here.
        closed, output_shape = jax.make_jaxpr(branch, return_shape=True)(operand)
        program = closed.jaxpr
        output_tree = jax.tree.structure(output_shape)

        def call(values, *constants):
            explicit = jax_core.ClosedJaxpr(program, constants)
            leaves = jax_core.jaxpr_as_fun(explicit)(*jax.tree.leaves(values))
            return jax.tree.unflatten(output_tree, leaves)

        return call, closed.consts

    true_call, true_constants = convert(true_fn)
    false_call, false_constants = convert(false_fn)

    def true_branch(arguments):
        values, constants, _ = arguments
        return true_call(values, *constants)

    def false_branch(arguments):
        values, _, constants = arguments
        return false_call(values, *constants)

    def ordinary(condition, arguments):
        return jax.lax.cond(condition, true_branch, false_branch, arguments)

    guarded = jax.custom_batching.custom_vmap(ordinary)

    @guarded.def_vmap
    def batch_rule(axis_size, in_batched, condition, arguments):
        def axis(batched):
            if batched:
                return 0
            return None

        predicate_axis, argument_axes = jax.tree.map(axis, in_batched)
        if axis_size == 1 or not in_batched[0]:
            if in_batched[0]:
                scalar_condition = condition[0]
            else:
                scalar_condition = condition
            output = jax.lax.cond(
                scalar_condition,
                jax.vmap(true_branch, in_axes=(argument_axes,), axis_size=axis_size),
                jax.vmap(false_branch, in_axes=(argument_axes,), axis_size=axis_size),
                arguments,
            )
        else:
            output = jax.vmap(ordinary, in_axes=(predicate_axis, argument_axes))(
                condition, arguments
            )
        return output, jax.tree.map(lambda value: True, output)

    return guarded(predicate, (operand, true_constants, false_constants))


class MCTSResult(NamedTuple):
    """model_calls counts accepted expansions, not masked work under vmap."""

    root: SearchRoot
    visits: jax.Array
    model_calls: jax.Array
    depth_reached: jax.Array


class Tree(NamedTuple):
    rows: jax.Array
    children: jax.Array
    visits: jax.Array
    totals: jax.Array
    priors: jax.Array
    codes: jax.Array
    valid: jax.Array
    values: jax.Array
    continuation: jax.Array
    terminal: jax.Array
    retained: jax.Array
    used: jax.Array
    depth_reached: jax.Array


def puct_action(priors, totals, visits, valid, parent_value):
    """PUCT in the critic's [-1, 1] units; unvisited Q uses parent V."""
    value = jnp.where(visits > 0, totals / jnp.maximum(visits, 1), parent_value)
    parent_visits = visits.sum()
    coefficient = 1.25 + jnp.log((parent_visits + 19652.0 + 1.0) / 19652.0)
    exploration = coefficient * priors * jnp.sqrt(parent_visits + 1.0) / (visits + 1)
    return jnp.argmax(jnp.where(valid, value + exploration, -jnp.inf))


def mcts_root(
    rows,
    legal,
    base_logits,
    rng,
    fns: SearchFns,
    *,
    simulations=64,
    depth=2,
    chance_samples=4,
    max_actions=16
):
    """One new state per simulation; at most simulations+1 resident states.

    Each edge has chance_samples equally weighted slots. An empty slot draws
    (root latent action, chance) or (interior latent action, chance) once and
    caches that successor. Slot selection NEVER depends on its value. Repeated
    visits refine decisions in that empirical stochastic tree. This budget
    limits dynamics evaluations to at most one per simulation per root, even
    when batched guards execute masked work. model_calls counts accepted cached
    expansions. More simulations do not remove the finite chance-bank
    approximation. Depth counts transitions, not chance nodes.

    Return root visit policy as an additive correction to the existing sampler.
    Every legal root action receives one initial simulation. Overflow falls
    back to the base policy, with zero accepted expansions and a diagnostic.
    """
    if simulations < max_actions or depth < 1 or chance_samples < 1:
        raise ValueError(
            "MCTS needs simulations >= max_actions, depth >= 1 and chance_samples >= 1"
        )
    cells = jnp.nonzero(legal, size=max_actions, fill_value=0)[0]
    num_legal = legal.sum()
    occupied = jnp.arange(max_actions) < num_legal
    overflow = num_legal > max_actions
    root_value = fns.value_fn(rows[CLS_ROW]).astype(jnp.float32)
    capacity = simulations + 1
    action_probs = jax.vmap(fns.encoder_fn, (None, 0))(rows, cells)
    num_codes = action_probs.shape[-1]
    root_priors = jax.nn.softmax(jnp.where(occupied, base_logits[cells], -jnp.inf))
    root_priors = jnp.where(occupied, root_priors, 0)
    tree = Tree(
        rows=jnp.zeros((capacity, *rows.shape), rows.dtype).at[0].set(rows),
        children=jnp.full((capacity, max_actions, chance_samples), -1, jnp.int32),
        visits=jnp.zeros((capacity, max_actions), jnp.int32),
        totals=jnp.zeros((capacity, max_actions), jnp.float32),
        priors=jnp.zeros((capacity, max_actions), jnp.float32).at[0].set(root_priors),
        codes=jnp.zeros((capacity, max_actions), jnp.int32).at[0].set(cells),
        valid=jnp.zeros((capacity, max_actions), bool).at[0].set(occupied),
        values=jnp.zeros(capacity, jnp.float32).at[0].set(root_value),
        continuation=jnp.ones(capacity, jnp.float32),
        terminal=jnp.zeros(capacity, jnp.float32),
        retained=jnp.zeros(capacity, jnp.float32),
        used=jnp.asarray(1, jnp.int32),
        depth_reached=jnp.asarray(0, jnp.int32),
    )

    def simulation(simulation_index, tree):
        simulation_key = jax.random.fold_in(rng, simulation_index)
        path_nodes = jnp.zeros(depth, jnp.int32)
        path_actions = jnp.zeros(depth, jnp.int32)

        # Read-only traversal: never evaluate the model inside this loop.
        def traverse(carry):
            node, level, stopped, path_nodes, path_actions, _, _ = carry
            slot_key = jax.random.split(jax.random.fold_in(simulation_key, level), 4)[0]
            selected = puct_action(
                tree.priors[node],
                tree.totals[node],
                tree.visits[node],
                tree.valid[node],
                tree.values[node],
            )
            selected = jnp.where(
                (node == 0) & (simulation_index < num_legal), simulation_index, selected
            )
            chance_slot = jax.random.randint(slot_key, (), 0, chance_samples)
            child = tree.children[node, selected, chance_slot]
            missing = child < 0
            child = jnp.where(missing, tree.used, child)
            path_nodes = path_nodes.at[level].set(node)
            path_actions = path_actions.at[level].set(selected)
            stopped = (
                missing | (tree.continuation[child] == 0) | ~tree.valid[child].any()
            )
            return (
                child,
                level + 1,
                stopped,
                path_nodes,
                path_actions,
                chance_slot,
                missing,
            )

        leaf, length, _, path_nodes, path_actions, chance_slot, missing = (
            jax.lax.while_loop(
                lambda carry: (carry[1] < depth) & ~carry[2],
                traverse,
                (
                    jnp.asarray(0, jnp.int32),
                    jnp.asarray(0, jnp.int32),
                    jnp.asarray(False),
                    path_nodes,
                    path_actions,
                    jnp.asarray(0, jnp.int32),
                    jnp.asarray(False),
                ),
            )
        )
        level = length - 1
        node = path_nodes[level]
        selected = path_actions[level]
        child = leaf
        _, action_key, chance_key, generate_key = jax.random.split(
            jax.random.fold_in(simulation_key, level), 4
        )

        def expand(arguments):
            (
                tree,
                node,
                selected,
                chance_slot,
                child,
                level,
                action_key,
                chance_key,
                generate_key,
                action_probs,
            ) = arguments
            source = tree.rows[node]
            root_code = jax.random.categorical(
                action_key, jnp.log(action_probs[selected])
            )
            code = jnp.where(node == 0, root_code, tree.codes[node, selected])
            action = jax.nn.one_hot(code, num_codes, dtype=jnp.float32)
            chance = sample_chance(fns.prior_fn(source, action), chance_key, 1)[0]
            successor = fns.imagine_fn(source, action, chance)
            value = fns.value_fn(successor[CLS_ROW]).astype(jnp.float32)
            continuation = fns.continue_fn(successor).astype(jnp.float32)
            terminal = fns.terminal_fn(successor).astype(jnp.float32)

            def candidates_at_node(arguments):
                successor, generate_key = arguments
                candidates = fns.generate_fn(successor, generate_key)
                padding = max_actions - candidates.codes.shape[0]
                if padding < 0:
                    raise ValueError("MCTS candidate width exceeds max_actions")
                valid = jnp.pad(candidates.occupied, (0, padding))
                codes = jnp.pad(candidates.codes, (0, padding))
                weights = jnp.pad(
                    jnp.where(candidates.occupied, candidates.rho_at_codes, 0),
                    (0, padding),
                )
                priors = weights / jnp.maximum(weights.sum(), 1e-8)
                return codes, valid, priors, candidates.retained_mass

            def no_candidates(_):
                return (
                    jnp.zeros(max_actions, jnp.int32),
                    jnp.zeros(max_actions, bool),
                    jnp.zeros(max_actions, jnp.float32),
                    jnp.zeros((), jnp.float32),
                )

            if depth == 1:
                codes, valid, priors, retained = no_candidates(None)
            else:
                codes, valid, priors, retained = _guarded_cond(
                    (level + 1 < depth) & (continuation > 0),
                    candidates_at_node,
                    no_candidates,
                    operand=(successor, generate_key),
                )
            return tree._replace(
                rows=tree.rows.at[child].set(successor),
                children=tree.children.at[node, selected, chance_slot].set(child),
                priors=tree.priors.at[child].set(priors),
                codes=tree.codes.at[child].set(codes),
                valid=tree.valid.at[child].set(valid),
                values=tree.values.at[child].set(
                    jnp.where(continuation == 0, terminal, value)
                ),
                continuation=tree.continuation.at[child].set(continuation),
                terminal=tree.terminal.at[child].set(terminal),
                retained=tree.retained.at[child].set(retained),
                used=tree.used + 1,
                depth_reached=jnp.maximum(tree.depth_reached, level + 1),
            )

        tree = _guarded_cond(
            missing,
            expand,
            lambda arguments: arguments[0],
            (
                tree,
                node,
                selected,
                chance_slot,
                child,
                level,
                action_key,
                chance_key,
                generate_key,
                action_probs,
            ),
        )

        def backup(offset, carry):
            tree, value = carry
            index = length - 1 - offset
            node = path_nodes[index]
            action = path_actions[index]
            tree = tree._replace(
                visits=tree.visits.at[node, action].add(1),
                totals=tree.totals.at[node, action].add(value),
            )
            # Mix terminal payoff ONCE at this node, before backing to its parent.
            value = (1 - tree.continuation[node]) * tree.terminal[
                node
            ] + tree.continuation[node] * value
            return tree, value

        tree, _ = jax.lax.fori_loop(0, length, backup, (tree, tree.values[leaf]))
        return tree

    tree = jax.lax.cond(
        (num_legal > 0) & ~overflow,
        lambda initial: jax.lax.fori_loop(0, simulations, simulation, initial),
        lambda initial: initial,
        tree,
    )
    counts = jnp.where(occupied, tree.visits[0], 0)
    q_cells = jnp.where(occupied, tree.totals[0] / jnp.maximum(counts, 1), 0)
    action_values = jax.ops.segment_sum(q_cells, cells, num_segments=legal.shape[-1])
    visits = jax.ops.segment_sum(counts, cells, num_segments=legal.shape[-1])
    log_visits = jnp.log(jnp.maximum(visits, 1).astype(jnp.float32))
    bonus = jnp.where(legal & ~overflow, log_visits - base_logits, 0)
    interior = (
        (jnp.arange(capacity) > 0)
        & (jnp.arange(capacity) < tree.used)
        & tree.valid.any(axis=-1)
    )
    denominator = jnp.maximum(interior.sum(), 1)
    root = SearchRoot(
        cell_values=action_values,
        bonus=bonus,
        num_legal=num_legal,
        legal_truncated=overflow,
        deep_gain=jnp.zeros((), jnp.float32),
        deep_continue=jnp.where(interior, tree.continuation, 0).sum() / denominator,
        candidate_retained_mass=jnp.where(interior, tree.retained, 0).sum()
        / denominator,
        candidate_occupied=jnp.where(interior, tree.valid.sum(axis=-1), 0).sum()
        / denominator,
    )
    return MCTSResult(root, visits, tree.used - 1, tree.depth_reached)
