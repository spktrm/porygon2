"""Search over the latent transition model (2026-09-06, Step 3 of the
stochastic-transition plan): the policy-improvement operator an EVAL actor
applies to the trained policy, so the model is priced by PLAY -- wr(search)
minus wr(no search) on the same checkpoint at the same temperature -- and
not by a reconstruction number.

Rung 1, depth-1 expectimax. For every legal root cell `a` and `num_samples`
chance codes `z ~ p(z | h, a)` from the PRIOR (the rollout-side code
distribution), one imagined step `g(h, a, z)` and the shared value head on
its CLS row: `Q(a) = E_z[V(g(h, a, z))]`. Sampling z from the prior is
expectimax against the EMPIRICAL opponent -- the league mixture the data
was played against -- and every panel that reads this says so. `Q / temp`
is added to the readout's logits on legal cells (Gumbel-MuZero's additive
`logits + sigma(Q)`; Q and Q - V are the same policy under the softmax),
so a bonus of zeros is exactly the trained policy and `root_kl` measures
the operator's size.

Everything here is a free function over callables so a test can hand it a
value head that ranks cells by construction (the positive control) and so
the model's own methods are not re-implemented: `prior_fn`, `action_rows_fn`
and `imagine_fn` are `TransitionModel.prior` / `.action_rows` / `.imagine`,
`value_fn` the model's `v_head`. Shapes are static: `max_cells` root cells
(padded with cell 0 and masked) x `num_samples` codes, one batched g call
per decision.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from rl.model.constants import CLS_ROW
from rl.model.transition import unimix_probs
from rl.model.utils import legal_log_policy


class SearchRoot(NamedTuple):
    """One decision's search read. `q` is Q(a) on legal cells and 0
    elsewhere; `bonus` is what the readout's logits receive (Q / temp on
    legal cells, 0 elsewhere -- illegal cells are already at -1e9)."""

    q: jax.Array
    bonus: jax.Array
    num_legal: jax.Array
    legal_truncated: jax.Array


class SearchDiagnostics(NamedTuple):
    """The panels: KL(pi_search || pi) over legal cells (the operator's
    size), the search policy's expected Q, and its gap to V at the root
    (sign-consistent with the wr delta if the model is calibrated)."""

    root_kl: jax.Array
    search_value: jax.Array
    root_value_gap: jax.Array


def depth_one_expectimax(
    rows: jax.Array,
    row_valid: jax.Array,
    legal: jax.Array,
    rng: jax.Array,
    prior_fn: Callable,
    action_rows_fn: Callable,
    imagine_fn: Callable,
    value_fn: Callable,
    num_samples: int,
    max_cells: int,
    temp: float,
) -> SearchRoot:
    """`rows` (R, D) the post-trunk policy-readable sequence, `row_valid`
    (R,), `legal` (NUM_ACTION_CELLS,) bool. Returns per-cell arrays over the
    full cell space."""
    num_cells = legal.shape[-1]
    cells = jnp.nonzero(legal, size=max_cells, fill_value=0)[0]
    num_legal = legal.sum()
    cell_valid = jnp.arange(max_cells) < num_legal

    prior_logits = jax.vmap(prior_fn, in_axes=(None, None, 0))(rows, row_valid, cells)
    code_probs = unimix_probs(prior_logits)
    code_classes = code_probs.shape[-1]
    samples = jax.random.categorical(
        rng, jnp.log(code_probs), axis=-1, shape=(num_samples, *code_probs.shape[:-1])
    )
    code_one_hot = jax.nn.one_hot(samples, code_classes, dtype=jnp.float32)

    src_rows, tgt_rows = jax.vmap(action_rows_fn, in_axes=(None, 0))(rows, cells)
    imagine_cells = jax.vmap(imagine_fn, in_axes=(None, None, 0, 0, 0))
    imagine_samples = jax.vmap(imagine_cells, in_axes=(None, None, None, None, 0))
    imagined = imagine_samples(rows, row_valid, src_rows, tgt_rows, code_one_hot)
    values = value_fn(imagined[:, :, CLS_ROW, :]).expectation.astype(jnp.float32)
    q_cells = jnp.where(cell_valid, values.mean(axis=0), 0.0)
    # Padded slots point at cell 0 with a zero value: a sum, never a set,
    # so a legal cell 0 keeps its own Q.
    q = jax.ops.segment_sum(q_cells, cells, num_segments=num_cells)
    bonus = jnp.where(legal, q / temp, 0.0)
    return SearchRoot(
        q=q,
        bonus=bonus,
        num_legal=num_legal,
        legal_truncated=num_legal > max_cells,
    )


def search_diagnostics(
    base_logits: jax.Array,
    root: SearchRoot,
    legal: jax.Array,
    root_value: jax.Array,
) -> SearchDiagnostics:
    """`base_logits` are the readout's legal-masked logits BEFORE the
    bonus; the search policy is the softmax after it, exactly the one
    `_score_and_sample` samples from."""
    log_pi = legal_log_policy(base_logits.astype(jnp.float32), legal)
    log_pi_search = legal_log_policy(
        base_logits.astype(jnp.float32) + root.bonus, legal
    )
    pi_search = jnp.where(legal, jnp.exp(log_pi_search), 0.0)
    root_kl = jnp.sum(jnp.where(legal, pi_search * (log_pi_search - log_pi), 0.0))
    search_value = jnp.sum(pi_search * root.q)
    return SearchDiagnostics(
        root_kl=root_kl,
        search_value=search_value,
        root_value_gap=search_value - root_value.astype(jnp.float32),
    )
