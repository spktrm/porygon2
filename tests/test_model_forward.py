"""Real player-model init + forward on the bundled example trajectory.

Marked gpu: runs wherever JAX puts it (the training box GPU, with
preallocation disabled by conftest so it coexists with a live learner).
Marked slow (~1 min): deselect with `-m "not slow"` for the quick suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import open_zero_init_paths

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_init_produces_finite_params(real_model_and_trajectory, real_model_apply):
    _, params, _, _ = real_model_and_trajectory
    leaves = jax.tree.leaves(params)
    assert leaves
    for leaf in leaves:
        assert np.isfinite(np.asarray(leaf, dtype=np.float32)).all()


def test_forward_outputs_finite_and_shaped(real_model_and_trajectory, real_model_apply):
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    out = real_model_apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]

    log_probs = np.asarray(out.value_head.log_probs, dtype=np.float32)
    assert log_probs.shape[0] == T
    assert np.isfinite(log_probs).all()
    # log_probs is a categorical distribution over the value support.
    np.testing.assert_allclose(np.exp(log_probs).sum(-1), 1.0, atol=1e-3)

    pi_lp = np.asarray(out.action_head.log_prob, dtype=np.float32)
    assert np.isfinite(pi_lp).all()


def test_q_head_forward_shapes(real_model_and_trajectory, real_model_apply):
    """The structural two-rung hierarchical Q readout (docs/
    q-critic-plan.md): owned adapter + shared MacroMicroHead params in the
    tree, (T, A, n_bins) logits per rung, rung conditioning alive."""

    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    for subtree in ("policy_head", "advantage_head", "v_head"):
        assert subtree in params["params"]

    out = real_model_apply(params, actor_input, actor_output, HeadParams())
    T = actor_input.env.done.shape[0]
    A = int(np.prod(actor_input.env.action_mask.shape[-2:]))
    adv = np.asarray(out.advantage, dtype=np.float32)
    assert adv.shape == (T, A)
    assert np.isfinite(adv).all()
    # At init the head is identically ZERO (e00a388's flat-at-init
    # contract zero-inits every advantage output path), so any test of
    # live geometry has to open the zero paths first — otherwise it
    # passes vacuously.
    opened = open_zero_init_paths(params, ("advantage_head",))
    out_open = real_model_apply(opened, actor_input, actor_output, HeadParams())
    adv_open = np.asarray(out_open.advantage, dtype=np.float32)
    assert np.isfinite(adv_open).all()
    assert adv_open.any()
    # Full-support log_policy is present in train mode — the Retrace
    # target's expectation bootstrap depends on it.
    assert np.asarray(out.action_head.log_policy).shape[-1] == A


def test_q_head_is_flat_at_init_and_local_routes_get_gradient(
    real_model_and_trajectory, real_model_apply
):
    """The flat-at-init contract (every Q cell exactly 0) AND the
    regression test for the 2026-08-24 finding: the within-modality
    route must be a single zero-init factor, so a within-modality
    signal reaches its kernels at init WITHOUT the pointer gate having
    moved. Loss = sum over legal cells of advantage * (+-1 pattern by rank
    parity within each modality) — pure within-modality by construction."""
    from rl.environment.data import FLAT_MODALITY_MASK
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    out = real_model_apply(params, actor_input, actor_output, HeadParams())
    assert not np.asarray(out.advantage, dtype=np.float32).any()

    flat = np.asarray(FLAT_MODALITY_MASK)
    pattern = np.zeros(flat.shape, np.float32)
    for mod in np.unique(flat):
        cells = np.flatnonzero(flat == mod)
        pattern[cells] = np.where(np.arange(len(cells)) % 2 == 0, 1.0, -1.0)
    legal = np.asarray(actor_input.env.action_mask).reshape(-1, flat.shape[0])

    def loss(p):
        o = real_model_apply(p, actor_input, actor_output, HeadParams())
        return jnp.sum(
            o.advantage.astype(jnp.float32) * jnp.asarray(pattern) * jnp.asarray(legal)
        )

    grads = jax.grad(loss)(params)["params"]["advantage_head"]["macro_micro"]
    micro_grads = grads["micro"]
    for name in ("micro_local_src", "micro_local_tgt"):
        g = np.asarray(micro_grads[name]["kernel"], dtype=np.float32)
        assert np.isfinite(g).all() and np.abs(g).max() > 0.0, name
    # The gated grid's q/k projections get NOTHING at init — the two-factor
    # product structure this test exists to document. The local routes above
    # are what carry the early gradient.
    for name in ("Dense_0", "Dense_1"):
        assert not np.asarray(
            micro_grads["micro_qk"][name]["kernel"], dtype=np.float32
        ).any(), name


@pytest.mark.gpu
@pytest.mark.slow
def test_micro_params_are_not_shared_between_slot_groups(
    real_model_and_trajectory, real_model_apply
):
    """The separation contract (2026-08-25).

    A loss that touches ONLY one slot group's cells must leave every other
    group's parameters bitwise untouched. Before this change the three groups
    shared one projection and one pair of local kernels, distinguished only
    by a scalar — and on the 84.9k-step run the target group's scalar was
    still exactly zero, i.e. that group had no trained readout at all.

    The per-group blocks live in disjoint COLUMN ranges of the shared Dense
    kernels (attention heads own disjoint output coordinates), so the
    assertion is per-column-block, not per-leaf.
    """
    from rl.environment.data import FLAT_SRC_GROUP_MASK, NUM_ACTION_SLOT_GROUPS
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    flat_group = np.asarray(FLAT_SRC_GROUP_MASK)
    legal = np.asarray(actor_input.env.action_mask).reshape(-1, flat_group.shape[0])
    # Differentiate at OPENED head params: at init every advantage output
    # path is zero, so d loss / d action_embeddings ≡ 0 and the encoder
    # half of this test would pass vacuously (its positive control caught
    # exactly this, 2026-08-27). The column/row isolation asserted below
    # is structural, so it holds at any params.
    opened = open_zero_init_paths(params, ("advantage_head",))

    def grads_for_group(g):
        # +-1 by rank parity WITHIN the group (a uniform weight lands in
        # the null space of the within-modality mean-centring and the loss
        # is identically zero), then BALANCED to a zero sum per row over
        # the group's legal cells: compose_q's advantage is pi-centred
        # over ALL legal cells (A - E_sg(pi)[A]), so any row weight with a
        # non-zero sum rides that centring into every other modality's
        # cells — a property of the Q identity, not a parameter leak, and
        # exactly the direction this privacy test must project out
        # (measured 2026-08-27: unbalanced parity put |grad| 4.3 on the
        # move decoder through the centring; balanced puts exactly 0).
        cells = flat_group == g
        parity = np.where(np.arange(flat_group.shape[0]) % 2 == 0, 1.0, -1.0)
        weight = (parity * cells)[None, :].astype(np.float32) * legal
        row_count = np.maximum((cells[None, :] * legal).sum(-1, keepdims=True), 1)
        weight = (
            weight
            - (weight.sum(-1, keepdims=True) / row_count) * cells[None, :] * legal
        )

        def loss(p):
            o = real_model_apply(p, actor_input, actor_output, HeadParams())
            return jnp.sum(o.advantage.astype(jnp.float32) * jnp.asarray(weight))

        grads = jax.grad(loss)(opened)["params"]
        return grads["advantage_head"]["macro_micro"]["micro"], grads["encoder"]

    target_group = 1  # switch
    g, encoder_grads = grads_for_group(target_group)

    # Encoder-level separation (2026-08-27): the per-group action decoders
    # are PRIVATE computation, so a switch-only loss must leave the move
    # and target decoders' parameters with exactly zero gradient — and the
    # switch decoder must receive some (the positive control, which is why
    # the loss runs at opened head params — see above).
    group_names = ("move", "switch", "target")
    for name in group_names:
        decoder_grads = np.concatenate(
            [
                np.asarray(leaf, dtype=np.float32).ravel()
                for leaf in jax.tree.leaves(encoder_grads[f"{name}_action_decoder"])
            ]
        )
        if name == group_names[target_group]:
            assert decoder_grads.any(), "switch decoder got no gradient"
        else:
            assert not decoder_grads.any(), f"{name} decoder leaked"

    # Local routes: kernel is (d, G) — exactly one column may be non-zero.
    for name in ("micro_local_src", "micro_local_tgt"):
        k = np.asarray(g[name]["kernel"], dtype=np.float32)
        assert k.shape[-1] == NUM_ACTION_SLOT_GROUPS, name
        for other in range(NUM_ACTION_SLOT_GROUPS):
            if other == target_group:
                continue
            assert not k[..., other].any(), f"{name}: group {other} leaked"

    # type_scale is (G, K) — same story, one row.
    ts = np.asarray(g["type_scale"], dtype=np.float32)
    for other in range(NUM_ACTION_SLOT_GROUPS):
        if other == target_group:
            continue
        assert not ts[other].any(), f"type_scale: group {other} leaked"

    # Positive control: the group we DID touch must receive gradient
    # somewhere, or the whole test passes vacuously.
    touched = any(
        np.asarray(g[name]["kernel"], dtype=np.float32)[..., target_group].any()
        for name in ("micro_local_src", "micro_local_tgt")
    )
    assert touched, "positive control: the touched group got no gradient"


def test_action_decoders_route_species_to_own_cell(
    real_model_and_trajectory, real_model_apply
):
    """Wiring test for the per-group action decoders (2026-08-27),
    following the latent-read pattern: live at init — perturbing one legal
    reserve's SPECIES token must move that reserve's own switch cells more
    than its siblings' (the advantage head's zero-init paths opened, else
    the composition is exactly 0 and everything passes vacuously). own > 0
    is the live check; own > 1.5x sibling is the separation claim."""
    import dataclasses

    from rl.environment.data import FLAT_MODALITY_MASK, RESERVE_ENTITY_INDICES
    from rl.environment.protos.features_pb2 import EntityPrivateNodeFeature
    from rl.environment.protos.service_pb2 import ModalityEnum
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    opened = open_zero_init_paths(params, ("advantage_head",))
    species_col = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
    num_cells = np.asarray(FLAT_MODALITY_MASK).shape[0]
    A = int(np.sqrt(num_cells))
    switch_cells = np.asarray(FLAT_MODALITY_MASK) == ModalityEnum.MODALITY_ENUM__SWITCH
    cell_tgt = np.arange(num_cells) % A

    env = actor_input.env
    flat_mask = np.asarray(env.action_mask).reshape(env.done.shape[0], -1)
    rows_ok = np.asarray(~env.done)
    # Two reserves that are both legal battle-switch targets somewhere.
    legal_counts = [
        ((flat_mask & (switch_cells & (cell_tgt == slot))).any(-1) & rows_ok).sum()
        for slot in RESERVE_ENTITY_INDICES
    ]
    ranked = np.argsort(legal_counts)[::-1]
    first, second = int(ranked[0]), int(ranked[1])
    assert legal_counts[first] > 0 and legal_counts[second] > 0

    team = np.asarray(env.private_team).copy()
    team[:, [first, second], species_col] = team[:, [second, first], species_col]
    swapped_input = dataclasses.replace(
        actor_input, env=dataclasses.replace(env, private_team=jnp.asarray(team))
    )

    base = real_model_apply(opened, actor_input, actor_output, HeadParams())
    swapped = real_model_apply(opened, swapped_input, actor_output, HeadParams())
    delta = np.abs(
        np.asarray(swapped.advantage, np.float32)
        - np.asarray(base.advantage, np.float32)
    )

    own_cells = switch_cells & np.isin(
        cell_tgt, [RESERVE_ENTITY_INDICES[first], RESERVE_ENTITY_INDICES[second]]
    )
    own_deltas, sibling_deltas = [], []
    for t in np.nonzero(rows_ok)[0]:
        legal = flat_mask[t]
        own = legal & own_cells
        sibling = legal & switch_cells & ~own_cells
        if own.any() and sibling.any():
            own_deltas.append(delta[t][own].mean())
            sibling_deltas.append(delta[t][sibling].mean())
    assert own_deltas, "no rows with both own and sibling legal switch cells"
    own_mean, sibling_mean = np.mean(own_deltas), np.mean(sibling_deltas)
    # Live at init AND separated: the perturbed reserves' own cells respond,
    # and more strongly than their unperturbed siblings.
    assert own_mean > 0.0
    assert own_mean > 1.5 * sibling_mean, (own_mean, sibling_mean)


def test_forward_is_deterministic(real_model_and_trajectory, real_model_apply):
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    a = real_model_apply(params, actor_input, actor_output, HeadParams())
    b = real_model_apply(params, actor_input, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(a.value_head.log_probs, dtype=np.float32),
        np.asarray(b.value_head.log_probs, dtype=np.float32),
    )
