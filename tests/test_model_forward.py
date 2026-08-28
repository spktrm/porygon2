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

    def grads_for_group(g):
        # +-1 by rank parity WITHIN the group, not a uniform +1: micro is
        # mean-centred over each modality's legal cells (compose_action_grid,
        # reduce="mean") and A is pi-centred over the row, so a uniform
        # weight lands exactly in the null space of both and the loss is
        # identically zero. This is the same trick the flat-at-init test
        # above uses, and forgetting it makes the test pass vacuously.
        cells = np.flatnonzero(flat_group == g)
        sel = np.zeros(flat_group.shape, np.float32)
        sel[cells] = np.where(np.arange(len(cells)) % 2 == 0, 1.0, -1.0)

        def loss(p):
            o = real_model_apply(p, actor_input, actor_output, HeadParams())
            return jnp.sum(
                o.advantage.astype(jnp.float32) * jnp.asarray(sel) * jnp.asarray(legal)
            )

        return jax.grad(loss)(params)["params"]["advantage_head"]["macro_micro"][
            "micro"
        ]

    target_group = 1  # switch
    g = grads_for_group(target_group)

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


def test_forward_is_deterministic(real_model_and_trajectory, real_model_apply):
    network, params, actor_input, actor_output = real_model_and_trajectory
    from rl.model.heads import HeadParams

    a = real_model_apply(params, actor_input, actor_output, HeadParams())
    b = real_model_apply(params, actor_input, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(a.value_head.log_probs, dtype=np.float32),
        np.asarray(b.value_head.log_probs, dtype=np.float32),
    )


def _encoder_entity_biases(network, params, actor_input):
    """(public per-entity bias, private per-entity bias) as the encoder
    actually builds them for the latent read, on one timestep."""

    def call(module, env_step):
        (_, _, _, public_bias), (_, _, _, private_bias) = (
            module.encoder._current_entity_tokens(env_step)
        )
        return public_bias, private_bias

    env_step = jax.tree.map(lambda x: x[0], actor_input.env)
    return jax.jit(lambda p, e: network.apply(p, e, method=call))(params, env_step)


def test_private_sheet_is_not_tagged_with_the_opponents_side(
    real_model_and_trajectory,
):
    """My private sheet must not carry the tag that marks OPPONENT rows.

    The service writes ENTITY_PUBLIC_NODE_FEATURE__SIDE = isMySide(...), so
    side_bias row 1 is mine and row 0 is theirs. Until 2026-08-28 the sheet
    was tagged side_bias(0) -- the opponent's row -- which put 48 of the
    read's keys under the wrong side. The sheet now owns private_side_bias.
    """
    network, params, actor_input, _ = real_model_and_trajectory
    _, private_bias = _encoder_entity_biases(network, params, actor_input)

    encoder = params["params"]["encoder"]
    own = jnp.asarray(encoder["private_side_bias"][0], dtype=private_bias.dtype)
    opponent_tag = jnp.asarray(
        encoder["side_bias"]["embedding"][0], dtype=private_bias.dtype
    )

    for row in private_bias:
        np.testing.assert_allclose(np.asarray(row), np.asarray(own), atol=0)
    # The regression itself: the old wiring made this equality hold.
    assert not np.allclose(np.asarray(private_bias[0]), np.asarray(opponent_tag))


def test_private_side_bias_is_the_live_route(real_model_and_trajectory):
    """Positive control for the test above: it compares against a param, so
    it would pass just as well if the bias never reached the tokens."""
    network, params, actor_input, _ = real_model_and_trajectory
    _, before = _encoder_entity_biases(network, params, actor_input)

    perturbed = jax.tree.map(lambda x: x, params)
    encoder = perturbed["params"]["encoder"]
    encoder["private_side_bias"] = encoder["private_side_bias"] + 1.0
    _, after = _encoder_entity_biases(network, perturbed, actor_input)

    assert not np.allclose(np.asarray(before), np.asarray(after))


def _field_tokens(network, params, actor_input):
    """The (global, my-side, opp-side) field token triple for one timestep."""

    def call(module, field):
        return module.encoder._embed_field(field)[0]

    field = jax.tree.map(lambda x: x[0], actor_input.env.field)
    return jax.jit(lambda p, f: network.apply(p, f, method=call))(params, field)


def _perturbed(params, path, delta=1.0):
    tree = jax.tree.map(lambda x: x, params)
    node = tree["params"]["encoder"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]] + delta
    return tree


def test_field_side_tokens_do_not_read_the_active_status_table(
    real_model_and_trajectory,
):
    """The my/opp side-condition tokens must not borrow pos_bias.

    pos_bias is indexed by ENTITY_PUBLIC_NODE_FEATURE__ACTIVE (= scoreOrder,
    {0, 2} in singles), so before 2026-08-28 its row 0 meant both "benched
    pokemon" and "opponent side conditions" — and that bias was the only
    thing separating my hazards from theirs, since both sides share
    side_condition_linear.
    """
    network, params, actor_input, _ = real_model_and_trajectory
    base = _field_tokens(network, params, actor_input)

    moved_pos = _field_tokens(
        network, _perturbed(params, ("pos_bias", "embedding")), actor_input
    )
    np.testing.assert_allclose(np.asarray(base), np.asarray(moved_pos), atol=0)

    # Positive control: the replacement IS on the path, so the test above is
    # not passing merely because nothing reaches these tokens.
    moved_side = _field_tokens(
        network, _perturbed(params, ("field_side_bias",)), actor_input
    )
    assert not np.allclose(np.asarray(base[1]), np.asarray(moved_side[1]))
    assert not np.allclose(np.asarray(base[2]), np.asarray(moved_side[2]))
    # The global field token carries no side, so it must be untouched.
    np.testing.assert_allclose(np.asarray(base[0]), np.asarray(moved_side[0]), atol=0)


def _slot_queries(network, params, actor_input):
    """The 12 entity-slot queries, in ACTION_SLOT_READ_INDICES order."""

    def call(module, env_step):
        public, private = module.encoder._current_entity_tokens(env_step)
        return module.encoder._action_slot_queries(public[0], private[0])

    env_step = jax.tree.map(lambda x: x[0], actor_input.env)
    return np.asarray(
        jax.jit(lambda p, e: network.apply(p, e, method=call))(params, env_step),
        dtype=np.float32,
    )


def test_entity_slot_queries_name_distinct_things(real_model_and_trajectory):
    """Each entity-derived action slot must ask for a different thing.

    ALLY_i_SWITCH and ALLY_i_TARGET name the SAME mon, so species + side
    alone makes them the same query and the readout could not tell "switch
    this mon out" from "target it". The role bias separates them once
    trained — but it is ZERO-INIT, so at step 0 the separation is carried
    entirely by ActionSlotRead's learned per-slot position term. That is the
    contract this pins: the queries the read actually consumes are pairwise
    distinct from step 0.
    """
    network, params, actor_input, _ = real_model_and_trajectory
    content = _slot_queries(network, params, actor_input)
    assert content.shape[0] == 12

    position = np.asarray(
        params["params"]["encoder"]["action_slot_read"]["slot_position"],
        dtype=np.float32,
    )
    effective = content + position
    for i in range(12):
        for j in range(i + 1, 12):
            assert not np.allclose(effective[i], effective[j]), (i, j)

    # The control that shows position is doing that work at init: without it
    # the two roles for the same mon collapse onto each other.
    np.testing.assert_allclose(content[0:2], content[2:4], atol=2e-2)


def test_reserve_queries_are_content_derived(real_model_and_trajectory):
    """A reserve slot asks for the mon that occupies it, not for its index:
    swapping two sheet rows' species must swap their queries. This is the key
    the read matches against the same species tag on that mon's PUBLIC row —
    the route probe C measured as absent before the redesign."""
    network, params, actor_input, _ = real_model_and_trajectory
    base = _slot_queries(network, params, actor_input)

    from rl.environment.protos.features_pb2 import EntityPrivateNodeFeature

    species = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
    env = actor_input.env
    private = jnp.asarray(env.private_team)
    first, second = private[:, 0, species], private[:, 1, species]
    swapped = private.at[:, 0, species].set(second).at[:, 1, species].set(first)
    moved = _slot_queries(
        network, params, actor_input.replace(env=env.replace(private_team=swapped))
    )

    reserve = slice(6, 12)
    assert not np.allclose(base[reserve][0], moved[reserve][0])
    np.testing.assert_allclose(base[reserve][0], moved[reserve][1], atol=2e-2)
    np.testing.assert_allclose(base[reserve][1], moved[reserve][0], atol=2e-2)
    # Slots for mons that did not move are untouched.
    np.testing.assert_allclose(base[reserve][2:], moved[reserve][2:], atol=2e-2)
