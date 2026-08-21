"""Cross-entity attribute pooling (2026-08-20): the current-state entities
are pooled with their attribute tokens mixed ACROSS entities, so a matchup
("their revealed Flamethrower vs this mon of mine") is a token-to-token
comparison rather than something the trunk has to reconstruct from two lossy
pooled summaries.

Three contracts make that safe, and all three are invisible in a shape test:
the mix is genuinely live, the pooling READ stays entity-local (so the output
is still one vector per entity), and masked-out entities stay inert. The
mix's residual gates are zero-init, so every "is it wired" assertion here has
to open them first or it passes vacuously.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from rl.model.config import get_player_model_config
from rl.model.encoder import (
    _PRIVATE_TOKEN_TYPES,
    _PUBLIC_TOKEN_TYPES,
    CrossEntityAttentionPool,
)

NUM_PUBLIC = 12
NUM_PRIVATE = 6
# Row layout of _PUBLIC_TOKEN_TYPES: species, ability, item, move0..3, ...
MOVE_TOKENS = slice(3, 7)
# Public rows are side-partitioned, actives first — 0-5 mine, 6-11 theirs.
OPPONENT_ACTIVE = 6


def _inputs(seed=0):
    cfg = get_player_model_config(generation=9, train=True).encoder
    rng = np.random.default_rng(seed)
    dim = cfg.entity_size

    def block(num, types):
        return jnp.asarray(
            rng.normal(size=(num, len(types), dim)), dtype=cfg.dtype
        )

    groups = (
        block(NUM_PUBLIC, _PUBLIC_TOKEN_TYPES),
        block(NUM_PRIVATE, _PRIVATE_TOKEN_TYPES),
    )
    masks = tuple(jnp.ones(group.shape[:2], dtype=bool) for group in groups)
    types = (_PUBLIC_TOKEN_TYPES, _PRIVATE_TOKEN_TYPES)
    return cfg, groups, masks, types


def _build(seed=0):
    cfg, groups, masks, types = _inputs(seed)
    module = CrossEntityAttentionPool(cfg)
    params = module.init(jax.random.key(seed), groups, masks, types)
    return module, params, groups, masks, types


def _open_mix(params):
    """Force the cross-entity attention's residual writes live.

    Its gates (`mha_a` / `ffn_a`) are zero-init, exactly like the
    entity-local block it replaces, so at init the mix is a no-op and any
    "information crosses entities" assertion would pass on a completely
    disconnected module. Note these are NOT named *_gate, so the ladder
    suite's `_open_all_gates` does not reach them.
    """
    flat = dict(flatten_dict(params))
    opened = 0
    for path, leaf in flat.items():
        if "attention" in path and path[-1] in ("mha_a", "ffn_a"):
            flat[path] = jnp.ones_like(leaf)
            opened += 1
    assert opened, "no residual scales found under the mix — contract changed?"
    return unflatten_dict(flat)


def _perturb(groups, group_index, entity, tokens=slice(None), seed=1):
    rng = np.random.default_rng(seed)
    group = groups[group_index]
    patch = jnp.asarray(
        rng.normal(size=group[entity, tokens].shape), dtype=group.dtype
    )
    perturbed = group.at[entity, tokens].set(patch)
    return tuple(
        perturbed if index == group_index else other
        for index, other in enumerate(groups)
    )


def test_pooled_output_keeps_the_per_entity_contract():
    """One vector per entity, groups split back in order — the state stream's
    row count must not move."""
    module, params, groups, masks, types = _build()
    public, private = module.apply(params, groups, masks, types)
    assert public.shape == (NUM_PUBLIC, module.cfg.entity_size)
    assert private.shape == (NUM_PRIVATE, module.cfg.entity_size)


def test_pool_read_is_entity_local():
    """With the mix closed (init gates), the block-diagonal read mask is the
    only thing standing between entities. If it were wrong, one entity's
    tokens would bleed into another's pooled vector even with no mixing."""
    module, params, groups, masks, types = _build()
    base_public, base_private = module.apply(params, groups, masks, types)

    moved = _perturb(groups, 0, OPPONENT_ACTIVE)
    public, private = module.apply(params, moved, masks, types)

    np.testing.assert_array_equal(np.asarray(private), np.asarray(base_private))
    others = [i for i in range(NUM_PUBLIC) if i != OPPONENT_ACTIVE]
    np.testing.assert_array_equal(
        np.asarray(public)[others], np.asarray(base_public)[others]
    )
    # The perturbed entity itself must still respond — otherwise the above
    # passes because nothing is connected at all.
    assert not np.allclose(
        np.asarray(public)[OPPONENT_ACTIVE],
        np.asarray(base_public)[OPPONENT_ACTIVE],
    )


def test_opponent_moves_reach_my_entity_vectors():
    """The point of the refactor: with the mix live, the opponent's move
    tokens must move MY private-sheet entity vectors — those are what
    warm-start the RESERVE_j switch slots, so this is the channel that lets
    a revealed threat inform a switch."""
    module, params, groups, masks, types = _build()
    opened = _open_mix(params)

    _, base_private = module.apply(opened, groups, masks, types)
    moved = _perturb(groups, 0, OPPONENT_ACTIVE, MOVE_TOKENS)
    _, private = module.apply(opened, moved, masks, types)

    assert not np.allclose(np.asarray(private), np.asarray(base_private)), (
        "opponent move tokens do not reach own-team entity vectors — the "
        "cross-entity mix is not wired"
    )

    # Same perturbation with the gates at their init value must do nothing,
    # which is what makes the assertion above attributable to the MIX rather
    # than to some other path.
    _, closed_base = module.apply(params, groups, masks, types)
    _, closed = module.apply(params, moved, masks, types)
    np.testing.assert_array_equal(np.asarray(closed), np.asarray(closed_base))


def test_masked_entities_are_inert():
    """An unrevealed/NULL/PAD row is a live row with a False token mask. It
    must not influence any other entity even with the mix wide open, or the
    absent half of a team would be feeding the matchup."""
    module, params, groups, masks, types = _build()
    opened = _open_mix(params)

    dead = NUM_PUBLIC - 1
    masks = (masks[0].at[dead].set(False), masks[1])
    base_public, base_private = module.apply(opened, groups, masks, types)

    moved = _perturb(groups, 0, dead, seed=7)
    public, private = module.apply(opened, moved, masks, types)

    np.testing.assert_array_equal(np.asarray(private), np.asarray(base_private))
    live = [i for i in range(NUM_PUBLIC) if i != dead]
    np.testing.assert_array_equal(
        np.asarray(public)[live], np.asarray(base_public)[live]
    )


def _biases(groups, seed=2):
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.normal(size=(group.shape[0], group.shape[-1])), dtype=group.dtype)
        for group in groups
    )


def test_identity_biases_enter_before_the_mix():
    """Ownership/position identity must be visible to the ATTENTION, not
    only stamped on the pooled output: changing the opponent active's
    per-entity bias has to move MY sheet vectors once the mix is open. With
    the mix closed the same change is confined to the row it belongs to,
    which is what attributes the cross-entity effect to the mix."""
    module, _, groups, masks, types = _build()
    biases = _biases(groups)
    params = module.init(jax.random.key(0), groups, masks, types, biases)
    opened = _open_mix(params)

    shifted = (
        biases[0].at[OPPONENT_ACTIVE].add(jnp.ones_like(biases[0][OPPONENT_ACTIVE])),
        biases[1],
    )
    _, base_private = module.apply(opened, groups, masks, types, biases)
    _, private = module.apply(opened, groups, masks, types, shifted)
    assert not np.allclose(np.asarray(private), np.asarray(base_private)), (
        "per-entity identity bias does not reach other entities through the mix"
    )

    closed_public, closed_private = module.apply(params, groups, masks, types, biases)
    moved_public, moved_private = module.apply(params, groups, masks, types, shifted)
    np.testing.assert_array_equal(np.asarray(moved_private), np.asarray(closed_private))
    untouched = np.array([i for i in range(NUM_PUBLIC) if i != OPPONENT_ACTIVE])
    np.testing.assert_array_equal(
        np.asarray(moved_public[untouched]), np.asarray(closed_public[untouched])
    )
    assert not np.allclose(
        np.asarray(moved_public[OPPONENT_ACTIVE]),
        np.asarray(closed_public[OPPONENT_ACTIVE]),
    )


def test_groups_are_distinguishable_at_init():
    """A sheet row and a public row with identical token content must not
    pool identically even with the mix closed — the per-group bias has a
    non-zero init so the two rows of the same mon are told apart from step 0."""
    module, params, groups, masks, types = _build()
    flat = dict(flatten_dict(params))
    (key,) = [k for k in flat if k[-1] == "group_bias"]
    assert np.asarray(flat[key]).any(), "group_bias is zero-init — groups degenerate"
