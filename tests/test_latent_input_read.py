"""Perceiver-style latent input read (2026-08-21): K learned latents
cross-attend one flat token set (entity attribute tokens + field +
prev-action + raw history) and become the trunk's state rows.

Contracts that a shape test cannot see: the read is LIVE at init (its
residual starts at 1.0 -- content reaches the latents only through it),
masked tokens are absent keys (a masked entity is inert), and identity
enters on the tokens BEFORE the read (a matchup is an (entity, side)
pairing the attention can only form if the tokens already say whose they
are). Privileged routing is not a module property -- it is the encoder
building two instances -- and is pinned by tests/test_value_ladder.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from rl.model.config import get_player_model_config
from rl.model.constants import (
    FIELD_TOKEN_TYPES,
    HISTORY_FIELD_TOKEN_TYPES,
    PREV_ACTION_TOKEN_TYPES,
    PRIVATE_TOKEN_TYPES,
    PUBLIC_READ_TOKEN_TYPES,
)
from rl.model.encoder import LatentInputRead

NUM_PUBLIC = 12
NUM_PRIVATE = 6
MOVE_TOKENS = slice(3, 7)
# Public rows are side-partitioned, actives first -- 0-5 mine, 6-11 theirs.
OPPONENT_ACTIVE = 6


def _inputs(seed=0):
    cfg = get_player_model_config(generation=9, train=True).encoder
    rng = np.random.default_rng(seed)
    dim = cfg.entity_size

    def block(num, types):
        return jnp.asarray(rng.normal(size=(num, len(types), dim)), dtype=cfg.dtype)

    types = (
        PUBLIC_READ_TOKEN_TYPES,
        PRIVATE_TOKEN_TYPES,
        FIELD_TOKEN_TYPES,
        PREV_ACTION_TOKEN_TYPES,
        HISTORY_FIELD_TOKEN_TYPES,
    )
    groups = (
        block(NUM_PUBLIC, types[0]),
        block(NUM_PRIVATE, types[1]),
        block(1, types[2]),
        block(1, types[3]),
        block(1, types[4]),
    )
    masks = tuple(jnp.ones(g.shape[:2], dtype=bool) for g in groups)
    biases = (
        jnp.asarray(rng.normal(size=(NUM_PUBLIC, dim)), dtype=cfg.dtype),
        jnp.asarray(rng.normal(size=(NUM_PRIVATE, dim)), dtype=cfg.dtype),
        None,
        None,
        None,
    )
    return cfg, groups, masks, types, biases


def _build(seed=0):
    cfg, groups, masks, types, biases = _inputs(seed)
    module = LatentInputRead(cfg, cfg.num_latents)
    params = module.init(jax.random.key(seed), groups, masks, types, biases)
    return module, params, groups, masks, types, biases


def _perturb(groups, group_index, entity, tokens=slice(None), seed=1):
    rng = np.random.default_rng(seed)
    group = groups[group_index]
    patch = jnp.asarray(rng.normal(size=group[entity, tokens].shape), dtype=group.dtype)
    return tuple(
        group.at[entity, tokens].set(patch) if i == group_index else other
        for i, other in enumerate(groups)
    )


def test_output_is_the_latent_array():
    module, params, groups, masks, types, biases = _build()
    out = module.apply(params, groups, masks, types, biases)
    assert out.shape == (module.cfg.num_latents, module.cfg.entity_size)
    assert bool(jnp.isfinite(out.astype(jnp.float32)).all())
    assert 12 * 11 + 6 * 8 + 3 + 2 + 1 == sum(g.shape[0] * g.shape[1] for g in groups)


def test_read_is_live_at_init_and_opponent_moves_reach_the_latents():
    """No gate opening needed: the read's residual starts at 1.0, so an
    opponent move token must move the latents on fresh params -- this is
    the channel that lets a revealed threat inform every downstream row."""
    module, params, groups, masks, types, biases = _build()
    base = module.apply(params, groups, masks, types, biases)
    moved = _perturb(groups, 0, OPPONENT_ACTIVE, MOVE_TOKENS)
    out = module.apply(params, moved, masks, types, biases)
    assert not np.allclose(np.asarray(out), np.asarray(base))


def test_masked_tokens_are_inert():
    """A masked token is an absent key: perturbing it leaves the latents
    bitwise unchanged, so an unrevealed/NULL/PAD row cannot feed anything."""
    module, params, groups, masks, types, biases = _build()
    masks = list(masks)
    masks[0] = masks[0].at[OPPONENT_ACTIVE].set(False)
    masks = tuple(masks)
    base = module.apply(params, groups, masks, types, biases)
    moved = _perturb(groups, 0, OPPONENT_ACTIVE)
    out = module.apply(params, moved, masks, types, biases)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(base))


def test_identity_biases_enter_before_the_read():
    """Ownership/position identity must be visible to the attention: the
    opponent active's per-entity bias alone (content fixed) must move the
    latents."""
    module, params, groups, masks, types, biases = _build()
    base = module.apply(params, groups, masks, types, biases)
    shifted = list(biases)
    shifted[0] = biases[0].at[OPPONENT_ACTIVE].add(jnp.ones_like(biases[0][0]))
    out = module.apply(params, groups, masks, types, tuple(shifted))
    assert not np.allclose(np.asarray(out), np.asarray(base))


def test_groups_and_types_are_distinguishable_at_init():
    """A sheet row and a public row with identical content must not be the
    same key at step 0: group_bias is non-zero-init. Token-type and row
    biases are zero-init by design (learned identities)."""
    module, params, *_ = _build()
    flat = dict(flatten_dict(params))
    (gb,) = [v for k, v in flat.items() if k[-1] == "group_bias"]
    assert np.asarray(gb).any()
    assert np.asarray(gb).shape[0] == 5


HISTORY_TOKEN = 10  # the 11th public token: that slot's recurrent state


def _swap(groups, group_index, entity_a, entity_b, token):
    group = groups[group_index]
    a, b = group[entity_a, token], group[entity_b, token]
    swapped = group.at[entity_a, token].set(b).at[entity_b, token].set(a)
    return tuple(
        swapped if i == group_index else other for i, other in enumerate(groups)
    )


def test_history_tokens_are_bound_to_their_entity():
    """Swapping two slots' history states must change the read.

    The per-slot recurrent state is public entity i's 11th token (2026-08-28),
    so it rides that entity's per-entity bias and two history states are not
    interchangeable. Encoder.__call__ re-aligns history-slot order to
    public-row order with PUBLIC_ORDER before the read; this is the property
    that makes that re-alignment load-bearing rather than a no-op.
    """
    module, params, groups, masks, types, biases = _build()
    base = module.apply(params, groups, masks, types, biases)
    swapped = _swap(groups, 0, 0, OPPONENT_ACTIVE, HISTORY_TOKEN)
    out = module.apply(params, swapped, masks, types, biases)
    assert not np.allclose(np.asarray(out), np.asarray(base))


def test_a_shared_history_group_could_not_tell_them_apart():
    """The control that proves the test above can fail: under the previous
    layout the 12 slot states were one entity's 13 tokens, sharing a single
    entity_bias row and a single HISTORY_SLOT type — a multiset of identically
    biased keys, so the same swap was BITWISE inert and public_order bought
    nothing."""
    from rl.model.constants import NUM_PUBLIC_SLOTS, TokenType

    cfg, groups, masks, types, biases = _inputs()
    rng = np.random.default_rng(7)
    dim = cfg.entity_size
    legacy_types = np.array(
        NUM_PUBLIC_SLOTS * [TokenType.HISTORY_SLOT] + [TokenType.HISTORY_FIELD],
        dtype=np.int32,
    )
    legacy_history = jnp.asarray(
        rng.normal(size=(1, len(legacy_types), dim)), dtype=cfg.dtype
    )
    groups = groups[:4] + (legacy_history,)
    masks = masks[:4] + (jnp.ones(legacy_history.shape[:2], dtype=bool),)
    types = types[:4] + (legacy_types,)

    module = LatentInputRead(cfg, cfg.num_latents)
    params = module.init(jax.random.key(0), groups, masks, types, biases)
    base = module.apply(params, groups, masks, types, biases)
    swapped = tuple(
        g.at[0, 0].set(groups[4][0, 1]).at[0, 1].set(groups[4][0, 0]) if i == 4 else g
        for i, g in enumerate(groups)
    )
    out = module.apply(params, swapped, masks, types, biases)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(base))
