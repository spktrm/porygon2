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
from rl.model.encoder import (
    _FIELD_TOKEN_TYPES,
    _HISTORY_TOKEN_TYPES,
    _PREV_ACTION_TOKEN_TYPES,
    _PRIVATE_TOKEN_TYPES,
    _PUBLIC_TOKEN_TYPES,
    LatentInputRead,
)

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
        _PUBLIC_TOKEN_TYPES,
        _PRIVATE_TOKEN_TYPES,
        _FIELD_TOKEN_TYPES,
        _PREV_ACTION_TOKEN_TYPES,
        _HISTORY_TOKEN_TYPES,
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
    assert 12 * 10 + 6 * 8 + 3 + 2 + 13 == sum(g.shape[0] * g.shape[1] for g in groups)


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
