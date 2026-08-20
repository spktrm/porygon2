"""Collapse diagnostics for a specific saved checkpoint (not a fresh
random-init model): dormant-unit / representation-rank collapse via the
same probe the live learner uses (Learner._make_plasticity_probe), plus
separation of the small learned embedding tables (value-ladder query rows,
target/pass action-grid rows) read directly off params.

Points at a checkpoint via:
    PORYGON2_CKPT_DIR  - explicit checkpoint dir (e.g. ./ckpts/gen9/ckpt_00093126)
    PORYGON2_CKPT_GEN  - generation to find the latest checkpoint under
                          ./ckpts/gen{N}/ when PORYGON2_CKPT_DIR is unset
                          (default 9)
Skips cleanly if neither resolves to an existing checkpoint.

Thresholds below are smoke bounds meant to catch gross collapse, not a
tight regression gate against a known-healthy baseline — tighten once
real per-checkpoint numbers are on hand.
"""

import os

import jax
import numpy as np
import pytest

from rl import checkpoint
from rl.online.artifact import read_manifest

pytestmark = [pytest.mark.slow]

_DEFAULT_CKPT_ROOT = "./ckpts/gen{generation}/"

_EMBEDDING_TABLE_NAMES = (
    "all_value_embeddings",
    "private_value_embeddings",
    "public_value_embeddings",
    "target_embeddings",
    "pass_embeddings",
)


def _resolve_ckpt_dir() -> str | None:
    explicit = os.environ.get("PORYGON2_CKPT_DIR")
    if explicit:
        return explicit
    generation = int(os.environ.get("PORYGON2_CKPT_GEN", "9"))
    root = _DEFAULT_CKPT_ROOT.format(generation=generation)
    return checkpoint.most_recent_ckpt_dir(root)


@pytest.fixture(scope="session")
def ckpt_dir() -> str:
    resolved = _resolve_ckpt_dir()
    if resolved is None:
        pytest.skip(
            "no checkpoint found — set PORYGON2_CKPT_DIR to point at one, "
            "or PORYGON2_CKPT_GEN to pick a generation under ./ckpts/"
        )
    return resolved


@pytest.fixture(scope="session")
def ckpt_target_params(ckpt_dir):
    """The EMA (target) params — same choice the league uses for opponents,
    since it's the smoothed, deployable snapshot rather than the noisier
    live optimiser params."""
    return checkpoint.load_component(ckpt_dir, "player", "target_params")


def _find_leaf(params, name: str) -> np.ndarray:
    matches = [
        np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
        if path and getattr(path[-1], "key", None) == name
    ]
    assert len(matches) == 1, f"expected exactly one {name!r} leaf, found {len(matches)}"
    return matches[0]


def _pairwise_cosine_similarities(table: np.ndarray) -> np.ndarray:
    table = table.astype(np.float32)
    unit = table / (np.linalg.norm(table, axis=-1, keepdims=True) + 1e-8)
    sim = unit @ unit.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return sim[iu]


@pytest.mark.gpu
def test_checkpoint_representation_not_collapsed(ckpt_dir, ckpt_target_params):
    """Runs the live learner's plasticity probe (dormant-unit fraction +
    srank@0.99, see learner._embedding_stats) against this checkpoint's
    params on the bundled example trajectory, for both trunk embedding
    streams (action, value-all)."""
    from rl.environment.interfaces import Batch, PlayerTransition
    from rl.environment.utils import get_ex_player_step
    from rl.model.config import get_player_model_config
    from rl.model.player_model import get_player_model
    from rl.online.learner import Learner

    manifest = read_manifest(ckpt_dir) or {}
    generation = int(manifest.get("generation", 9))

    network = get_player_model(get_player_model_config(generation=generation, train=True))
    actor_input, actor_output = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())

    # This probe reads a checkpoint through TODAY's model code, so it is only
    # meaningful while the two agree on the architecture. A checkpoint that
    # predates an added module would be probed with that module randomly
    # initialised — a contaminated reading dressed up as a collapse verdict —
    # and a strict apply would just raise ScopeCollectionNotFound. merge_params
    # is the same audit the learner's params-mode restart uses, so the skip
    # reason names exactly what a resume would carry over fresh.
    from rl.model.heads import HeadParams
    from rl.online.artifact import merge_params

    fresh = network.init(jax.random.key(0), actor_input, actor_output, HeadParams())
    _, kept_fresh = merge_params(fresh, ckpt_target_params)
    if kept_fresh:
        pytest.skip(
            f"checkpoint predates the current architecture "
            f"({len(kept_fresh)} param path(s) would init fresh, e.g. "
            f"{', '.join(kept_fresh[:3])}) — reprobe after a run on this code"
        )
    # Probe vmaps over axis 1 (batch); re-add a batch axis of 1 to the
    # unbatched example (same reshape test_plasticity.py uses).
    batched = jax.tree.map(lambda x: np.asarray(x)[:, None], actor_input)
    batch = Batch(
        player_transitions=PlayerTransition(env_output=batched.env),
        player_packed_history=batched.packed_history,
        player_history=batched.history,
    )

    probe = Learner._make_plasticity_probe(None, network)
    logs = probe(ckpt_target_params, batch)

    for name in ("action", "value"):
        dormant_frac = float(np.asarray(logs[f"plasticity_{name}_emb_dormant_frac"]))
        srank_frac = float(np.asarray(logs[f"plasticity_{name}_emb_srank_frac"]))
        assert np.isfinite(dormant_frac) and np.isfinite(srank_frac)
        assert dormant_frac < 0.5, (
            f"{name} stream: {dormant_frac:.1%} of units dormant — looks collapsed"
        )
        assert srank_frac > 0.01, (
            f"{name} stream: srank@0.99 is only {srank_frac:.1%} of width — "
            "representation rank looks collapsed"
        )


def test_checkpoint_embedding_tables_not_collapsed(ckpt_target_params):
    """The small learned embedding tables (value-ladder query rows,
    target/pass action-grid rows) shouldn't collapse into near-duplicate
    row vectors — read directly off params, no forward pass needed."""
    for name in _EMBEDDING_TABLE_NAMES:
        table = _find_leaf(ckpt_target_params, name)
        assert np.isfinite(table).all(), f"{name}: contains non-finite values"

        sims = _pairwise_cosine_similarities(table)
        assert sims.max() < 0.999, (
            f"{name}: two rows are near-duplicate (max cosine {sims.max():.4f}) "
            "— table looks collapsed"
        )
        assert sims.mean() < 0.9, (
            f"{name}: rows average cosine similarity {sims.mean():.4f} — "
            "table looks poorly separated"
        )
