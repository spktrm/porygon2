"""Loading offline critic artifacts for the RL learner.

The offline critic is a STANDALONE model: it shares Encoder code with the
RL model for convenience, but its trained params never enter the RL
network. The single consumption mode is the learned state potential — set
``Porygon2LearnerConfig.offline_critic_ckpt_path`` and the learner loads
the critic once, holds its params outside the train state (never in the
optimizer, never donated, stop-gradient at use), evaluates Φ(s) once per
trajectory at replay-buffer insert (the critic is frozen, so Φ is
immutable data cached across replay reuse), and feeds it into
compute_player_targets' potential advantage channel, gated by
``player_potential_advantage_coef_fn``. The RL model itself trains fully
from scratch — no frozen or warm-started subtrees.

The critic is public-only by construction: it operates exclusively on the
recurrent history pathway plus an offline-only antisymmetric probe, so
private fields, movesets, and action masks are architecturally
unreachable, and the history inputs are built from the same protocol
events offline and live — the frozen Φ carries no train/serve
distribution bias into RL training.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from rl.environment.interfaces import PlayerActorInput
from rl.learner import checkpoint as checkpoint_lib
from rl.model.config import get_player_model_config
from rl.model.utils import Params
from rl.offline.model import Porygon2OfflineCritic


def load_critic_params(artifact_paths: str | Sequence[str]) -> Params:
    """Loads one artifact or an ensemble, stacking param trees along a
    leading ensemble axis (K=1 for a single path — one unified code path)."""
    if isinstance(artifact_paths, str):
        artifact_paths = (artifact_paths,)
    trees = [
        checkpoint_lib.load_component(path, "player", "params")
        for path in artifact_paths
    ]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


def make_potential_apply(
    generation: int,
    uncertainty_scale: float = 0.0,
    readout: str = "margin",
) -> Callable[[Params, PlayerActorInput], jax.Array]:
    """Builds the frozen-critic potential: (stacked params, (T, B, ...)
    actor input) -> Φ in [-1, 1] with shape (T, B), float32, stop-gradient.

    ``readout`` selects what Φ reads off the critic's 13-bin margin
    distribution (the training target is margin bins either way — the
    richer distributional supervision is kept regardless of readout):
      - "margin": expected margin / MAX_MARGIN. Grades decisiveness —
        keeps shaping gradient alive inside already-decided positions,
        at the cost of transiently rewarding margin-seeking lines where
        they diverge from win-optimal ones (e.g. refusing a sack).
      - "win": P(win) − P(loss), the signed sign-mass of the same bins.
        Pure outcome belief — flat across decided positions, never
        prefers a wider win over a likelier one.
    Both are mirror-antisymmetric readouts of the same head (mirroring
    swaps the distribution's arms), so Φ stays exactly zero-sum.

    Params carry a leading ensemble axis (see load_critic_params). With
    uncertainty_scale > 0, Φ = mean_k(Φ_k) * exp(-scale * std_k(Φ_k)):
    where the ensemble members (trained on disjoint replay splits) agree,
    shaping speaks at full strength; where they disagree — off the human
    data distribution, exactly where the critic is extrapolating — it goes
    quiet. Each Φ_k is mirror-antisymmetric and std is mirror-invariant,
    so the gated Φ stays exactly zero-sum. A confidence-scaled potential
    is still a state potential, so PBRS invariance is untouched."""
    if readout not in ("margin", "win"):
        raise ValueError(f"unknown potential readout: {readout!r}")
    model = Porygon2OfflineCritic(get_player_model_config(generation, train=False))
    single = jax.vmap(model.apply, in_axes=(None, 1), out_axes=1)
    ensemble = jax.vmap(single, in_axes=(0, None))

    def potential(params: Params, actor_input: PlayerActorInput) -> jax.Array:
        value_head = ensemble(params, actor_input)
        if readout == "win":
            probs = jnp.exp(value_head.log_probs.astype(jnp.float32))
            half = probs.shape[-1] // 2  # bins: [-6..-1, 0, +1..+6]
            phi = probs[..., half + 1 :].sum(-1) - probs[..., :half].sum(-1)
        else:
            phi = value_head.expectation.astype(jnp.float32)  # (K, T, B)
        mean = phi.mean(axis=0)
        if uncertainty_scale > 0.0:
            mean = mean * jnp.exp(-uncertainty_scale * phi.std(axis=0))
        return jax.lax.stop_gradient(mean)

    return potential