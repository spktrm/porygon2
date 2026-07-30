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
``player_potential_target_adv_share_fn``. The RL model itself trains fully
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
from rl.environment.protos.features_pb2 import InfoFeature
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


def has_rating_conditioning(params: Params) -> bool:
    """True when the artifact was trained with the Elo-conditioned
    architecture (its rating embedding subtree exists)."""
    return "rating_embed" in params.get("params", {})


def make_potential_apply(
    generation: int,
    uncertainty_scale: float = 0.0,
    readout: str = "margin",
    with_aux: bool = False,
    rating_conditioning: bool = False,
    condition_rating: int = 0,
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
    is still a state potential, so PBRS invariance is untouched.

    With ``with_aux=True`` the function returns ``(phi, aux)`` where aux
    holds per-step (T, B) diagnostics: the ungated ensemble mean
    (``phi_raw``), member disagreement (``ensemble_std``), and the
    confidence gate (``gate``). The gated phi is unchanged either way.

    ``rating_conditioning`` must match the artifact's architecture (detect
    with has_rating_conditioning). With it on, ``condition_rating`` > 0
    overwrites both rating features so Φ answers "how do games between
    players of THAT rating resolve from here" — e.g. condition high to
    shape RL toward strong-play outcome beliefs instead of ladder-average
    conversion. 0 leaves the input's ratings alone (live self-play carries
    none, so the critic falls back to its unknown-rating bucket)."""
    if readout not in ("margin", "win"):
        raise ValueError(f"unknown potential readout: {readout!r}")
    model = Porygon2OfflineCritic(
        get_player_model_config(generation, train=False),
        rating_conditioning=rating_conditioning,
    )
    single = jax.vmap(model.apply, in_axes=(None, 1), out_axes=1)
    ensemble = jax.vmap(single, in_axes=(0, None))

    def potential(params: Params, actor_input: PlayerActorInput) -> jax.Array:
        if rating_conditioning and condition_rating > 0:
            info = actor_input.env.info
            info = info.at[..., InfoFeature.INFO_FEATURE__MY_RATING].set(
                condition_rating
            )
            info = info.at[..., InfoFeature.INFO_FEATURE__OPP_RATING].set(
                condition_rating
            )
            actor_input = actor_input.replace(env=actor_input.env.replace(info=info))
        value_head = ensemble(params, actor_input)
        if readout == "win":
            probs = jnp.exp(value_head.log_probs.astype(jnp.float32))
            half = probs.shape[-1] // 2  # bins: [-6..-1, 0, +1..+6]
            phi = probs[..., half + 1 :].sum(-1) - probs[..., :half].sum(-1)
        else:
            phi = value_head.expectation.astype(jnp.float32)  # (K, T, B)
        mean = phi.mean(axis=0)
        std = phi.std(axis=0)
        gate = (
            jnp.exp(-uncertainty_scale * std)
            if uncertainty_scale > 0.0
            else jnp.ones_like(std)
        )
        gated = jax.lax.stop_gradient(mean * gate)
        if not with_aux:
            return gated
        aux = jax.lax.stop_gradient(
            {"phi_raw": mean, "ensemble_std": std, "gate": gate}
        )
        return gated, aux

    return potential
