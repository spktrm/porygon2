"""Decision-time equilibrium search over the value latent (deploy-only).

Train-time cost is a handful of auxiliary heads on the encoder's value
stream; actors and replay are untouched. Deploy-time, the heads support a
depth-limited simultaneous-move search whose budget (depth / branching /
solver iterations) can be raised without retraining.

Everything the heads learn is self-supervised from self-play transitions;
the only labels are things that are *known*, never reconstructed: my own
taken action and the game outcome. In particular the opponent's
simultaneous action is not decoded from the battle log — it is inferred:

- **Intent u**: a small continuous latent inferred per transition by a
  posterior encoder that sees (s, s', my action). It is the opponent's
  contribution to the transition (plus whatever else gradient routes
  there — see the routing note below). A decision-time Gaussian prior
  p(u | s) is trained to anticipate it; at solve time the opponent's
  "action set" is k samples from that prior, and the anchored matrix
  game tilts adversarially *within* the sampled support.
- **Chance z**: a second, prior-free latent conditioned on u, absorbing
  the residual (crits, rolls). Deploy-time search averages over it.

Both latents are trained with no KL machinery. Their joint marginal over
the training distribution is pinned to an isotropic Gaussian by SIGReg
(LeJEPA), which is the collapse guard: an isotropic Gaussian has no dead
directions, no dominant mode and no degenerate tails, and the guarantee
applies exactly to the aggregate quantity that per-transition losses
leave unconstrained. It also makes deploy-time sampling exact — chance
draws come straight from N(0, I), intent draws from the learned
conditional prior whose target marginal is N(0, I).

This inverts the two choices that sank the latent_opponent experiment
(removed in 0877bde) *without* reintroducing symbolic label
reconstruction: continuous latents + SIGReg replace the discrete codebook
and its anti-collapse stack (VICReg hinges, balanced KL, free bits,
usage entropies), and the solve runs only at deploy time in
``search_step`` — a bad payoff model degrades the deployed bot, never the
learner.

Routing caveat (the honest cost of self-supervision): nothing hard
separates "opponent decision" from "RNG" without labels. The asymmetries
are structural — u gets first claim on transition information (z is
conditioned on u), u alone enters the payoff matrix, and only u has an
anticipation prior. Information leaking the wrong way shows up in the
telemetry: a dead intent channel means payoff columns collapse and the
search degenerates to the anchor (intent usage gap ~ 0), while RNG
leaking into u makes the solve treat luck as adversarial (risk-averse
play, bounded by the uniform anchor over prior samples).

The search latent is the encoder's value stream (``value_embeddings``).
It already has an external owner — the value cross-entropy — so the
latent-prediction dynamics loss cannot collapse it, and imagined states
can be scored by the existing payoff/value machinery with no separate
probe.

Soundness caveat: depth-limited search on one player's information states
without public belief states is not sound in the imperfect-information
sense. The guard is the piKL anchor — my side of every matrix solve is a
KL-anchored quantal response around the network policy's abstract
marginal, and the opponent side is anchored uniform over prior samples —
so the solve is a trust region around the trained (approximately
regularized) self-play policy and fades to the anchor wherever the
payoff model is uninformative. Search budget should only grow with
validated head quality; the anchor temperatures are the coupling.

My side acts in a small *abstract* action space so payoff matrices and
dynamics stay enumerable at imagined nodes: the 16 move slots (regular +
wildcard) + 6 reserve switch targets + 1 other (pass/structural), a
static function of the flat (src, tgt) grid. The root converts the
solved abstract mix back to flat actions by reweighting the network
policy within each abstract class, so fine-grained targeting stays with
the gram head. There are no legality masks anywhere in the search —
availability lives in the priors (my root prior is the flat policy's
abstract marginal, which is exactly zero on illegal classes; my
imagined-node prior is a head distilled from that marginal). The only
hard mask left is the flat action mask at the root's final sampling
step — the environment interface, not the search.
"""

import math

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.environment.data import (
    ALLY_SWITCH_INDICES,
    MOVE_INDICES,
    NUM_ACTION_FEATURES,
    RESERVE_ENTITY_INDICES,
)
from rl.environment.interfaces import SearchHeadOutput
from rl.model.modules import MLP

# --- My abstract action space: 16 move slots + 6 reserve switches + other.
NUM_MY_MOVE_SLOTS = len(MOVE_INDICES)
NUM_RESERVE_SLOTS = len(RESERVE_ENTITY_INDICES)
MY_ABSTRACT_SWITCH_OFFSET = NUM_MY_MOVE_SLOTS
MY_ABSTRACT_OTHER = NUM_MY_MOVE_SLOTS + NUM_RESERVE_SLOTS
NUM_MY_ABSTRACT_ACTIONS = MY_ABSTRACT_OTHER + 1


def _build_flat_to_my_abstract() -> np.ndarray:
    """Static (src, tgt) grid -> my abstract action id.

    Moves key on the src slot; battle switches on the reserve tgt (the src
    ALLY_i_SWITCH token carries the outgoing active, the tgt names the
    incoming candidate); team-preview switches use the reserve slot as src.
    Everything else (pass, structural targets) folds into OTHER.
    """
    table = np.full(
        (NUM_ACTION_FEATURES, NUM_ACTION_FEATURES), MY_ABSTRACT_OTHER, dtype=np.int32
    )
    for pos, slot in enumerate(MOVE_INDICES):
        table[slot, :] = pos
    for pos, slot in enumerate(RESERVE_ENTITY_INDICES):
        table[slot, :] = MY_ABSTRACT_SWITCH_OFFSET + pos
    for ally_slot in ALLY_SWITCH_INDICES:
        table[ally_slot, :] = MY_ABSTRACT_OTHER
        for pos, slot in enumerate(RESERVE_ENTITY_INDICES):
            table[ally_slot, slot] = MY_ABSTRACT_SWITCH_OFFSET + pos
    return table.reshape(-1)


FLAT_TO_MY_ABSTRACT = _build_flat_to_my_abstract()
# (A*A, NUM_MY_ABSTRACT_ACTIONS) membership, for marginalization without
# segment ops.
MY_ABSTRACT_ONEHOT = np.eye(NUM_MY_ABSTRACT_ACTIONS, dtype=np.float32)[
    FLAT_TO_MY_ABSTRACT
]


@chex.dataclass
class SearchLabels:
    """The only supervised quantities: my own action and pair validity.

    The opponent's action is deliberately NOT reconstructed from the
    battle log — it is inferred self-supervised as the intent latent
    (see SearchHeads.transition_outputs).
    """

    my_label: jax.Array
    pair_valid: jax.Array


def compute_search_labels(done: jax.Array, action_index: jax.Array) -> SearchLabels:
    """My abstract action (known, not reconstructed) + transition validity.

    A pair is usable when step t is a real decision step and t+1 exists in
    the buffer; post-episode padding is excluded by the learner's masks.
    """
    num_steps = done.shape[0]
    has_next = jnp.arange(num_steps) < (num_steps - 1)
    pair_valid = jnp.logical_not(done.astype(jnp.bool_)) & has_next
    my_label = jnp.take(jnp.asarray(FLAT_TO_MY_ABSTRACT), action_index)
    return SearchLabels(my_label=my_label, pair_valid=pair_valid)


def standardize(x: jax.Array) -> jax.Array:
    """Per-vector zero-mean / unit-rms (layer norm without params), fp32.

    The dynamics losses compare latents in this space. Every consumer of a
    latent (q_trunk, the policy/value MLPs, the value head) layer-norms
    its input, so a vector's mean and scale carry no information — and the
    raw value stream's norm grows over training, which makes an
    unnormalized MSE (and its gradients) scale-unbounded: it can dominate
    the global clip budget and squash the main objective's gradients.
    """
    x = x.astype(jnp.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    return x / jnp.sqrt(jnp.square(x).mean(axis=-1, keepdims=True) + 1e-6)


def my_abstract_marginal(log_policy: jax.Array, flat_mask: jax.Array) -> jax.Array:
    """Marginalize the flat (src x tgt) policy onto my abstract actions.

    log_policy: (..., A*A) from legal_log_policy — 0 (not -inf) on illegal
    entries, so the mask must gate before exp. Returns (..., NUM_MY)
    probabilities.
    """
    probs = jnp.where(flat_mask, jnp.exp(log_policy.astype(jnp.float32)), 0.0)
    return probs @ jnp.asarray(MY_ABSTRACT_ONEHOT)


def sigreg_loss(
    z: jax.Array,
    weights: jax.Array,
    rng_key: jax.Array,
    num_directions: int,
) -> jax.Array:
    """Sketched Isotropic Gaussian Regularization (LeJEPA).

    Pushes the weighted marginal of ``z`` toward N(0, I): project onto
    ``num_directions`` random unit directions (resampled every step) and
    score each 1-D marginal against a standard normal with the
    Epps-Pulley statistic — the squared error between the empirical and
    Gaussian characteristic functions, integrated under a N(0, 1) weight,
    which has the closed form

        sum_ij p_i p_j exp(-(x_i - x_j)^2 / 2)
        - sqrt(2) sum_i p_i exp(-x_i^2 / 4) + 1/sqrt(3)

    (>= 0, and 0 iff the weighted sample matches N(0, 1)). Differentiable
    everywhere with bounded gradients; O(num_directions * N^2).

    z: (N, K) samples; weights: (N,) validity weights (invalid rows get
    zero mass); rng_key: direction-sketch key.
    """
    z = z.astype(jnp.float32)
    weights = weights.astype(jnp.float32)
    p = weights / weights.sum().clip(min=1.0)

    directions = jax.random.normal(rng_key, (num_directions, z.shape[-1]))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True).clip(
        min=1e-6
    )
    x = z @ directions.T  # (N, M)

    pairwise = jnp.exp(-0.5 * jnp.square(x[:, None] - x[None, :]))  # (N, N, M)
    pairwise_term = jnp.einsum("i,j,ijm->m", p, p, pairwise)
    single_term = math.sqrt(2.0) * jnp.einsum(
        "i,im->m", p, jnp.exp(-0.25 * jnp.square(x))
    )
    return (pairwise_term - single_term + 1.0 / math.sqrt(3.0)).mean()


def solve_matrix_game(
    payoff: jax.Array,
    my_log_anchor: jax.Array,
    opp_log_anchor: jax.Array,
    num_steps: int,
    my_temp: float,
    opp_temp: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Symmetric piKL-hedge on a zero-sum matrix game, batched.

    Each side plays the KL-anchored quantal response to the running-average
    utility; the anchor is the off-support guard — the payoff is a
    regression estimate grounded only near the behavior distribution, and
    an unanchored solve would max-exploit exactly the entries the data
    never constrained. Averaged iterates carry the convergence guarantee.

    payoff: (..., A, B) fp32, my payoff (opponent sees -payoff).
    Anchors: (..., A) / (..., B) fp32 log-priors.
    Returns (x_avg, y_avg, x_avg @ payoff @ y_avg).
    """
    payoff = payoff.astype(jnp.float32)
    my_log_anchor = jax.nn.log_softmax(my_log_anchor.astype(jnp.float32), axis=-1)
    opp_log_anchor = jax.nn.log_softmax(opp_log_anchor.astype(jnp.float32), axis=-1)

    x = jnp.exp(my_log_anchor)
    x_acc = jnp.zeros_like(my_log_anchor)
    y_acc = jnp.zeros_like(opp_log_anchor)
    x_util_acc = jnp.zeros_like(my_log_anchor)
    y_util_acc = jnp.zeros_like(opp_log_anchor)
    for i in range(1, num_steps + 1):
        y_util_acc = y_util_acc - jnp.einsum("...a,...ab->...b", x, payoff)
        y = jax.nn.softmax(opp_log_anchor + y_util_acc / (i * opp_temp), axis=-1)
        y_acc = y_acc + y
        x_util_acc = x_util_acc + jnp.einsum("...ab,...b->...a", payoff, y)
        x = jax.nn.softmax(my_log_anchor + x_util_acc / (i * my_temp), axis=-1)
        x_acc = x_acc + x

    x_avg = x_acc / num_steps
    y_avg = y_acc / num_steps
    value = jnp.einsum("...a,...ab,...b->...", x_avg, payoff, y_avg)
    return x_avg, y_avg, value


class SearchHeads(nn.Module):
    """Self-supervised heads on the value latent for decision-time search.

    All heads are functions of (latent, my abstract action id, continuous
    transition latents) only, so they evaluate identically at real and
    imagined nodes; nothing here reads per-slot action embeddings or any
    decoded opponent state.
    """

    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size
        latent_size = 4 * entity_size
        embed_kwargs = dict(features=entity_size, dtype=self.cfg.dtype)
        # Small final init keeps priors near-uniform, latents near-zero and
        # payoffs near-zero at init, so the anchored solve starts as a
        # no-op.
        logits_kwargs = dict(
            kernel_init=nn.initializers.orthogonal(1e-2),
            dtype=self.cfg.dtype,
        )

        self.my_action_embedding = nn.Embed(
            NUM_MY_ABSTRACT_ACTIONS, name="my_action_embedding", **embed_kwargs
        )

        # My abstract policy q(a | s), distilled from the flat policy's
        # marginal so imagined nodes have a my-side anchor too.
        self.my_policy_mlp = MLP((entity_size,), name="my_policy_mlp")
        self.my_policy_logits = nn.Dense(
            NUM_MY_ABSTRACT_ACTIONS, **logits_kwargs, name="my_policy_logits"
        )

        # Intent u: the opponent's (latent) contribution to the transition,
        # inferred after the fact; its decision-time Gaussian prior is the
        # anticipation signal and the sampler for the payoff columns.
        # Chance z: prior-free residual, conditioned on u so it only ever
        # explains what intent does not. Both marginals are pinned to
        # N(0, I) by the learner's SIGReg loss — the collapse guard and
        # what makes deploy-time sampling exact.
        self.intent_posterior_mlp = MLP((entity_size,), name="intent_posterior_mlp")
        self.intent_posterior_head = nn.Dense(
            self.cfg.intent_dim, **logits_kwargs, name="intent_posterior_head"
        )
        self.intent_prior_mlp = MLP((entity_size,), name="intent_prior_mlp")
        self.intent_prior_head = nn.Dense(
            2 * self.cfg.intent_dim, **logits_kwargs, name="intent_prior_head"
        )
        self.chance_posterior_mlp = MLP((entity_size,), name="chance_posterior_mlp")
        self.chance_posterior_head = nn.Dense(
            self.cfg.chance_dim, **logits_kwargs, name="chance_posterior_head"
        )

        # Joint payoff Q(s, a, u): bilinear over state-conditioned my-action
        # keys and intent keys, regressed on the realized cell's v-trace
        # return. Off-support cells are generalization anchored by the
        # prior-mean consistency loss — exactly why the solve is anchored.
        qk_size = self.cfg.qk_size
        self.q_trunk = MLP((entity_size,), name="q_trunk")
        self.q_my_state_keys = nn.Dense(
            NUM_MY_ABSTRACT_ACTIONS * qk_size, **logits_kwargs, name="q_my_state_keys"
        )
        self.q_my_action_keys = nn.Dense(
            features=qk_size,
            use_bias=False,
            dtype=self.cfg.dtype,
            name="q_my_action_keys",
        )
        self.q_intent_mlp = MLP((entity_size,), name="q_intent_mlp")
        self.q_intent_keys = nn.Dense(qk_size, **logits_kwargs, name="q_intent_keys")

        # Latent afterstate dynamics g(s, a, u, z) -> s'. The target space
        # (the value stream) is owned by the value loss, so constant-target
        # collapse is not available.
        self.dynamics_mlp = MLP((latent_size, latent_size), name="dynamics_mlp")

    def my_log_policy(self, latent: jax.Array) -> jax.Array:
        # Full softmax — availability is learned from the distill target's
        # zeros, never imposed; imagined nodes have no mask to impose.
        logits = self.my_policy_logits(self.my_policy_mlp(latent))
        return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

    def intent_prior(self, latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        """p(u | s) as a diagonal Gaussian; (..., d_u) mu and sigma, fp32.

        The sigma floor bounds the NLL's 1/sigma^2 gradient: with the
        intent marginal pinned to N(0, 1), conditional precision beyond
        sigma = 0.05 carries no useful anticipation signal and only
        manufactures exploding losses on the occasional surprising
        transition.
        """
        raw = self.intent_prior_head(self.intent_prior_mlp(latent)).astype(jnp.float32)
        mu, sigma_raw = jnp.split(raw, 2, axis=-1)
        return mu, jax.nn.softplus(sigma_raw) + 0.05

    def q_payoff(self, latent: jax.Array, intents: jax.Array) -> jax.Array:
        """(..., latent), (..., K, d_u) -> (..., NUM_MY, K) fp32 payoff."""
        qk_size = self.cfg.qk_size
        trunk = self.q_trunk(latent)
        my_keys = self.q_my_state_keys(trunk).reshape(
            *trunk.shape[:-1], NUM_MY_ABSTRACT_ACTIONS, qk_size
        ) + self.q_my_action_keys(self.my_action_embedding.embedding)
        trunk_per_intent = jnp.broadcast_to(
            trunk[..., None, :], (*intents.shape[:-1], trunk.shape[-1])
        )
        intent_keys = self.q_intent_keys(
            self.q_intent_mlp(
                jnp.concatenate(
                    (trunk_per_intent, intents.astype(trunk.dtype)), axis=-1
                )
            )
        )  # (..., K, qk)
        payoff = jnp.einsum("...aq,...kq->...ak", my_keys, intent_keys) / math.sqrt(
            qk_size
        )
        return payoff.astype(jnp.float32)

    def dynamics(
        self,
        latent: jax.Array,
        my_embedding: jax.Array,
        intent: jax.Array,
        chance: jax.Array,
    ) -> jax.Array:
        """Residual afterstate step; inputs broadcast over leading dims."""
        shape = jnp.broadcast_shapes(
            latent.shape[:-1],
            my_embedding.shape[:-1],
            intent.shape[:-1],
            chance.shape[:-1],
        )
        inputs = jnp.concatenate(
            (
                jnp.broadcast_to(latent, (*shape, latent.shape[-1])),
                jnp.broadcast_to(my_embedding, (*shape, my_embedding.shape[-1])),
                jnp.broadcast_to(
                    intent.astype(latent.dtype), (*shape, intent.shape[-1])
                ),
                jnp.broadcast_to(
                    chance.astype(latent.dtype), (*shape, chance.shape[-1])
                ),
            ),
            axis=-1,
        )
        return latent + self.dynamics_mlp(inputs)

    def transition_outputs(
        self,
        latent: jax.Array,
        next_latent: jax.Array,
        labels: SearchLabels,
        my_marginal: jax.Array,
    ) -> SearchHeadOutput:
        """Train-time per-step losses over one trajectory.

        latent/next_latent: (T, 4D) value-stream latents at consecutive
        requests; my_marginal: (T, NUM_MY) the flat policy's abstract
        marginal (fresh forward). Prediction targets are stop-gradiented;
        the latent space is shaped by the value loss plus these auxiliaries
        under the learner's coefficients.
        """
        my_embedding = self.my_action_embedding(labels.my_label)

        # --- Transition latents. Intent gets first claim on transition
        # information (chance is conditioned on it); no KL tax anywhere —
        # informativeness is free, the chance/intent dimensionality
        # bottlenecks keep them from carrying what the deterministic
        # (s, a) path can explain, and the learner's SIGReg pins their
        # joint marginal to N(0, I).
        intent = self.intent_posterior_head(
            self.intent_posterior_mlp(
                jnp.concatenate((latent, next_latent, my_embedding), axis=-1)
            )
        )  # (T, d_u)
        chance = self.chance_posterior_head(
            self.chance_posterior_mlp(
                jnp.concatenate((latent, next_latent, my_embedding, intent), axis=-1)
            )
        )  # (T, d_z)

        predictions = self.dynamics(latent, my_embedding, intent, chance)
        target = standardize(jax.lax.stop_gradient(next_latent))
        dynamics_loss = jnp.square(standardize(predictions) - target).mean(axis=-1)

        # Usage-gap diagnostics (logged, never optimized — excluded from
        # the training loss, so no gradient flows through them): re-predict
        # with one channel's latent replaced by its neighbour's. Outcomes
        # are ~independent across turns, so the excess error measures how
        # much information each channel actually carries; ~0 means the
        # channel is being ignored. A roll stands in for a shuffle because
        # the learner forward has no rng stream.
        shuffled_intent_predictions = self.dynamics(
            latent, my_embedding, jnp.roll(intent, 1, axis=0), chance
        )
        dynamics_loss_shuffled_intent = jnp.square(
            standardize(shuffled_intent_predictions) - target
        ).mean(axis=-1)
        shuffled_chance_predictions = self.dynamics(
            latent, my_embedding, intent, jnp.roll(chance, 1, axis=0)
        )
        dynamics_loss_shuffled_chance = jnp.square(
            standardize(shuffled_chance_predictions) - target
        ).mean(axis=-1)

        # --- Anticipation: the prior chases the (stop-gradiented) inferred
        # intent — Gaussian NLL, so sigma honestly reports how much of the
        # opponent's mixture is unpredictable at decision time. Intent
        # inference itself is shaped by the dynamics and SIGReg losses
        # only, never pulled toward prior-convenient assignments.
        prior_mu, prior_sigma = self.intent_prior(latent)
        intent_sg = jax.lax.stop_gradient(intent).astype(jnp.float32)
        intent_prior_loss = (
            0.5 * jnp.square((intent_sg - prior_mu) / prior_sigma)
            + jnp.log(prior_sigma)
        ).sum(axis=-1)

        # --- My abstract policy: distilled from the flat policy's marginal
        # (stop-gradient — the head chases the policy, never the reverse).
        # The marginal is exactly zero on illegal classes, so the distill
        # target itself teaches availability.
        my_log_probs = self.my_log_policy(latent)
        distill_target = jax.lax.stop_gradient(my_marginal)
        my_policy_loss = -jnp.sum(distill_target * my_log_probs, axis=-1)

        # --- Payoff: realized cell + prior-mean consistency scalars; the
        # learner supplies the regression targets (v-trace return, value
        # expectation). The intent enters through stop-gradient so the
        # value regression cannot rewrite intent semantics.
        realized_payoff = self.q_payoff(latent, intent_sg[:, None, :]).squeeze(
            -1
        )  # (T, NUM_MY)
        q_taken = jnp.take_along_axis(
            realized_payoff, labels.my_label[:, None], axis=-1
        ).squeeze(-1)
        mean_intent_payoff = self.q_payoff(
            latent, jax.lax.stop_gradient(prior_mu)[:, None, :]
        ).squeeze(-1)
        q_prior_mean = jnp.sum(distill_target * mean_intent_payoff, axis=-1)

        return SearchHeadOutput(
            my_policy_loss=my_policy_loss,
            q_taken=q_taken,
            q_prior_mean=q_prior_mean,
            dynamics_loss=dynamics_loss,
            dynamics_loss_shuffled_intent=dynamics_loss_shuffled_intent,
            dynamics_loss_shuffled_chance=dynamics_loss_shuffled_chance,
            intent_prior_loss=intent_prior_loss,
            intent_prior_sigma=prior_sigma.mean(axis=-1),
            transition_posterior=jnp.concatenate((intent, chance), axis=-1).astype(
                jnp.float32
            ),
            pair_valid=labels.pair_valid,
        )

    def solve_tree(
        self, latent: jax.Array, my_log_prior: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Depth-limited simultaneous-move search from one (batch of) node(s).

        Fixed-shape and fully vectorized. At every node the opponent's
        "action set" is num_intent_samples draws from the intent prior
        p(u | s); the payoff head scores my abstract actions against each
        sampled intent and the anchored matrix game tilts adversarially
        within that support (opponent anchor: uniform over samples, since
        the samples already carry the prior's weighting). Levels below the
        root additionally expand the top k_me of my prior times
        num_chance_samples draws of z ~ N(0, I) — the correct chance
        marginal by construction, since SIGReg pins the posterior's
        marginal to it — replace those payoff rows with chance-averaged
        backed-up child values, and re-solve. depth == 1 is the pure root
        solve on the payoff head (Layer 1).

        Everything below the root is latent: my actions are abstract ids,
        the opponent's are prior samples, and availability lives in the
        learned priors, not in masks (imagined nodes have no observation
        to mask from).
        """
        return self._solve_level(latent, my_log_prior, self.cfg.depth)

    def _solve_level(self, latent, my_log_prior, depth):
        num_intents = self.cfg.num_intent_samples
        prior_mu, prior_sigma = self.intent_prior(latent)
        intent_noise = jax.random.normal(
            self.make_rng("sampling"),
            (*latent.shape[:-1], num_intents, self.cfg.intent_dim),
            dtype=jnp.float32,
        )
        intents = prior_mu[..., None, :] + prior_sigma[..., None, :] * intent_noise
        payoff = jax.lax.stop_gradient(
            self.q_payoff(latent, intents)
        )  # (..., NUM_MY, K)

        if depth > 1:
            k_me = self.cfg.top_k_me
            num_chance = self.cfg.num_chance_samples

            my_top = jax.lax.top_k(my_log_prior, k_me)[1]  # (..., k_me)
            my_embedding = self.my_action_embedding(my_top)
            # Child grid: (..., k_me, K, num_chance, latent).
            chance = jax.random.normal(
                self.make_rng("sampling"),
                (
                    *latent.shape[:-1],
                    k_me,
                    num_intents,
                    num_chance,
                    self.cfg.chance_dim,
                ),
                dtype=jnp.float32,
            )
            child_latent = self.dynamics(
                latent[..., None, None, None, :],
                my_embedding[..., :, None, None, :],
                intents[..., None, :, None, :],
                chance,
            )

            _, _, child_value = self._solve_level(
                child_latent,
                self.my_log_policy(child_latent),
                depth - 1,
            )
            pair_value = child_value.mean(axis=-1)  # (..., k_me, K)

            my_onehot = jax.nn.one_hot(my_top, NUM_MY_ABSTRACT_ACTIONS)
            covered = (my_onehot.sum(axis=-2) > 0)[..., None]  # (..., NUM_MY, 1)
            backed_up = jnp.einsum("...ia,...ik->...ak", my_onehot, pair_value)
            payoff = jnp.where(covered, backed_up, payoff)

        opp_log_anchor = jnp.zeros(payoff.shape[:-2] + payoff.shape[-1:])
        return solve_matrix_game(
            payoff,
            my_log_prior,
            opp_log_anchor,
            num_steps=self.cfg.solver_steps,
            my_temp=self.cfg.my_anchor_temp,
            opp_temp=self.cfg.opp_anchor_temp,
        )
