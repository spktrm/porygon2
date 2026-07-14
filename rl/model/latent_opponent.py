"""Latent opponent-action model.

The opponent's simultaneous action is never observed directly, but its
consequences are: each request-to-request transition of the world model's
recurrent field state summarizes what happened between two decision points.
This module learns a small codebook of K abstract opponent actions from
those consequences, with no action labels and no simulator:

- posterior q(z | s_t, s_t+1, my action): infers after the fact which code
  explains the transition (inverse dynamics; train-time only).
- forward model f(s_t, my action, z): predicts the next field state, so a
  code must actually carry the transition information it claims to explain.
  The loss is the posterior expectation of per-code error, E_q[||f(z) -
  s_t+1||^2] — each code the posterior selects must individually explain
  the transition, so codes cannot smear into basis vectors whose average
  predicts well.
- prior p(z | s_t): predicts the posterior before the turn resolves — this
  is the anticipation signal, trained with KL(sg(q) || p). Only trained on
  pairs that span a turn boundary: consecutive requests inside one turn
  (forced switches after a faint, etc.) contain no opponent decision, so
  there is nothing to anticipate; a turn-advanced flag conditions the
  posterior and forward model instead, so request mechanics are explained
  without burning code capacity.

At decision time a bilinear payoff over (my action slots x codes) is solved
with a fixed unroll: my side runs regret matching, while the opponent side
plays a KL-anchored quantal response y ∝ prior * exp(-payoff^T x / temp) —
the learned prior is the anchor throughout the solve, not just an
initialization, so anticipation genuinely constrains the imagined opponent.
The bonus uses the iterate-averaged opponent mix (the quantity with the
convergence guarantee), the solve itself runs under stop-gradient (gradient
reaches the payoff only through the final, differentiable payoff @ sg(y)
matmul), and the result enters the policy logits behind a zero-init gate —
so the whole subsystem starts as a no-op and only shapes the policy as the
optimizer grows the gate.

The payoff matrix is grounded by an auxiliary regression: the taken
action's payoff row, weighted by the inferred (stop-gradient) posterior
code, regresses onto the realized advantage — so payoff entries mean
"advantage of this action against that opponent intent" before the gate
ever opens, instead of learning solely from policy gradient through a
gated scalar. The codebook embeddings are stop-gradiented inside every
payoff read: their semantics belong to the forward model alone, and
neither policy gradient nor the grounding loss may rewrite them.

The latent is factored: alongside the intent codes, a second small
categorical noise latent absorbs the unpredictable part of the transition
(crits, misses, damage rolls, simultaneous-resolution order), keeping the
intent codes and the anticipation prior clean. Routing is enforced by cost
asymmetry: the intent KL is balanced (the posterior pays for deviating from
the state prior, Dreamer-style), while the noise channel is taxed only by a
weaker KL to uniform — so predictable information is near-free through the
intent channel and unpredictable information is cheaper through the noise
channel. The noise latent has no prior head and never enters the payoff
matrix: it is not strategic, only explanatory.
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import LatentOpponentHeadOutput
from rl.model.modules import MLP


class LatentOpponentModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        entity_size = self.cfg.entity_size
        num_codes = self.cfg.num_codes

        self.codebook = nn.Embed(
            num_codes, entity_size, dtype=self.cfg.dtype, name="codebook"
        )
        self.noise_codebook = nn.Embed(
            self.cfg.num_noise_codes,
            entity_size,
            dtype=self.cfg.dtype,
            name="noise_codebook",
        )

        # Small final init keeps all distributions near uniform at init.
        logits_kwargs = dict(
            kernel_init=nn.initializers.orthogonal(1e-2),
            dtype=self.cfg.dtype,
        )
        self.prior_mlp = MLP((entity_size,), name="prior_mlp")
        self.prior_logits = nn.Dense(
            features=num_codes, **logits_kwargs, name="prior_logits"
        )
        # One posterior trunk, two heads: intent codes and noise codes both
        # read the same transition evidence.
        self.posterior_mlp = MLP((entity_size,), name="posterior_mlp")
        self.posterior_logits = nn.Dense(
            features=num_codes, **logits_kwargs, name="posterior_logits"
        )
        self.noise_posterior_logits = nn.Dense(
            features=self.cfg.num_noise_codes,
            **logits_kwargs,
            name="noise_posterior_logits",
        )
        self.forward_mlp = MLP((entity_size, entity_size), name="forward_mlp")

        qk_size = self.cfg.qk_size
        payoff_kwargs = dict(features=qk_size, use_bias=False, dtype=self.cfg.dtype)
        self.payoff_src = nn.Dense(**payoff_kwargs, name="payoff_src")
        self.payoff_tgt = nn.Dense(**payoff_kwargs, name="payoff_tgt")
        self.payoff_code = nn.Dense(**payoff_kwargs, name="payoff_code")
        self.bonus_gate = self.param("bonus_gate", nn.initializers.zeros_init(), ())

    def prior_log_probs(self, value_embeddings: jax.Array) -> jax.Array:
        """p(z | s_t) from the value-query stream. (..., 4D) -> (..., K) fp32."""
        logits = self.prior_logits(self.prior_mlp(value_embeddings))
        return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

    def _code_keys(self) -> jax.Array:
        # Codebook semantics belong to the forward model; every payoff read
        # sees the codes through stop-gradient so neither policy gradient nor
        # the grounding loss can rewrite them into something convenient.
        codes = jax.lax.stop_gradient(self.codebook.embedding).astype(self.cfg.dtype)
        return self.payoff_code(codes)  # (K, q)

    def action_logit_bonus(
        self,
        action_embeddings: jax.Array,
        flat_valid_mask: jax.Array,
        prior_log_probs: jax.Array,
    ) -> jax.Array:
        """Per-flat-action payoff against the anchored opponent response.

        action_embeddings: (A, D); flat_valid_mask: (A*A,) over the src x tgt
        grid; prior_log_probs: (K,). Returns (A*A,) logit bonus, gated.
        """
        num_actions = action_embeddings.shape[0]
        code_keys = self._code_keys()
        scale = 1.0 / math.sqrt(code_keys.shape[-1])
        src_payoff = self.payoff_src(action_embeddings) @ code_keys.T * scale  # (A, K)
        tgt_payoff = self.payoff_tgt(action_embeddings) @ code_keys.T * scale  # (A, K)
        payoff = (src_payoff[:, None, :] + tgt_payoff[None, :, :]).reshape(
            num_actions * num_actions, -1
        )

        # The matrix game is solved in fp32 entirely under stop-gradient:
        # backprop through unrolled solver iterates is noisy, so gradient
        # reaches the payoff only through the final payoff @ sg(y) matmul.
        # The prior enters through stop-gradient so it trains only against
        # the posterior, never toward a policy-convenient opponent.
        payoff_f32 = payoff.astype(jnp.float32)
        payoff_sg = jax.lax.stop_gradient(payoff_f32)
        log_prior = jax.lax.stop_gradient(prior_log_probs)
        valid = flat_valid_mask.astype(jnp.float32)
        x0 = valid / valid.sum().clip(min=1)

        # My side: regret matching over valid actions. Opponent side: a
        # KL-anchored quantal response to my current mix — the exact
        # maximizer of <y, -M^T x> - temp * KL(y || prior) — so the learned
        # anticipation constrains the imagined opponent at every iterate
        # rather than serving only as an initialization.
        temp = self.cfg.anchor_temp
        x = x0
        x_regrets = jnp.zeros_like(x0)
        y_acc = jnp.zeros_like(log_prior)
        for _ in range(self.cfg.num_rm_steps):
            y = jax.nn.softmax(log_prior - (x @ payoff_sg) / temp)
            y_acc = y_acc + y
            x_util = payoff_sg @ y  # my payoff per action
            x_regrets = nn.relu(x_regrets + (x_util - x @ x_util) * valid)
            x_sum = x_regrets.sum()
            x = jnp.where(x_sum > 0, x_regrets / x_sum.clip(min=1e-9), x0)

        # Average iterate, not last: the averaged strategy is the one with
        # the convergence guarantee; the last iterate can cycle.
        y_avg = y_acc / self.cfg.num_rm_steps

        bonus = (payoff_f32 @ y_avg).astype(self.cfg.dtype)
        return self.bonus_gate.astype(self.cfg.dtype) * bonus * flat_valid_mask

    def transition_outputs(
        self,
        field_states: jax.Array,
        next_field_states: jax.Array,
        taken_src_embeddings: jax.Array,
        taken_tgt_embeddings: jax.Array,
        prior_log_probs: jax.Array,
        pair_valid: jax.Array,
        turn_advanced: jax.Array,
    ) -> LatentOpponentHeadOutput:
        """Train-time losses over a trajectory.

        field_states / next_field_states: (T, D) world-model field snapshots
        at consecutive requests; taken_src/tgt_embeddings: (T, D) the two
        halves of the taken action; prior_log_probs: (T, K); pair_valid:
        (T,) bool for transitions whose both endpoints are real steps;
        turn_advanced: (T,) bool for pairs that span a turn boundary (the
        only pairs that contain an opponent decision).
        """
        taken_action_embeddings = taken_src_embeddings + taken_tgt_embeddings
        turn_flag = turn_advanced.astype(self.cfg.dtype)[..., None]
        posterior_input = jnp.concatenate(
            (field_states, next_field_states, taken_action_embeddings, turn_flag),
            axis=-1,
        )
        posterior_trunk = self.posterior_mlp(posterior_input)
        posterior_log_probs = jax.nn.log_softmax(
            self.posterior_logits(posterior_trunk).astype(jnp.float32), axis=-1
        )
        posterior_probs = jnp.exp(posterior_log_probs)  # (T, K)
        noise_log_probs = jax.nn.log_softmax(
            self.noise_posterior_logits(posterior_trunk).astype(jnp.float32), axis=-1
        )
        noise_probs = jnp.exp(noise_log_probs)  # (T, N)

        # Forward prediction for every intent code; the loss is the posterior
        # expectation of per-code error, so every code the posterior selects
        # must individually predict the transition (mixing predictions before
        # the loss would let codes smear into a basis whose average predicts
        # well without any single code meaning an opponent action). The noise
        # latent enters as a soft-mixed embedding instead of being enumerated
        # — intent multimodality is what the payoff matrix needs preserved;
        # noise is only a conditioning input that soaks up unpredictable
        # variance. The turn flag explains request mechanics (same-turn
        # forced switches) for free, without burning code capacity.
        codes = self.codebook.embedding.astype(self.cfg.dtype)  # (K, D)
        noise_codes = self.noise_codebook.embedding.astype(self.cfg.dtype)  # (N, D)
        num_codes, entity_size = codes.shape
        num_steps = field_states.shape[0]
        noise_mix = noise_probs.astype(self.cfg.dtype) @ noise_codes  # (T, D)
        base = jnp.concatenate(
            (field_states, taken_action_embeddings, noise_mix, turn_flag), axis=-1
        )
        forward_input = jnp.concatenate(
            (
                jnp.broadcast_to(base[:, None], (num_steps, num_codes, base.shape[-1])),
                jnp.broadcast_to(codes[None], (num_steps, num_codes, entity_size)),
            ),
            axis=-1,
        )
        predictions = self.forward_mlp(forward_input)  # (T, K, D)

        target = jax.lax.stop_gradient(next_field_states).astype(jnp.float32)
        per_code_loss = jnp.square(
            predictions.astype(jnp.float32) - target[:, None]
        ).mean(
            axis=-1
        )  # (T, K)
        forward_loss = jnp.sum(posterior_probs * per_code_loss, axis=-1)

        # Balanced intent KL (Dreamer-style): the sg(q)-side teaches the
        # prior; the sg(p)-side taxes the posterior for encoding what the
        # prior cannot predict. Without the posterior-side tax the intent
        # channel is free and would swallow RNG; with it, unpredictable
        # information routes to the (more weakly taxed) noise channel. The
        # learner averages this only over intent_valid pairs — same-turn
        # request pairs have no opponent decision to anticipate.
        balance = self.cfg.kl_balance
        kl_prior_side = jnp.sum(
            jax.lax.stop_gradient(posterior_probs)
            * (jax.lax.stop_gradient(posterior_log_probs) - prior_log_probs),
            axis=-1,
        )
        kl_posterior_side = jnp.sum(
            posterior_probs
            * (posterior_log_probs - jax.lax.stop_gradient(prior_log_probs)),
            axis=-1,
        )
        kl = balance * kl_prior_side + (1.0 - balance) * kl_posterior_side

        # Noise KL to uniform = log N - H(q_noise); trains the noise
        # posterior directly (there is no noise prior to teach).
        num_noise_codes = noise_probs.shape[-1]
        noise_kl = math.log(num_noise_codes) + jnp.sum(
            noise_probs * noise_log_probs, axis=-1
        )

        # Payoff grounding: the taken action's payoff row against the
        # inferred code mix regresses (in the learner) onto the realized
        # advantage, so payoff entries mean "advantage of this action
        # against that opponent intent" before the gate opens. The posterior
        # is stop-gradiented: intent inference is shaped by the forward/KL
        # losses only, never pulled toward value-explaining assignments.
        code_keys = self._code_keys()
        scale = 1.0 / math.sqrt(code_keys.shape[-1])
        taken_payoff_per_code = (
            (
                self.payoff_src(taken_src_embeddings)
                + self.payoff_tgt(taken_tgt_embeddings)
            )
            @ code_keys.T
            * scale
        )  # (T, K)
        taken_payoff = jnp.sum(
            taken_payoff_per_code.astype(jnp.float32)
            * jax.lax.stop_gradient(posterior_probs),
            axis=-1,
        )

        posterior_entropy = -jnp.sum(posterior_probs * posterior_log_probs, axis=-1)
        prior_probs = jnp.exp(prior_log_probs)
        prior_entropy = -jnp.sum(prior_probs * prior_log_probs, axis=-1)

        return LatentOpponentHeadOutput(
            forward_loss=forward_loss,
            kl=kl,
            noise_kl=noise_kl,
            pair_valid=pair_valid,
            intent_valid=pair_valid & turn_advanced,
            taken_payoff=taken_payoff,
            posterior_probs=posterior_probs,
            noise_posterior_probs=noise_probs,
            prior_entropy=prior_entropy,
            posterior_entropy=posterior_entropy,
        )
