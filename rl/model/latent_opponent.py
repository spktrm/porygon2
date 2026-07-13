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
- prior p(z | s_t): predicts the posterior before the turn resolves — this
  is the anticipation signal, trained with KL(sg(q) || p).

At decision time a bilinear payoff over (my action slots x codes) is solved
with a fixed unroll of regret matching, opponent side initialized at the
prior, and the resulting per-action payoffs enter the policy logits behind
a zero-init gate — so the whole subsystem starts as a no-op and only shapes
the policy as the optimizer grows the gate.

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

    def action_logit_bonus(
        self,
        action_embeddings: jax.Array,
        flat_valid_mask: jax.Array,
        prior_log_probs: jax.Array,
    ) -> jax.Array:
        """Per-flat-action payoff against the regret-matched opponent mix.

        action_embeddings: (A, D); flat_valid_mask: (A*A,) over the src x tgt
        grid; prior_log_probs: (K,). Returns (A*A,) logit bonus, gated.
        """
        num_actions = action_embeddings.shape[0]
        codes = self.codebook.embedding.astype(self.cfg.dtype)

        code_keys = self.payoff_code(codes)  # (K, q)
        scale = 1.0 / math.sqrt(code_keys.shape[-1])
        src_payoff = self.payoff_src(action_embeddings) @ code_keys.T * scale  # (A, K)
        tgt_payoff = self.payoff_tgt(action_embeddings) @ code_keys.T * scale  # (A, K)
        payoff = (src_payoff[:, None, :] + tgt_payoff[None, :, :]).reshape(
            num_actions * num_actions, -1
        )

        # The matrix game is solved in fp32; the prior enters through
        # stop-gradient so it trains only against the posterior, never toward
        # a policy-convenient opponent.
        payoff_f32 = payoff.astype(jnp.float32)
        valid = flat_valid_mask.astype(jnp.float32)
        x0 = valid / valid.sum().clip(min=1)
        y0 = jnp.exp(jax.lax.stop_gradient(prior_log_probs))

        x, y = x0, y0
        x_regrets = jnp.zeros_like(x0)
        y_regrets = jnp.zeros_like(y0)
        for _ in range(self.cfg.num_rm_steps):
            x_util = payoff_f32 @ y  # my payoff per action
            y_util = -(x @ payoff_f32)  # zero-sum: opponent sees -M
            x_regrets = nn.relu(x_regrets + (x_util - x @ x_util) * valid)
            y_regrets = nn.relu(y_regrets + (y_util - y @ y_util))
            x_sum = x_regrets.sum()
            y_sum = y_regrets.sum()
            x = jnp.where(x_sum > 0, x_regrets / x_sum.clip(min=1e-9), x0)
            y = jnp.where(y_sum > 0, y_regrets / y_sum.clip(min=1e-9), y0)

        bonus = (payoff_f32 @ y).astype(self.cfg.dtype)
        return self.bonus_gate.astype(self.cfg.dtype) * bonus * flat_valid_mask

    def transition_outputs(
        self,
        field_states: jax.Array,
        next_field_states: jax.Array,
        taken_action_embeddings: jax.Array,
        prior_log_probs: jax.Array,
        pair_valid: jax.Array,
    ) -> LatentOpponentHeadOutput:
        """Train-time losses over a trajectory.

        field_states / next_field_states: (T, D) world-model field snapshots
        at consecutive requests; taken_action_embeddings: (T, D);
        prior_log_probs: (T, K); pair_valid: (T,) bool for transitions whose
        both endpoints are real steps.
        """
        posterior_input = jnp.concatenate(
            (field_states, next_field_states, taken_action_embeddings), axis=-1
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

        # Forward prediction for every intent code, mixed by the posterior:
        # the soft assignment keeps everything differentiable and lets
        # gradient shape the codebook directly. The noise latent enters as a
        # soft-mixed embedding instead of being enumerated — intent
        # multimodality is what the payoff matrix needs preserved; noise is
        # only a conditioning input that soaks up unpredictable variance.
        codes = self.codebook.embedding.astype(self.cfg.dtype)  # (K, D)
        noise_codes = self.noise_codebook.embedding.astype(self.cfg.dtype)  # (N, D)
        num_codes, entity_size = codes.shape
        num_steps = field_states.shape[0]
        noise_mix = noise_probs.astype(self.cfg.dtype) @ noise_codes  # (T, D)
        base = jnp.concatenate(
            (field_states, taken_action_embeddings, noise_mix), axis=-1
        )
        forward_input = jnp.concatenate(
            (
                jnp.broadcast_to(base[:, None], (num_steps, num_codes, base.shape[-1])),
                jnp.broadcast_to(codes[None], (num_steps, num_codes, entity_size)),
            ),
            axis=-1,
        )
        predictions = self.forward_mlp(forward_input)  # (T, K, D)
        prediction = jnp.einsum(
            "tk,tkd->td", posterior_probs.astype(self.cfg.dtype), predictions
        )

        target = jax.lax.stop_gradient(next_field_states).astype(jnp.float32)
        forward_loss = jnp.square(prediction.astype(jnp.float32) - target).mean(axis=-1)

        # Balanced intent KL (Dreamer-style): the sg(q)-side teaches the
        # prior; the sg(p)-side taxes the posterior for encoding what the
        # prior cannot predict. Without the posterior-side tax the intent
        # channel is free and would swallow RNG; with it, unpredictable
        # information routes to the (more weakly taxed) noise channel.
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

        posterior_entropy = -jnp.sum(posterior_probs * posterior_log_probs, axis=-1)
        prior_probs = jnp.exp(prior_log_probs)
        prior_entropy = -jnp.sum(prior_probs * prior_log_probs, axis=-1)

        return LatentOpponentHeadOutput(
            forward_loss=forward_loss,
            kl=kl,
            noise_kl=noise_kl,
            pair_valid=pair_valid,
            posterior_probs=posterior_probs,
            noise_posterior_probs=noise_probs,
            prior_entropy=prior_entropy,
            posterior_entropy=posterior_entropy,
        )
