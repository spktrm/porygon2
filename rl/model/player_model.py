from dotenv import load_dotenv

load_dotenv()
import functools
import os
from pprint import pprint

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
from ml_collections import ConfigDict

from rl.environment.data import (
    FLAT_MODALITY_MASK,
    NUM_MODALITY_FEATURES,
    SRC_MODALITY_MASK,
    ModalityEnum,
)
from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.environment.utils import get_ex_player_step
from rl.learner import checkpoint
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import (
    CategoricalValueLogitHead,
    HeadParams,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.modules import MLP
from rl.model.utils import get_num_params


def calculate_hierarchical_prior(valid_mask: jax.Array) -> jax.Array:

    valid_moves = valid_mask & (FLAT_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__MOVE)
    valid_switches = valid_mask & (
        FLAT_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__SWITCH
    )
    valid_wildcard = valid_mask & (
        FLAT_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__WILDCARD
    )
    valid_other = valid_mask & (FLAT_MODALITY_MASK == ModalityEnum.MODALITY_ENUM__OTHER)

    num_valid_moves = jnp.maximum(valid_moves.sum(), 1)
    num_valid_switches = jnp.maximum(valid_switches.sum(), 1)
    num_valid_wildcard = jnp.maximum(valid_wildcard.sum(), 1)
    num_valid_other = jnp.maximum(valid_other.sum(), 1)

    has_moves = valid_moves.any()
    has_switches = valid_switches.any()
    has_wildcard = valid_wildcard.any()
    has_other = valid_other.any()

    # Distribute probability mass equally among the *available* categories
    total_active_categories = (
        has_moves.astype(jnp.float32)
        + has_switches.astype(jnp.float32)
        + has_wildcard.astype(jnp.float32)
        + has_other.astype(jnp.float32)
    )

    category_mass = 1.0 / jnp.maximum(total_active_categories, 1.0)

    # Distribute the category mass evenly among the valid options inside it
    move_prior = jnp.where(valid_moves, category_mass / num_valid_moves, 0.0)
    switch_prior = jnp.where(valid_switches, category_mass / num_valid_switches, 0.0)
    wildcard_prior = jnp.where(valid_wildcard, category_mass / num_valid_wildcard, 0.0)
    other_prior = jnp.where(valid_other, category_mass / num_valid_other, 0.0)

    return move_prior + switch_prior + wildcard_prior + other_prior


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.v_head = CategoricalValueLogitHead(self.cfg.v_head)
        # Per-modality mass head. Zero-init output layer: the bias starts as
        # a no-op and the initial policy is the centered gram head alone.
        self.modality_bias_mlp = MLP(
            (self.cfg.entity_size,), name="modality_bias_mlp"
        )
        self.modality_bias_logits = nn.Dense(
            NUM_MODALITY_FEATURES,
            kernel_init=nn.initializers.zeros_init(),
            name="modality_bias_logits",
        )
        # Learnable mass semantics, per modality:
        #   mass_logit(m) = b_m + LME_tau(c) + kappa * log N_m
        # where LME_tau is the temperature-tau log-mean-exp of the centered
        # gram logits. tau (softplus; init 1 = exact no-op) interpolates the
        # mass score from "mean option" (tau -> 0) through the flat-softmax
        # spread term (tau = 1) to "best option" (tau -> inf); kappa (init 0)
        # restores count-proportional flat-softmax mass at 1.
        self.mass_temp_raw = self.param(
            "mass_temp_raw",
            nn.initializers.constant(float(np.log(np.expm1(1.0)))),
            (NUM_MODALITY_FEATURES,),
        )
        self.mass_count_coef = self.param(
            "mass_count_coef",
            nn.initializers.zeros_init(),
            (NUM_MODALITY_FEATURES,),
        )

    def _modality_bias(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        value_embeddings: jax.Array,
    ):
        """Per-modality mass logits, (NUM_MODALITY_FEATURES,).

        The mass decision sees both the state (value stream) and who the
        candidates are: every action embedding that participates in a valid
        pair of a modality — from either side, so switches see the reserve
        tgts, not just the shared switch src token — masked-mean pooled per
        modality (zeros for modalities with no valid pair; those are masked
        out of the softmax anyway).
        """
        dtype = action_embeddings.dtype
        src_modality_oh = jax.nn.one_hot(
            SRC_MODALITY_MASK, NUM_MODALITY_FEATURES, dtype=dtype
        )
        valid = valid_mask.astype(dtype)
        src_part = valid.max(axis=-1, keepdims=True) * src_modality_oh  # (A, M)
        tgt_part = (
            jnp.einsum("ab,am->bm", valid, src_modality_oh) > 0
        ).astype(dtype)
        weights = jnp.maximum(src_part, tgt_part)  # (A, M)
        pooled = jnp.einsum("am,ad->md", weights, action_embeddings) / weights.sum(
            axis=0
        )[..., None].clip(min=1.0)
        bias_input = jnp.concatenate((pooled.reshape(-1), value_embeddings), axis=-1)
        return self.modality_bias_logits(self.modality_bias_mlp(bias_input))

    def _forward_pi_head(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        value_embeddings: jax.Array,
    ):
        """Gram-matrix logits, factored into per-modality mass + preference.

        action_embeddings: (NUM_ACTION_FEATURES, entity_size), already
        normed by the encoder's out-norms. The gram term is mean-centered
        over the valid actions of each modality — zero-sum within a
        modality, so it carries only "which action" — while the learned
        per-modality bias carries "how much of each modality". Category-mass
        gradients land on the bias instead of rotating every embedding of a
        modality toward a common direction (which erodes within-modality
        discrimination). Returns (NUM_ACTION_FEATURES**2,) src x tgt logits.
        """
        square_logits = jnp.einsum(
            "ae,be->ab", action_embeddings, action_embeddings
        ) / np.array(action_embeddings.shape[-1] ** 0.5)
        logits = square_logits.reshape(-1)

        modality_oh = jax.nn.one_hot(
            FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES, dtype=logits.dtype
        )
        valid_bool = valid_mask.reshape(-1)
        valid = valid_bool.astype(logits.dtype)
        raw_counts = valid @ modality_oh  # (M,)
        counts = raw_counts.clip(min=1.0)
        # Differentiable through the mean: that is what makes the gram
        # term's within-modality gradients exactly zero-sum.
        modality_mean = ((valid * logits) @ modality_oh) / counts
        bias = self._modality_bias(action_embeddings, valid_mask, value_embeddings)

        # Mass semantics. Grouped by modality, the flat softmax over
        # (c + b - log N) implicitly gives mass_logit(m) = b_m + LME_1(c):
        # -log(count) cancels the +log N head start option-rich modalities
        # would get (which also swings with the mask — bench faints shrink
        # N_switch), scoring a modality by its mean option instead. The
        # correction below then swaps LME_1 for LME_tau and adds
        # kappa*log N, giving mass_logit(m) = b_m + LME_tau(c) + kappa*logN.
        # It is computed on stop_gradient(c): the swap is forward-only
        # calibration — within-modality gradients remain exactly those of
        # the tau=1 flat softmax, and mass gradients reach tau/kappa but
        # never the embeddings. log N is constant w.r.t. every parameter,
        # so kappa's term only ever trains kappa.
        log_counts32 = jnp.log(counts.astype(jnp.float32))
        centered = logits - modality_mean[FLAT_MODALITY_MASK]
        c_sg = jax.lax.stop_gradient(centered).astype(jnp.float32)
        tau = jax.nn.softplus(self.mass_temp_raw.astype(jnp.float32))  # (M,)
        member = valid_bool[:, None] & (modality_oh > 0)  # (A**2, M)
        lse_tau = nn.logsumexp(jnp.where(member, c_sg[:, None] * tau, -1e9), axis=0)
        lme_tau = (lse_tau - log_counts32) / tau
        lse_one = nn.logsumexp(jnp.where(member, c_sg[:, None], -1e9), axis=0)
        lme_one = lse_one - log_counts32
        mass_corr = jnp.where(
            raw_counts > 0,
            lme_tau
            - lme_one
            + self.mass_count_coef.astype(jnp.float32) * log_counts32,
            0.0,
        ).astype(logits.dtype)

        correction = bias - modality_mean - log_counts32.astype(logits.dtype) + mass_corr
        return logits + correction[FLAT_MODALITY_MASK]

    def _calculate_entropy_metrics(
        self, policy_metrics: PolicyHeadOutput, flat_valid_mask: jax.Array
    ):
        modality_oh = jax.nn.one_hot(
            FLAT_MODALITY_MASK,
            NUM_MODALITY_FEATURES,
            dtype=policy_metrics.log_policy.dtype,
        )
        valid_modality_mask = flat_valid_mask[..., None] * modality_oh

        modality_log_probs = nn.logsumexp(
            jnp.where(
                valid_modality_mask,
                policy_metrics.log_policy[..., None],
                -1e9,
            ),
            axis=0,
        )
        modality_probs = jnp.exp(modality_log_probs)

        # Count valid actions per modality
        valid_actions_per_modality = valid_modality_mask.sum(axis=0)

        # --- THE FIX ---

        # 1. Count how many total modalities actually have valid options
        num_valid_modalities = (valid_actions_per_modality > 0).sum(
            dtype=modality_probs.dtype
        )

        # 2. Calculate the raw entropy safely
        raw_modality_entropy = -jnp.sum(
            jnp.where(
                valid_actions_per_modality > 0, modality_probs * modality_log_probs, 0.0
            )
        )

        # 3. Calculate max possible entropy
        max_modality_entropy = jnp.log(jnp.maximum(num_valid_modalities, 1.0))

        # --- THE FIX ---
        # Create a safe denominator that is never 0.0, even when the mask is False
        safe_max_modality_entropy = jnp.where(
            num_valid_modalities > 1, max_modality_entropy, 1.0
        )

        # 4. Safely normalize using the safe denominator
        return jnp.where(
            num_valid_modalities > 1,
            raw_modality_entropy / safe_max_modality_entropy,
            0.0,
        )

    def _forward_action_head(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        value_embeddings: jax.Array,
    ):
        flat_valid_mask = valid_mask.reshape(-1)

        micro_logits = self._forward_pi_head(
            action_embeddings, valid_mask, value_embeddings
        )
        micro_logits = micro_logits / temp

        # --- Hierarchical Prior & Metrics ---
        # hierarchical_prior = calculate_hierarchical_prior(flat_valid_mask)
        # policy_metrics = compute_policy_metrics(
        #     logits=pi_logits, valid_mask=flat_valid_mask, prior=hierarchical_prior
        # )
        policy_metrics = compute_policy_metrics(
            logits=micro_logits, valid_mask=flat_valid_mask
        )

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(flat_valid_mask, micro_logits, -1e9),
                self.make_rng("sampling"),
            )

        log_prob = jnp.take(policy_metrics.log_policy, action_index, axis=-1)

        mask_width = valid_mask.shape[-1]
        src_index = action_index // mask_width
        tgt_index = action_index % mask_width

        normalized_modality_entropy = self._calculate_entropy_metrics(
            policy_metrics, flat_valid_mask
        )

        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            # Full support only in the learner: the magnet KL needs both
            # distributions; actors skip it so replay transitions stay small.
            log_policy=policy_metrics.log_policy if self.cfg.train else (),
            src_index=src_index,
            tgt_index=tgt_index,
            entropy=policy_metrics.entropy,
            normalized_entropy=policy_metrics.normalized_entropy,
            magnet_kl=policy_metrics.magnet_kl,
            logit_l2_norm=policy_metrics.logit_l2_norm,
            normalized_modality_entropy=normalized_modality_entropy,
        )

    def _forward_value_head(self, value_embeddings: jax.Array):
        """value_embeddings: (4 * entity_size,)."""
        return self.v_head(value_embeddings)

    def get_head_outputs(
        self,
        action_embeddings: jax.Array,
        value_embeddings: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        *,
        head_params: HeadParams,
    ):
        action_head = self._forward_action_head(
            action_embeddings,
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
            value_embeddings=value_embeddings,
        )

        return PlayerActorOutput(
            action_head=action_head,
            value_head=self._forward_value_head(value_embeddings),
        )

    def __call__(
        self,
        actor_input: PlayerActorInput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):
        """
        Shared forward pass for encoder and policy head.
        """
        action_embeddings, value_embeddings, *_ = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        head_fn = functools.partial(self.get_head_outputs, head_params=head_params)
        return jax.vmap(head_fn)(
            action_embeddings, value_embeddings, actor_input.env, actor_output
        )


def get_player_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_player_model_config()
    return Porygon2PlayerModel(config)


def create_attention_graph(path, value):
    # Extract string/integer keys from the JAX path tuple
    path_keys = [p.key for p in path[:-1]]

    if len(path_keys) > 0 and path_keys[-1] == "attn_weights":
        path_str = " -> ".join(str(k) for k in path_keys)

        if getattr(value, "val", None) is not None:
            value = value.val

        avg_attn = jnp.max(value[:, :20], axis=1)  # Shape: (H, S, S)
        if avg_attn.ndim > 3:
            avg_attn = jnp.max(avg_attn, 0)

        assert (
            avg_attn.ndim == 3
        ), f"Expected 3D array for {path_str} attention weights, got shape {avg_attn.shape}"

        avg_attn_np = np.asarray(avg_attn)

        # Use Plotly Express to facet the 3D array along the 0th dimension (Heads)
        fig = px.imshow(
            avg_attn_np,
            facet_col=0,
            facet_col_wrap=4,
            text_auto=True,
            aspect="auto",
            labels=dict(color="Attn Prob", facet_col="Head"),
            title=path_str,
        )

        fig.for_each_annotation(
            lambda a: a.update(text=a.text.replace("facet_col=", "Head "))
        )

        # --- File Saving Logic ---
        base_dir = "attn_weights"
        os.makedirs(base_dir, exist_ok=True)

        module_name = "_".join(str(k) for k in path_keys[:-1])

        save_path = os.path.join(base_dir, f"{module_name}.html")
        fig.write_html(save_path)

        print(f"Saved concatenated attention maps to {save_path}")
        # -------------------------

        return fig

    return value


def get_attention_maps(
    model: nn.Module,
    params: dict,
    actor_input: PlayerActorInput,
    actor_output: PlayerActorOutput,
    head_params: HeadParams,
    rng_key: jax.Array,
) -> dict:
    # Calling apply with mutable=['intermediates'] collects all variables sown to that collection
    outputs, state = model.apply(
        params,
        actor_input,
        actor_output,
        head_params,
        rngs={"sampling": rng_key},
        mutable=["intermediates"],
    )

    # Extract the nested dictionary of attention weights
    intermediates = state.get("intermediates", {})
    return jax.tree.map_with_path(create_attention_graph, intermediates)


def main(generation: int = 9):
    actor_network = get_player_model(get_player_model_config(generation, train=False))
    learner_network = get_player_model(get_player_model_config(generation, train=True))

    ex_actor_input, ex_actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    )
    key = jax.random.key(42)

    latest_ckpt = checkpoint.most_recent_ckpt_dir(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        params = checkpoint.load_component(latest_ckpt, "player", "params")
    else:
        params = learner_network.init(
            key, ex_actor_input, ex_actor_output, HeadParams()
        )

    actor_output = actor_network.apply(
        params,
        ex_actor_input,
        PlayerActorOutput(),
        HeadParams(temp=0.8),
        rngs={"sampling": key},
    )
    pprint(actor_output)

    attention_data = get_attention_maps(
        model=actor_network,
        params=params,
        actor_input=ex_actor_input,
        actor_output=actor_output,
        head_params=HeadParams(temp=0.8),
        rng_key=key,
    )

    try:
        pprint(get_num_params(params), sort_dicts=False)
    except Exception as e:
        print(f"Error calculating number of parameters: {e}")


if __name__ == "__main__":
    main()