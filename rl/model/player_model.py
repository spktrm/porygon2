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

from rl.environment.data import FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES, ModalityEnum
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

    def _forward_pi_head(self, action_embeddings: jax.Array):
        """Gram-matrix logits.

        action_embeddings: (NUM_ACTION_FEATURES, entity_size), already
        normed by the encoder's out-norms. Returns
        (NUM_ACTION_FEATURES**2,) src x tgt logits.
        """
        square_logits = jnp.einsum(
            "ae,be->ab", action_embeddings, action_embeddings
        ) / np.array(action_embeddings.shape[-1] ** 0.5)
        return square_logits.reshape(-1)

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
    ):
        flat_valid_mask = valid_mask.reshape(-1)

        micro_logits = self._forward_pi_head(action_embeddings) / temp

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

        mean_valid_logit = jnp.mean(micro_logits, where=flat_valid_mask)

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
        head_params: HeadParams,
    ):

        action_head = self._forward_action_head(
            action_embeddings,
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
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
        action_embeddings, value_embeddings = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(action_embeddings, value_embeddings, actor_input.env, actor_output)


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
