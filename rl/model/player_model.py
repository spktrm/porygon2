from dotenv import load_dotenv

load_dotenv()
import functools
import os
from pprint import pprint

import cloudpickle as pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
from ml_collections import ConfigDict

from rl.environment.data import FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES, ModalityEnum
from rl.environment.interfaces import (
    CategoricalValueHeadOutput,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.environment.utils import CategoricalValueHeadOutput, get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import (
    HeadParams,
    PointerLogits,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.utils import get_most_recent_file, get_num_params


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
        """
        Initializes the encoder, policy head, value head, and Q-value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.macro_weights = self.param(
            "macro_weights",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.cfg.entity_size,),
        )
        self.micro_pi_head = PointerLogits(**self.cfg.pi_head.qk_logits.to_dict())
        self.v_head = PointerLogits(**self.cfg.v_head.qk_logits.to_dict())

    def _forward_macro_head(self, macro_action_embeddings: jax.Array):
        embedding_dtype = macro_action_embeddings.dtype
        scale = jnp.sqrt(self.cfg.entity_size).astype(embedding_dtype)
        macro_pi_logits = (
            macro_action_embeddings @ self.macro_weights.astype(embedding_dtype)
        ) / scale
        modality_mask_oh = jax.nn.one_hot(
            FLAT_MODALITY_MASK, num_classes=NUM_MODALITY_FEATURES, dtype=embedding_dtype
        )
        return modality_mask_oh @ macro_pi_logits

    def _forward_action_head(
        self,
        macro_action_embeddings: jax.Array,
        micro_action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        flat_valid_mask = valid_mask.reshape(-1)

        macro_pi_logits = self._forward_macro_head(macro_action_embeddings)
        micro_pi_logits = (
            self.micro_pi_head(micro_action_embeddings, micro_action_embeddings)
            .squeeze(-1)
            .reshape(-1)
        )

        pi_logits = (macro_pi_logits + micro_pi_logits) / temp

        hierarchical_prior = calculate_hierarchical_prior(flat_valid_mask)
        policy_metrics = compute_policy_metrics(
            logits=pi_logits, valid_mask=flat_valid_mask, prior=hierarchical_prior
        )

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(flat_valid_mask, pi_logits, -1e9), self.make_rng("sampling")
            )

        log_prob = jnp.take(policy_metrics.log_policy, action_index, axis=-1)

        mask_width = valid_mask.shape[-1]
        src_index = action_index // mask_width
        tgt_index = action_index % mask_width

        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            entropy=policy_metrics.entropy,
            normalized_entropy=policy_metrics.normalized_entropy,
            src_index=src_index,
            tgt_index=tgt_index,
            magnet_kl=policy_metrics.magnet_kl,
        )

    def _forward_value_head(self, value_embeddings: jax.Array):
        v_heads = self.v_head(value_embeddings, value_embeddings)

        value_gate = v_heads[..., 0].reshape(-1)
        value_gate_probs = nn.softmax(value_gate)

        value_logits = jax.lax.collapse(v_heads[..., 1:], 0, -1)

        agg_value_logits = value_gate_probs @ value_logits
        agg_value_log_probs = nn.log_softmax(agg_value_logits, axis=-1)
        agg_value_probs = jnp.exp(agg_value_log_probs)

        return CategoricalValueHeadOutput(
            logits=agg_value_logits,
            log_probs=agg_value_log_probs,
            entropy=-jnp.sum(agg_value_probs * agg_value_log_probs, axis=-1),
            expectation=agg_value_probs @ self.cfg.v_head.category_values,
        )

    def get_head_outputs(
        self,
        macro_action_embeddings: jax.Array,
        micro_action_embeddings: jax.Array,
        value_embeddings: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):

        action_head = self._forward_action_head(
            macro_action_embeddings,
            micro_action_embeddings,
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
        )
        value_head = self._forward_value_head(value_embeddings)

        return PlayerActorOutput(action_head=action_head, value_head=value_head)

    def __call__(
        self,
        actor_input: PlayerActorInput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        macro_action_embeddings, micro_action_embeddings, value_embeddings = (
            self.encoder(
                actor_input.env, actor_input.packed_history, actor_input.history
            )
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(
            macro_action_embeddings,
            micro_action_embeddings,
            value_embeddings,
            actor_input.env,
            actor_output,
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

        # axes_to_mean = tuple(range(value.ndim - 2))
        # avg_attn = jnp.sum(value, axis=axes_to_mean) / jnp.sum(
        #     value != 0, axis=axes_to_mean
        # ).clip(min=1)

        avg_attn = value[:, -1]  # Shape: (H, S, S)
        if avg_attn.ndim > 3:
            avg_attn = jnp.mean(avg_attn, 0)

        assert (
            avg_attn.ndim == 3
        ), f"Expected 3D array for {path_str} attention weights, got shape {avg_attn.shape}"

        avg_attn_np = np.asarray(avg_attn)

        # Use Plotly Express to facet the 3D array along the 0th dimension (Heads)
        fig = px.imshow(
            avg_attn_np,
            facet_col=0,  # Creates subplots for each index in the first dimension (H)
            facet_col_wrap=4,  # Wraps to a new row after 4 heads (great for 8 or 12 heads)
            text_auto=True,  # Tip: Set to False if your sequence length (S) is large
            aspect="auto",
            labels=dict(color="Attn Prob", facet_col="Head"),
            title=path_str,
        )

        # Clean up the subplot titles (Changes default "facet_col=0" to "Head 0")
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.replace("facet_col=", "Head "))
        )

        # --- File Saving Logic ---
        # 1. Ensure the base directory exists
        base_dir = "attn_weights"
        os.makedirs(base_dir, exist_ok=True)

        # 2. Create a safe file name from the path (excluding 'attn_weights')
        module_name = "_".join(str(k) for k in path_keys[:-1])

        # 3. Save to disk
        save_path = os.path.join(base_dir, f"{module_name}.html")
        fig.write_html(save_path)

        # Optional print to track progress during execution
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

    latest_ckpt = get_most_recent_file(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["player_state"]["params"]
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
