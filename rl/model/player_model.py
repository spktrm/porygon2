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

from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import (
    CategoricalValueLogitHead,
    HeadParams,
    PointerLogits,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.utils import get_most_recent_file, get_num_params


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, value head, and Q-value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.action_head = PointerLogits(**self.cfg.action_head.qk_logits.to_dict())
        self.winloss_head = CategoricalValueLogitHead(self.cfg.winloss_head)
        self.q_value_head = PointerLogits(**self.cfg.q_value_head.qk_logits.to_dict())

    def _forward_action_head(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        head_logits = self.action_head(action_embeddings, action_embeddings)
        logits = head_logits.squeeze(-1).reshape(-1) / temp

        flat_valid_mask = valid_mask.reshape(-1)

        policy_metrics = compute_policy_metrics(
            logits=logits, valid_mask=flat_valid_mask, prior=None
        )

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(
                jnp.where(flat_valid_mask, logits, -1e9), self.make_rng("sampling")
            )

        log_prob = jnp.take(policy_metrics.log_policy, action_index, axis=-1)

        mask_width = valid_mask.shape[-1]
        src_index = action_index // mask_width
        tgt_index = action_index % mask_width

        # Compute q_values using the same PointerLogits structure as policy logits
        q_logits = self.q_value_head(action_embeddings, action_embeddings)
        q_values = q_logits.squeeze(-1).reshape(-1)
        q_values = jnp.where(flat_valid_mask, q_values, -1e9)

        action_head_output = PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            entropy=policy_metrics.entropy,
            normalized_entropy=policy_metrics.normalized_entropy,
            src_index=src_index,
            tgt_index=tgt_index,
            magnet_kl=policy_metrics.magnet_kl,
        )

        return action_head_output, q_values

    def _forward_value_head(self, state_embedding: jax.Array):
        return self.winloss_head(state_embedding)

    def get_head_outputs(
        self,
        value_embedding: jax.Array,
        action_embeddings: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):

        action_head, q_values = self._forward_action_head(
            action_embeddings,
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
        )
        value_head = self._forward_value_head(value_embedding)

        return PlayerActorOutput(
            action_head=action_head, value_head=value_head, q_values=q_values
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
        # Get current state and action embeddings from the encoder
        value_embedding, action_embeddings = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(value_embedding, action_embeddings, actor_input.env, actor_output)


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
