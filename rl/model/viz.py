"""Model inspection utilities: attention-map dumps and compile-cost analysis.

Kept out of player_model.py so the model definition has no plotting or
checkpoint-loading dependencies.
"""

import functools
import os
from pprint import pprint

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px

from rl.environment.interfaces import PlayerActorInput, PlayerActorOutput
from rl.environment.utils import get_ex_player_step
from rl.learner import checkpoint
from rl.model.config import get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model


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


def _load_or_init_params(learner_network: nn.Module, generation: int, key: jax.Array):
    ex_actor_input, ex_actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    )
    latest_ckpt = checkpoint.most_recent_ckpt_dir(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        params = checkpoint.load_component(latest_ckpt, "player", "params")
    else:
        params = learner_network.init(
            key, ex_actor_input, ex_actor_output, HeadParams()
        )
    return params, ex_actor_input, ex_actor_output


def dump_attention_maps(generation: int = 9):
    """Render every sown attention map of the latest checkpoint to html."""
    actor_network = get_player_model(get_player_model_config(generation, train=False))
    learner_network = get_player_model(get_player_model_config(generation, train=True))
    key = jax.random.key(42)
    params, ex_actor_input, ex_actor_output = _load_or_init_params(
        learner_network, generation, key
    )

    actor_output = actor_network.apply(
        params,
        ex_actor_input,
        PlayerActorOutput(),
        HeadParams(temp=0.8),
        rngs={"sampling": key},
    )
    pprint(actor_output)

    get_attention_maps(
        model=actor_network,
        params=params,
        actor_input=ex_actor_input,
        actor_output=actor_output,
        head_params=HeadParams(temp=0.8),
        rng_key=key,
    )


def cost_analysis(generation: int = 9):
    """Print the XLA cost analysis of one learner forward pass."""
    learner_network = get_player_model(get_player_model_config(generation, train=True))
    key = jax.random.key(42)
    params, ex_actor_input, ex_actor_output = _load_or_init_params(
        learner_network, generation, key
    )

    f = functools.partial(learner_network.apply, params)
    z = jax.jit(f).lower(ex_actor_input, ex_actor_output, HeadParams())
    print(z.compile().cost_analysis())


def main(generation: int = 9):
    dump_attention_maps(generation)
    cost_analysis(generation)


if __name__ == "__main__":
    main()
