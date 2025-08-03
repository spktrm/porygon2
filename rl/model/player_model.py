import functools
import math
import pickle
from pprint import pprint
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PolicyHeadOutput,
)
from rl.environment.utils import get_player_ex_step
from rl.model.config import get_model_config
from rl.model.encoder import Encoder
from rl.model.heads import PolicyHead, ScalarHead
from rl.model.utils import BIAS_VALUE, Params, legal_log_policy, legal_policy
from rl.utils import init_jax_jit_cache


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)

        self.action_type_head = ScalarHead(self.cfg.action_type_head)
        self.move_head = PolicyHead(self.cfg.move_head)
        self.switch_head = PolicyHead(self.cfg.switch_head)

        self.value_head = ScalarHead(self.cfg.value_head)

    def get_head_outputs(
        self,
        entity_embeddings: jax.Array,
        action_embeddings: jax.Array,
        entity_mask: jax.Array,
        env_step: PlayerEnvOutput,
        temp: float = 1.0,
    ):
        action_type_head_logits = self.action_type_head(entity_embeddings, entity_mask)
        action_type_policy = legal_policy(
            action_type_head_logits, env_step.action_type_mask, temp
        )
        action_type_log_policy = legal_log_policy(
            action_type_head_logits, env_step.action_type_mask, temp
        )

        move_embeddings = action_embeddings[:4]
        switch_embeddings = action_embeddings[4:]

        # Apply the value head
        value = jnp.tanh(self.value_head(entity_embeddings, entity_mask))

        # Return the model output
        return PlayerActorOutput(
            action_type_head=PolicyHeadOutput(
                logits=jnp.where(
                    env_step.action_type_mask, action_type_head_logits, BIAS_VALUE
                ),
                policy=action_type_policy,
                log_policy=action_type_log_policy,
            ),
            move_head=self.move_head(
                move_embeddings,
                entity_embeddings,
                env_step.move_mask,
                entity_mask,
                temp,
            ),
            switch_head=self.switch_head(
                switch_embeddings,
                entity_embeddings,
                env_step.switch_mask,
                entity_mask,
                temp,
            ),
            v=value,
        )

    def __call__(self, timestep: PlayerActorInput, temp: float = 1.0):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        entity_embeddings, action_embeddings, entity_mask = self.encoder(
            timestep.env, timestep.history
        )

        return jax.vmap(functools.partial(self.get_head_outputs, temp=temp))(
            entity_embeddings,
            action_embeddings,
            entity_mask,
            timestep.env,
        )


def get_num_params(vars: Params, n: int = 3) -> Dict[str, Dict[str, float]]:
    def calculate_params(key: str, vars: Params) -> int:
        total = 0
        for key, value in vars.items():
            if isinstance(value, jax.Array):
                total += math.prod(value.shape)
            else:
                total += calculate_params(key, value)
        return total

    def build_param_dict(
        vars: Params, total_params: int, current_depth: int
    ) -> Dict[str, Dict[str, float]]:
        param_dict = {}
        for key, value in vars.items():
            if isinstance(value, jax.Array):
                num_params = math.prod(value.shape)
                param_dict[key] = {
                    "num_params": num_params,
                    "ratio": num_params / total_params,
                }
            else:
                nested_params = calculate_params(key, value)
                param_entry = {
                    "num_params": nested_params,
                    "ratio": nested_params / total_params,
                }
                if current_depth < n - 1:
                    param_entry["details"] = build_param_dict(
                        value, total_params, current_depth + 1
                    )
                param_dict[key] = param_entry
        return param_dict

    total_params = calculate_params("base", vars)
    return build_param_dict(vars, total_params, 0)


def get_player_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_model_config()
    return Porygon2PlayerModel(config)


def assert_no_nan_or_inf(gradients, path=""):
    if isinstance(gradients, dict):
        for key, value in gradients.items():
            new_path = f"{path}/{key}" if path else key
            assert_no_nan_or_inf(value, new_path)
    else:
        if jnp.isnan(gradients).any() or jnp.isinf(gradients).any():
            raise ValueError(f"Gradient at {path} contains NaN or Inf values.")


def main():
    init_jax_jit_cache()
    network = get_player_model()
    ts = jax.device_put(jax.tree.map(lambda x: x[:, 0], get_player_ex_step()))

    latest_ckpt = None  # get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["player_state"]["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ts)

    network.apply(params, ts)
    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
