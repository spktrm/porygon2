import functools
import math
import pickle
from pprint import pprint
from typing import Dict

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ml.arch.config import get_model_cfg
from ml.arch.encoder import Encoder
from ml.arch.heads import PolicyHead, ValueHead
from ml.func import legal_log_policy, legal_policy
from ml.utils import Params, get_most_recent_file
from rlenv.env import get_ex_step
from rlenv.interfaces import EnvStep, ModelOutput, TimeStep


class Model(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.policy_head = PolicyHead(self.cfg.policy_head)
        self.value_head = ValueHead(self.cfg.value_head)

    def get_head_outputs(
        self,
        latent_embeddings: jax.Array,
        action_embeddings: jax.Array,
        legal: jax.Array,
        temp: float = 1.0,
    ):
        # Apply action argument heads
        logit, pi, log_pi = self.policy_head(
            latent_embeddings, action_embeddings, legal, temp
        )

        # Apply the value head
        value = jnp.tanh(self.value_head(latent_embeddings))

        # Return the model output
        return ModelOutput(logit=logit, pi=pi, log_pi=log_pi, v=value)

    def __call__(self, timestep: TimeStep, temp: float = 1):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        latent_embeddings, action_embeddings = self.encoder(
            timestep.env, timestep.history
        )

        return jax.vmap(functools.partial(self.get_head_outputs, temp=temp))(
            latent_embeddings, action_embeddings, timestep.env.legal
        )


class DummyModel(nn.Module):

    @nn.compact
    def __call__(self, timestep: TimeStep) -> ModelOutput:

        def _forward(env_step: EnvStep) -> ModelOutput:
            mask = env_step.legal.astype(jnp.float32)
            v = jnp.tanh(nn.Dense(1)(mask))
            logit = nn.Dense(mask.shape[-1])(mask)
            masked_logits = jnp.where(mask, logit, -1e30)
            pi = legal_policy(logit, env_step.legal)
            log_pi = legal_log_policy(logit, env_step.legal)
            return ModelOutput(logit=masked_logits, pi=pi, log_pi=log_pi, v=v)

        return jax.vmap(_forward)(timestep.env)


def get_num_params(vars: Params, n: int = 3) -> Dict[str, Dict[str, float]]:
    def calculate_params(key: str, vars: Params) -> int:
        total = 0
        for key, value in vars.items():
            if isinstance(value, chex.Array):
                total += math.prod(value.shape)
            else:
                total += calculate_params(key, value)
        return total

    def build_param_dict(
        vars: Params, total_params: int, current_depth: int
    ) -> Dict[str, Dict[str, float]]:
        param_dict = {}
        for key, value in vars.items():
            if isinstance(value, chex.Array):
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


def get_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_model_cfg()
    return Model(config)


def get_dummy_model() -> nn.Module:
    return DummyModel()


def assert_no_nan_or_inf(gradients, path=""):
    if isinstance(gradients, dict):
        for key, value in gradients.items():
            new_path = f"{path}/{key}" if path else key
            assert_no_nan_or_inf(value, new_path)
    else:
        if jnp.isnan(gradients).any() or jnp.isinf(gradients).any():
            raise ValueError(f"Gradient at {path} contains NaN or Inf values.")


def main():
    network = get_model()
    ts = jax.tree.map(lambda x: x[:, 0], get_ex_step())

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ts)

    network.apply(params, ts)
    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
