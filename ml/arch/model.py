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
from rlenv.env import clip_history, get_ex_step
from rlenv.interfaces import EnvStep, HistoryStep, ModelOutput


class Model(nn.Module):
    cfg: ConfigDict
    training: bool = True

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.policy_head = PolicyHead(self.cfg.policy_head)
        self.value_head = ValueHead(self.cfg.value_head)

    def _shared_forward(self, env_step: EnvStep, history_step: HistoryStep):
        """
        Shared forward pass for encoder and policy head.
        """
        # Get current state and action embeddings from the encoder
        latent_embeddings, action_embeddings = self.encoder(env_step, history_step)

        # Apply action argument heads
        logit, pi, log_pi = jax.vmap(self.policy_head)(
            latent_embeddings, action_embeddings, env_step.legal
        )
        return logit, pi, log_pi, latent_embeddings, action_embeddings

    def train_step(self, env_step: EnvStep, history_step: HistoryStep) -> ModelOutput:
        """
        Forward pass for training. Computes policy and value.
        """
        logit, pi, log_pi, latent_embeddings, _ = self._shared_forward(
            env_step, history_step
        )

        # Apply the value head
        value = jax.vmap(self.value_head)(latent_embeddings)
        value = jnp.tanh(value)

        # Return the model output
        return ModelOutput(logit=logit, pi=pi, log_pi=log_pi, v=value)

    def predict_step(self, env_step: EnvStep, history_step: HistoryStep) -> ModelOutput:
        """
        Forward pass for inference. Computes policy only. Value is None.
        """
        logit, pi, log_pi, *_ = self._shared_forward(env_step, history_step)

        # Return the model output, value is not computed
        return ModelOutput(logit=logit, pi=pi, log_pi=log_pi, v=None)

    def __call__(self, env_step: EnvStep, history_step: HistoryStep) -> ModelOutput:
        """
        Default forward pass.
        Calls train_step if training is True, otherwise calls predict_step.
        """
        if self.training:
            return self.train_step(env_step, history_step)
        else:
            return self.predict_step(env_step, history_step)


class DummyModel(nn.Module):

    @nn.compact
    def __call__(self, env_step: EnvStep, history_step: HistoryStep) -> ModelOutput:

        def _forward(env_step: EnvStep) -> ModelOutput:
            mask = env_step.legal.astype(jnp.float32)
            v = jnp.tanh(nn.Dense(1)(mask))
            logit = nn.Dense(mask.shape[-1])(mask)
            pi = legal_policy(logit, env_step.legal)
            log_pi = legal_log_policy(logit, env_step.legal)
            return ModelOutput(logit=logit, pi=pi, log_pi=log_pi, v=v)

        return jax.vmap(_forward)(env_step)


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


def get_model(config: ConfigDict = None, training: bool = True) -> nn.Module:
    if config is None:
        config = get_model_cfg()
    return Model(config, training)


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
    network = get_model(training=True)
    ex, hx = get_ex_step()
    hx = jax.tree.map(lambda x: x[:, None], hx)
    hx = clip_history(hx, resolution=64)
    hx = jax.tree.map(lambda x: x[:, 0], hx)

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ex, hx)

    network.apply(params, ex, hx)
    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
