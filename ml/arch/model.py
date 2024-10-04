import math
import pickle
from pprint import pprint
from typing import Any, Dict

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from ml.arch.config import get_model_cfg
from ml.arch.encoder import Encoder
from ml.arch.heads import PolicyHead, ValueHead
from ml.utils import Params, get_most_recent_file
from rlenv.env import get_ex_step
from rlenv.interfaces import ActorStep, EnvStep, ModelOutput, TimeStep


class Model(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.policy_head = PolicyHead(self.cfg.policy_head)
        self.value_head = ValueHead(self.cfg.value_head)

    def __call__(self, env_step: EnvStep) -> ModelOutput:
        """
        Forward pass for the Model. It first processes the env_step through the encoder,
        and then applies the policy and value heads to generate the output.
        """
        # Get current state and action embeddings from the encoder
        current_state, action_embeddings = self.encoder(env_step)

        # Apply the policy head
        logit, pi, log_pi = self.policy_head(
            current_state, action_embeddings, env_step.legal
        )

        # Apply the value head
        v = self.value_head(current_state)

        # Return the model output
        return ModelOutput(pi=pi, v=v, log_pi=log_pi, logit=logit)


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


def get_model(config: ConfigDict) -> nn.Module:
    return Model(config)


def assert_no_nan_or_inf(gradients, path=""):
    if isinstance(gradients, dict):
        for key, value in gradients.items():
            new_path = f"{path}/{key}" if path else key
            assert_no_nan_or_inf(value, new_path)
    else:
        if jnp.isnan(gradients).any() or jnp.isinf(gradients).any():
            raise ValueError(f"Gradient at {path} contains NaN or Inf values.")


def test(params: Any):
    from ml.config import VtraceConfig
    from ml.learner import Learner
    from rlenv.utils import stack_steps

    config = get_model_cfg()
    network = get_model(config)

    learner = Learner(network, VtraceConfig())

    learner.params = params
    learner.params_target = params
    learner.params_prev = params
    learner.params_prev_ = params

    batch_size = 8

    batch = stack_steps(
        [
            TimeStep(
                env=jax.tree.map(
                    lambda x: x[None].repeat(batch_size, 0),
                    get_ex_step(),
                ),
                actor=ActorStep(
                    action=np.random.randint(0, 9, (batch_size,)),
                    policy=np.random.random((batch_size, 10)),
                    win_rewards=np.zeros((batch_size, 2)),
                    hp_rewards=np.zeros((batch_size, 2)),
                    fainted_rewards=np.zeros((batch_size, 2)),
                ),
            )
            for _ in range(64)
        ]
    )
    logs = learner.step(batch)
    pprint(logs)


def main():
    config = get_model_cfg()
    network = get_model(config)
    ex_step = get_ex_step()

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, ex_step)

    network.apply(params, ex_step)
    pprint(get_num_params(params))

    test(params)


if __name__ == "__main__":
    main()
