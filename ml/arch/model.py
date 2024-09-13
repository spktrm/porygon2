import jax
import math
import chex
import pickle
import jax.numpy as jnp
import flax.linen as nn

from pprint import pprint
from typing import Dict
from ml_collections import ConfigDict

from ml.arch.config import get_model_cfg
from ml.arch.encoder import Encoder, SupervisedBackbone
from ml.arch.heads import PolicyHead, ValueHead
from ml.utils import Params, get_most_recent_file

from rlenv.env import get_ex_step
from rlenv.interfaces import EnvStep, ModelOutput


class Model(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.policy_head = PolicyHead(self.cfg.policy_head)
        self.value_head = ValueHead(self.cfg.value_head)

    def __call__(self, env_step: EnvStep) -> ModelOutput:
        current_state, my_action_embeddings = self.encoder(env_step)
        logit, pi, log_pi = self.policy_head(
            current_state, my_action_embeddings, env_step.legal
        )
        v = self.value_head(current_state)
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


# def main2():
#     model_cfg = get_model_cfg().encoder
#     model = SupervisedBackbone(model_cfg)

#     with open("replays/bad_batch", "rb") as f:
#         bad_batch = pickle.load(f)

#     key = jax.random.key(42)

#     params = model.init(key, get_ex_step())
#     output = jax.vmap(model.apply, (None, 0))(params, bad_batch)
#     return output


def main():
    config = get_model_cfg()
    network = get_model(config)

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            step = pickle.load(f)
        params = step["params"]
    else:
        key = jax.random.key(42)
        params = network.init(key, get_ex_step())

    network.apply(params, get_ex_step())

    pprint(get_num_params(params))


if __name__ == "__main__":
    main()
