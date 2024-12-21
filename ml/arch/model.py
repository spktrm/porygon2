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
from ml.utils import Params, get_most_recent_file
from rlenv.data import NUM_ENTITY_FIELDS, NUM_MOVE_FIELDS
from rlenv.env import get_ex_step
from rlenv.interfaces import EnvStep, ModelOutput
from rlenv.protos.features_pb2 import FeatureEdge, FeatureEntity, FeatureMoveset


class Model(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """
        Initializes the encoder, policy head, and value head using the configuration.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.policy_head = PolicyHead(self.cfg.policy_head)
        self.value_head = ValueHead(self.cfg.value_head)

    def _preprocess(self, env_step: EnvStep) -> EnvStep:
        player_id = env_step.player_id
        leading_dims = env_step.valid.shape[:-1]

        moveset = env_step.moveset.reshape((*leading_dims, 2, -1, NUM_MOVE_FIELDS))
        moveset.at[..., FeatureMoveset.MOVESET_SIDE].set(
            moveset[..., FeatureMoveset.MOVESET_SIDE] ^ player_id
        )

        team = env_step.team.reshape((*leading_dims, 2, 6, NUM_ENTITY_FIELDS))
        team.at[..., FeatureEntity.ENTITY_SIDE].set(
            team[..., FeatureEntity.ENTITY_SIDE] ^ player_id
        )

        history_edges = env_step.history_edges
        edge_affecting_side = history_edges[..., FeatureEdge.EDGE_AFFECTING_SIDE]
        history_edges.at[..., FeatureEdge.EDGE_AFFECTING_SIDE].set(
            jnp.where(
                edge_affecting_side < 2,
                edge_affecting_side ^ player_id,
                edge_affecting_side,
            )
        )

        history_entities = env_step.history_entities
        history_entities.at[..., FeatureEntity.ENTITY_SIDE].set(
            history_entities[..., FeatureEntity.ENTITY_SIDE] ^ player_id
        )

        return EnvStep(
            ts=env_step.ts,
            draw_ratio=env_step.draw_ratio,
            valid=env_step.valid,
            draw=env_step.draw,
            turn=env_step.turn,
            game_id=env_step.game_id,
            player_id=env_step.player_id,
            seed_hash=env_step.seed_hash,
            moveset=moveset,
            legal=env_step.legal,
            team=team,
            heuristic_action=env_step.heuristic_action,
            win_rewards=env_step.win_rewards,
            fainted_rewards=env_step.fainted_rewards,
            switch_rewards=env_step.switch_rewards,
            longevity_rewards=env_step.longevity_rewards,
            hp_rewards=env_step.hp_rewards,
            history_edges=history_edges,
            history_entities=history_entities,
            history_side_conditions=env_step.history_side_conditions,
            history_field=env_step.history_field,
        )

    def __call__(self, env_step: EnvStep) -> ModelOutput:
        """
        Forward pass for the Model. It first processes the env_step through the encoder,
        and then applies the policy and value heads to generate the output.
        """
        env_step = self._preprocess(env_step)

        # Get current state and action embeddings from the encoder
        (contextual_entity_embeddings, valid_entity_mask, action_embeddings) = (
            self.encoder(env_step)
        )

        # Apply action argument heads
        logit, pi, log_pi = self.policy_head(action_embeddings, env_step)

        # Apply the value head
        v = self.value_head(contextual_entity_embeddings, valid_entity_mask)

        # Return the model output
        return ModelOutput(logit=logit, pi=pi, log_pi=log_pi, v=v)


class DummyModel(nn.Module):

    @nn.compact
    def __call__(self, env_step: EnvStep) -> ModelOutput:
        mask = env_step.legal.astype(jnp.float32)
        v = nn.Dense(1)(mask)
        logit = log_pi = pi = mask / mask.sum()
        return ModelOutput(logit=logit, pi=pi, log_pi=log_pi, v=v)


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


if __name__ == "__main__":
    main()
