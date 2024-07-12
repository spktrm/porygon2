import functools
import json
import pprint

import flax.linen as nn

from ml.arch.encoder import (
    Encoder,
    EntityEncoder,
    FieldEncoder,
    HistoryEncoder,
    MoveEncoder,
    SideEncoder,
    TeamEncoder,
)
from ml.arch.heads import PolicyHead, ValueHead
from ml.arch.interfaces import ModuleConfigDict
from ml.arch.modules import (
    GatingType,
    Logits,
    Resnet,
    ToAvgVector,
    Transformer,
    VectorMerge,
    PointerLogits,
)


def get_move_encoder_config(entity_size: int, **kwargs):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    cfg.module_fns = {
        "move_linear": functools.partial(nn.Dense, features=entity_size),
    }
    return cfg


def get_entity_encoder_config(entity_size: int, **kwargs):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    cfg.module_fns = {
        "onehot_linear": functools.partial(nn.Dense, features=entity_size),
        "species_linear": functools.partial(nn.Dense, features=entity_size),
        "ability_linear": functools.partial(nn.Dense, features=entity_size),
        "item_linear": functools.partial(nn.Dense, features=entity_size),
        "moveset_linear": functools.partial(nn.Dense, features=entity_size),
    }
    return cfg


def get_side_encoder_config(
    entity_size: int, vector_size: int, use_layer_norm: bool, **kwargs
):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    cfg.module_fns = {
        "linear": functools.partial(nn.Dense, features=entity_size),
        "merge": functools.partial(
            VectorMerge,
            output_size=vector_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
        ),
    }
    return cfg


def get_team_encoder_config(
    entity_size: int, vector_size: int, use_layer_norm: bool, **kwargs
):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    cfg.module_fns = {
        "transformer": functools.partial(
            Transformer,
            units_stream_size=entity_size,
            transformer_num_layers=2,
            transformer_num_heads=2,
            transformer_key_size=entity_size // 2,
            transformer_value_size=entity_size // 2,
            resblocks_num_before=2,
            resblocks_num_after=2,
            resblocks_hidden_size=entity_size // 2,
            use_layer_norm=use_layer_norm,
        ),
        "to_vector": functools.partial(
            ToAvgVector,
            units_hidden_sizes=(entity_size, vector_size),
            use_layer_norm=use_layer_norm,
        ),
    }
    return cfg


def get_field_encoder_config(vector_size: int, **kwargs):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    cfg.module_fns = {
        "linear": functools.partial(nn.Dense, features=vector_size),
    }
    return cfg


def get_history_encoder_config(vector_size: int, use_layer_norm: bool, **kwargs):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    cfg.module_fns = {
        "transformer": functools.partial(
            Transformer,
            units_stream_size=vector_size,
            transformer_num_layers=2,
            transformer_num_heads=2,
            transformer_key_size=vector_size // 2,
            transformer_value_size=vector_size // 2,
            resblocks_num_before=2,
            resblocks_num_after=2,
            resblocks_hidden_size=vector_size // 2,
            use_layer_norm=use_layer_norm,
        ),
        "to_vector": functools.partial(
            ToAvgVector,
            units_hidden_sizes=(vector_size,),
            use_layer_norm=use_layer_norm,
        ),
    }
    return cfg


def get_encoder_config(**kwargs):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    vector_size = kwargs.get("vector_size")
    use_layer_norm = kwargs.get("use_layer_norm")
    cfg.module_fns = {
        "move_encoder": functools.partial(
            MoveEncoder, get_move_encoder_config(**kwargs)
        ),
        "entity_encoder": functools.partial(
            EntityEncoder, get_entity_encoder_config(**kwargs)
        ),
        "team_encoder": functools.partial(
            TeamEncoder, get_team_encoder_config(**kwargs)
        ),
        "side_encoder": functools.partial(
            SideEncoder, get_side_encoder_config(**kwargs)
        ),
        "field_encoder": functools.partial(
            FieldEncoder, get_field_encoder_config(**kwargs)
        ),
        "history_merge": functools.partial(
            VectorMerge,
            output_size=vector_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
        ),
        "history_encoder": functools.partial(
            HistoryEncoder, get_history_encoder_config(**kwargs)
        ),
        "state_merge": functools.partial(
            VectorMerge,
            output_size=vector_size,
            gating_type=GatingType.NONE,
            use_layer_norm=use_layer_norm,
        ),
    }
    return cfg


def get_value_head_cfg(**kwargs):
    cfg = ModuleConfigDict()
    cfg.constants = {**kwargs}
    depth_factor = kwargs.get("depth_factor")
    cfg.module_fns = {
        "resnet": functools.partial(
            Resnet,
            num_resblocks=max(int(depth_factor * 2), 1),
            use_layer_norm=True,
        ),
        "logits": functools.partial(Logits, num_logits=1, use_layer_norm=True),
    }
    return cfg


def get_policy_head_cfg(
    entity_size: int, vector_size: int, use_layer_norm: bool, **kwargs
):
    cfg = ModuleConfigDict()
    cfg.constants = dict(key_size=entity_size)
    depth_factor = kwargs.get("depth_factor")
    cfg.module_fns = {
        "state_query": functools.partial(
            Resnet,
            num_resblocks=max(int(depth_factor * 2), 1),
            use_layer_norm=use_layer_norm,
        ),
        "action_logits": functools.partial(
            PointerLogits,
            num_layers_query=1,
            num_layers_keys=2,
            key_size=entity_size,
            use_layer_norm=use_layer_norm,
        ),
        "select_logits": functools.partial(
            PointerLogits,
            num_layers_query=1,
            num_layers_keys=2,
            key_size=entity_size,
            use_layer_norm=use_layer_norm,
        ),
    }
    return cfg


def get_model_cfg():
    cfg = ModuleConfigDict()

    depth_factor = 0.5
    width_factor = 0.5
    default_constants = dict(
        entity_size=int(128 * width_factor),
        vector_size=int(512 * width_factor),
        depth_factor=depth_factor,
        width_factor=width_factor,
        use_layer_norm=True,
    )

    cfg.constants = default_constants
    cfg.module_fns = {
        "encoder": functools.partial(Encoder, get_encoder_config(**default_constants)),
        "value_head": functools.partial(
            ValueHead, get_value_head_cfg(**default_constants)
        ),
        "policy_head": functools.partial(
            PolicyHead, get_policy_head_cfg(**default_constants)
        ),
    }
    return cfg


def main():
    cfg = get_model_cfg()
    pprint.pprint(json.loads(cfg.to_json_best_effort()))


if __name__ == "__main__":
    main()
