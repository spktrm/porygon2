import json
import pprint

from ml_collections import ConfigDict

from ml.arch.modules import GatingType


def get_model_cfg():
    cfg = ConfigDict()

    depth_factor = 1
    width_factor = 0.5

    entity_size = int(256 * width_factor)
    vector_size = int(1024 * width_factor)
    use_layer_norm = True

    # Encoder Configuration
    cfg.encoder = ConfigDict()

    cfg.encoder.move_encoder = ConfigDict()
    cfg.encoder.move_encoder.entity_size = entity_size

    cfg.encoder.entity_encoder = ConfigDict()
    cfg.encoder.entity_encoder.entity_size = entity_size

    cfg.encoder.side_encoder = ConfigDict()
    cfg.encoder.side_encoder.entity_size = entity_size

    cfg.encoder.side_encoder.merge = ConfigDict()
    cfg.encoder.side_encoder.merge.output_size = vector_size // 2
    cfg.encoder.side_encoder.merge.gating_type = GatingType.NONE
    cfg.encoder.side_encoder.merge.use_layer_norm = use_layer_norm

    cfg.encoder.team_encoder = ConfigDict()
    cfg.encoder.team_encoder.transformer = ConfigDict()
    cfg.encoder.team_encoder.transformer.units_stream_size = entity_size
    cfg.encoder.team_encoder.transformer.transformer_num_layers = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.team_encoder.transformer.transformer_num_heads = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.team_encoder.transformer.transformer_key_size = entity_size // 2
    cfg.encoder.team_encoder.transformer.transformer_value_size = entity_size // 2
    cfg.encoder.team_encoder.transformer.resblocks_num_before = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.team_encoder.transformer.resblocks_num_after = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.team_encoder.transformer.resblocks_hidden_size = entity_size // 2
    cfg.encoder.team_encoder.transformer.use_layer_norm = use_layer_norm

    cfg.encoder.team_encoder.to_vector = ConfigDict()
    cfg.encoder.team_encoder.to_vector.units_hidden_sizes = (
        entity_size,
        int(entity_size * 3 / 2),
    )
    cfg.encoder.team_encoder.to_vector.use_layer_norm = use_layer_norm

    cfg.encoder.field_encoder = ConfigDict()
    cfg.encoder.field_encoder.vector_size = entity_size

    cfg.encoder.history_encoder = ConfigDict()
    cfg.encoder.history_encoder.entity_size = entity_size
    cfg.encoder.history_encoder.vector_size = vector_size
    cfg.encoder.history_encoder.transformer = ConfigDict()
    cfg.encoder.history_encoder.transformer.units_stream_size = vector_size
    cfg.encoder.history_encoder.transformer.transformer_num_layers = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.history_encoder.transformer.transformer_num_heads = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.history_encoder.transformer.transformer_key_size = vector_size // 2
    cfg.encoder.history_encoder.transformer.transformer_value_size = vector_size // 2
    cfg.encoder.history_encoder.transformer.resblocks_num_before = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.history_encoder.transformer.resblocks_num_after = max(
        int(depth_factor * 2), 1
    )
    cfg.encoder.history_encoder.transformer.resblocks_hidden_size = vector_size // 2
    cfg.encoder.history_encoder.transformer.use_layer_norm = use_layer_norm

    cfg.encoder.history_encoder.to_vector = ConfigDict()
    cfg.encoder.history_encoder.to_vector.units_hidden_sizes = (vector_size,)
    cfg.encoder.history_encoder.to_vector.use_layer_norm = use_layer_norm

    cfg.encoder.history_merge = ConfigDict()
    cfg.encoder.history_merge.output_size = vector_size
    cfg.encoder.history_merge.gating_type = GatingType.NONE
    cfg.encoder.history_merge.use_layer_norm = use_layer_norm

    cfg.encoder.state_merge = ConfigDict()
    cfg.encoder.state_merge.output_size = vector_size
    cfg.encoder.state_merge.gating_type = GatingType.POINTWISE
    cfg.encoder.state_merge.use_layer_norm = use_layer_norm

    cfg.encoder.state_resnet = ConfigDict()
    cfg.encoder.state_resnet.num_resblocks = max(int(depth_factor * 2), 1)
    cfg.encoder.state_resnet.use_layer_norm = use_layer_norm

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.key_size = entity_size

    cfg.policy_head.query = ConfigDict()
    cfg.policy_head.query.num_resblocks = max(int(depth_factor * 2), 1)
    cfg.policy_head.query.use_layer_norm = use_layer_norm

    cfg.policy_head.logits = ConfigDict()
    cfg.policy_head.logits.num_logits = 2
    cfg.policy_head.logits.use_layer_norm = use_layer_norm

    cfg.policy_head.pointer_logits = ConfigDict()
    cfg.policy_head.pointer_logits.num_layers_query = max(int(depth_factor * 1), 1)
    cfg.policy_head.pointer_logits.num_layers_keys = max(int(depth_factor * 3), 1)
    cfg.policy_head.pointer_logits.key_size = entity_size
    cfg.policy_head.pointer_logits.use_layer_norm = use_layer_norm

    # Value Head Configuration
    cfg.value_head = ConfigDict()
    cfg.value_head.resnet = ConfigDict()
    cfg.value_head.resnet.num_resblocks = max(int(depth_factor * 2), 1)
    cfg.value_head.resnet.use_layer_norm = use_layer_norm

    cfg.value_head.logits = ConfigDict()
    cfg.value_head.logits.num_logits = 1
    cfg.value_head.logits.use_layer_norm = use_layer_norm

    return cfg


def main():
    cfg = get_model_cfg()
    pprint.pprint(json.loads(cfg.to_json_best_effort()))


if __name__ == "__main__":
    main()
