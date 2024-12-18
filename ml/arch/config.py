import pprint

from ml_collections import ConfigDict

from ml.arch.modules import GatingType


def get_model_cfg():
    cfg = ConfigDict()

    depth_factor = 0.2
    width_factor = 0.5

    base_entity_size = 256
    base_vector_size = 1024

    entity_size = int(base_entity_size * width_factor)
    vector_size = int(base_vector_size * width_factor)
    use_layer_norm = True

    # Encoder Configuration
    cfg.encoder = ConfigDict()
    cfg.encoder.entity_size = entity_size
    cfg.encoder.vector_size = vector_size

    # move encoder
    cfg.encoder.move_encoder = ConfigDict()
    cfg.encoder.move_encoder.entity_size = entity_size
    cfg.encoder.move_encoder.vector_size = vector_size

    # public entity encoder
    cfg.encoder.entity_encoder = ConfigDict()
    cfg.encoder.entity_encoder.entity_size = entity_size
    cfg.encoder.entity_encoder.vector_size = vector_size

    # public edge encoder
    cfg.encoder.edge_encoder = ConfigDict()
    cfg.encoder.edge_encoder.entity_size = entity_size
    cfg.encoder.edge_encoder.vector_size = vector_size

    transformer_num_layers = 1  # max(int(depth_factor * 3), 1)
    transformer_num_heads = 2

    # public context transformer
    #
    # transforms context from each turn into each public entity from that turn
    cfg.encoder.context_transformer = ConfigDict()
    cfg.encoder.context_transformer.hidden_size = entity_size
    cfg.encoder.context_transformer.num_layers = transformer_num_layers
    cfg.encoder.context_transformer.num_heads = transformer_num_heads
    cfg.encoder.context_transformer.use_layer_norm = use_layer_norm

    cfg.encoder.context_merge = ConfigDict()
    cfg.encoder.context_merge.gating_type = GatingType.NONE
    cfg.encoder.context_merge.use_layer_norm = use_layer_norm
    cfg.encoder.context_merge.output_size = vector_size

    # public history transformer
    #
    # contextualizes each entity along its previous history
    cfg.encoder.history_transformer = ConfigDict()
    cfg.encoder.history_transformer.hidden_size = entity_size
    cfg.encoder.history_transformer.num_layers = transformer_num_layers
    cfg.encoder.history_transformer.num_heads = transformer_num_heads
    cfg.encoder.history_transformer.use_layer_norm = use_layer_norm

    # action transformer
    cfg.encoder.action_transformer = ConfigDict()
    cfg.encoder.action_transformer.hidden_size = entity_size
    cfg.encoder.action_transformer.num_layers = transformer_num_layers
    cfg.encoder.action_transformer.num_heads = transformer_num_heads
    cfg.encoder.action_transformer.use_layer_norm = use_layer_norm

    # to vector
    #
    # pools the private entities into a single vector
    cfg.encoder.to_vector = ConfigDict()
    cfg.encoder.to_vector.units_hidden_sizes = (entity_size, vector_size)
    cfg.encoder.to_vector.output_stream_size = vector_size
    cfg.encoder.to_vector.use_layer_norm = use_layer_norm

    # state resnet
    #
    # resnet for feature extraction
    cfg.encoder.state_resnet = ConfigDict()
    cfg.encoder.state_resnet.num_resblocks = max(int(depth_factor * 8), 1)
    cfg.encoder.state_resnet.use_layer_norm = use_layer_norm

    # action merge
    #
    # merges the entity with its relevant action
    cfg.encoder.action_merge = ConfigDict()
    cfg.encoder.action_merge.gating_type = GatingType.NONE
    cfg.encoder.action_merge.use_layer_norm = use_layer_norm
    cfg.encoder.action_merge.output_size = entity_size

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.key_size = entity_size

    cfg.policy_head.query = ConfigDict()
    cfg.policy_head.query.num_resblocks = max(int(depth_factor * 2), 1)
    cfg.policy_head.query.use_layer_norm = use_layer_norm

    cfg.policy_head.pointer_logits = ConfigDict()
    cfg.policy_head.pointer_logits.num_layers_query = max(int(depth_factor * 1), 1)
    cfg.policy_head.pointer_logits.num_layers_keys = max(int(depth_factor * 3), 1)
    cfg.policy_head.pointer_logits.key_size = 64
    cfg.policy_head.pointer_logits.use_layer_norm = use_layer_norm
    cfg.policy_head.entity_size = entity_size
    cfg.policy_head.vector_size = vector_size

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
    pprint.pprint(cfg)


if __name__ == "__main__":
    main()
