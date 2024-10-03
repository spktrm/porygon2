import json
import pprint

from ml_collections import ConfigDict


def get_model_cfg():
    cfg = ConfigDict()

    depth_factor = 0.2
    width_factor = 0.25

    entity_size = int(256 * width_factor)
    vector_size = int(1024 * width_factor)
    use_layer_norm = True

    # Encoder Configuration
    cfg.encoder = ConfigDict()
    cfg.encoder.entity_size = entity_size
    cfg.encoder.vector_size = vector_size

    # move encoder
    cfg.encoder.move_encoder = ConfigDict()
    cfg.encoder.move_encoder.entity_size = entity_size

    cfg.encoder.public = ConfigDict()
    cfg.encoder.public.entity_size = entity_size

    # public entity encoder
    cfg.encoder.public.entity_encoder = ConfigDict()
    cfg.encoder.public.entity_encoder.entity_size = entity_size

    # public edge encoder
    cfg.encoder.public.edge_encoder = ConfigDict()
    cfg.encoder.public.edge_encoder.entity_size = entity_size

    transformer_num_layers = 1  # max(int(depth_factor * 3), 1)
    transformer_num_heads = 2

    # public context transformer
    #
    # transforms context from each turn into each public entity from that turn
    cfg.encoder.public.context_transformer = ConfigDict()
    cfg.encoder.public.context_transformer.stream_size = entity_size
    cfg.encoder.public.context_transformer.num_layers = transformer_num_layers
    cfg.encoder.public.context_transformer.num_heads = transformer_num_heads
    cfg.encoder.public.context_transformer.key_size = entity_size // 2
    cfg.encoder.public.context_transformer.value_size = entity_size // 2
    cfg.encoder.public.context_transformer.resblocks_hidden_size = entity_size // 2
    cfg.encoder.public.context_transformer.use_layer_norm = use_layer_norm

    # public history transformer
    #
    # contextualizes each entity along its previous history
    cfg.encoder.public.history_transformer = ConfigDict()
    cfg.encoder.public.history_transformer.stream_size = entity_size
    cfg.encoder.public.history_transformer.num_layers = transformer_num_layers
    cfg.encoder.public.history_transformer.num_heads = transformer_num_heads
    cfg.encoder.public.history_transformer.key_size = entity_size // 2
    cfg.encoder.public.history_transformer.value_size = entity_size // 2
    cfg.encoder.public.history_transformer.resblocks_hidden_size = entity_size // 2
    cfg.encoder.public.history_transformer.use_layer_norm = use_layer_norm

    # context transformer
    #
    # transforms context from each contextualized historical entity into the private entities
    # available for choice that turn
    cfg.encoder.context_transformer = ConfigDict()
    cfg.encoder.context_transformer.stream_size = entity_size
    cfg.encoder.context_transformer.num_layers = transformer_num_layers
    cfg.encoder.context_transformer.num_heads = transformer_num_heads
    cfg.encoder.context_transformer.key_size = entity_size // 2
    cfg.encoder.context_transformer.value_size = entity_size // 2
    cfg.encoder.context_transformer.resblocks_hidden_size = entity_size // 2
    cfg.encoder.context_transformer.use_layer_norm = use_layer_norm

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

    # Policy Head Configuration
    cfg.policy_head = ConfigDict()
    cfg.policy_head.key_size = entity_size

    cfg.policy_head.query = ConfigDict()
    cfg.policy_head.query.num_resblocks = max(int(depth_factor * 2), 1)
    cfg.policy_head.query.use_layer_norm = use_layer_norm

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
