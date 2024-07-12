import functools

from ml.arch.heads import PolicyHead, ValueHead
from ml.arch.interfaces import ModuleConfigDict
from ml.arch.modules import Logits, Resnet


def get_value_head_cfg(
    entity_size: int, vector_size: int, depth_factor: int, width_factor: int
):
    value_head_cfg = ModuleConfigDict()
    value_head_cfg.constants = dict(entity_size=entity_size, vector_size=vector_size)
    value_head_cfg.module_fns = {
        "resnet": functools.partial(
            Resnet,
            num_resblocks=max(int(depth_factor * 2), 1),
            use_layer_norm=True,
        ),
        "logits": functools.partial(Logits, num_logits=1, use_layer_norm=True),
    }
    return value_head_cfg


def get_policy_head_cfg(
    entity_size: int, vector_size: int, depth_factor: int, width_factor: int
):
    policy_head_cfg = ModuleConfigDict()
    policy_head_cfg.constants = dict(entity_size=entity_size, vector_size=vector_size)
    policy_head_cfg.module_fns = {
        "resnet": functools.partial(
            Resnet,
            num_resblocks=max(int(depth_factor * 2), 1),
            use_layer_norm=True,
        ),
        "logits": functools.partial(Logits, num_logits=1, use_layer_norm=True),
    }
    return policy_head_cfg


def get_model_cfg():
    model_cfg = ModuleConfigDict()

    depth_factor = 1
    width_factor = 2
    default_constants = dict(
        entity_size=int(128 * width_factor),
        vector_size=int(512 * width_factor),
        depth_factor=depth_factor,
        width_factor=width_factor,
    )

    model_cfg.constants = default_constants
    model_cfg.module_fns = {
        "value_head": functools.partial(
            ValueHead, get_value_head_cfg(**default_constants)
        ),
        "policy_head": functools.partial(
            PolicyHead, get_policy_head_cfg(**default_constants)
        ),
    }
    return model_cfg


def main():
    print(get_model_cfg())


if __name__ == "__main__":
    main()
