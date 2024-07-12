import flax.linen as nn


from typing import Callable, Dict
from ml_collections import ConfigDict


class ModuleConfigDict(ConfigDict):
    constants: Dict[str, str | int | float]
    module_fns: Dict[str, Callable[[], nn.Module]]
