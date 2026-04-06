import json
import os

_dir = os.path.dirname(__file__)

with open(os.path.join(_dir, "data.json"), "r") as f:
    _data = json.load(f)

NUM_HISTORY: int = _data["NUM_HISTORY"]
MAX_RATIO_TOKEN: int = _data["MAX_RATIO_TOKEN"]
