# ruff: noqa: E402 -- load deployment environment before model imports.
from dotenv import load_dotenv

from constants import NUM_HISTORY  # noqa: E402

load_dotenv()
import argparse
import secrets
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from rich.pretty import pprint

from inference.interfaces import ResetResponse, StepResponse
from inference.model import InferenceModel
from rl.environment.env import process_state
from rl.environment.protos.service_pb2 import EnvironmentState
from rl.model.heads import HeadParams

app = FastAPI()


parser = argparse.ArgumentParser(
    description="Play against a fixed checkpoint with optional search"
)
parser.add_argument(
    "--search", choices=("plain", "expectimax", "mcts"), default="plain"
)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--search-depth", type=int, default=2)
parser.add_argument("--simulations", type=int, default=64)
parser.add_argument("--chance-samples", type=int, default=4)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--device", choices=("cpu", "gpu"), default=None)
if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = parser.parse_args([])

model = InferenceModel(
    generation=9,
    seed=secrets.randbits(32),
    player_head_params=HeadParams(temp=args.temperature),
    fpath=args.checkpoint,
    search_mode=args.search,
    search_depth=args.search_depth,
    simulations=args.simulations,
    chance_samples=args.chance_samples,
    device=args.device,
    builder_head_params=HeadParams(temp=1.0),
)


def pprint_nparray(arr: np.ndarray) -> None:
    print(np.array2string(arr, precision=3, suppress_small=True))


@app.get("/ping", response_class=PlainTextResponse)
async def ping() -> Literal["pong"]:
    return "pong"


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request) -> ResetResponse:
    await request.body()

    response = await run_in_threadpool(model.reset)

    pprint(response)

    return response


@app.post("/step", response_model=StepResponse)
async def step(request: Request) -> StepResponse:
    data = await request.body()
    state = EnvironmentState.FromString(data)

    ts = process_state(state, max_history=NUM_HISTORY)
    response = await run_in_threadpool(model.step, ts)
    pprint(response)

    return response


# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
