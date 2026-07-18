from dotenv import load_dotenv

from constants import NUM_HISTORY

load_dotenv()
import os
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


# Initialize the model. USE_SEARCH=1 enables decision-time equilibrium
# search (requires a checkpoint trained with the search heads);
# SEARCH_DEPTH raises the deploy-time budget (see cfg.search).
model = InferenceModel(
    generation=9,
    seed=secrets.randbits(32),
    player_head_params=HeadParams(temp=0.5),
    builder_head_params=HeadParams(temp=1.0),
    use_search=os.environ.get("USE_SEARCH", "0") == "1",
    search_overrides={"depth": int(os.environ.get("SEARCH_DEPTH", "1"))},
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
