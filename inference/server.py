from dotenv import load_dotenv

load_dotenv()
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

app = FastAPI()


# Initialize the model
model = InferenceModel(
    generation=9,
    seed=secrets.randbits(32),
    # player_head_params=HeadParams(min_p=0.03, temp=0.5),
    # builder_head_params=HeadParams(min_p=0.03),
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

    ts = process_state(state, max_history=512)
    response = await run_in_threadpool(model.step, ts)
    pprint(response)

    return response


# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
