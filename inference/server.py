import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from model import InferenceModel
from rich.pretty import pprint

from inference.interfaces import ResetResponse, StepResponse
from rl.environment.env import process_state
from rl.environment.protos.service_pb2 import EnvironmentState
from rl.utils import init_jax_jit_cache

app = FastAPI()
init_jax_jit_cache()


# Initialize the model
model = InferenceModel(do_threshold=True)


def pprint_nparray(arr: np.ndarray):
    print(np.array2string(arr, precision=3, suppress_small=True))


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request):
    await request.body()

    response = await run_in_threadpool(model.reset)
    pprint(response)

    return response


@app.post("/step", response_model=StepResponse)
async def step(request: Request):
    data = await request.body()
    state = EnvironmentState.FromString(data)

    ts = process_state(state)
    response = await run_in_threadpool(model.step, ts)
    pprint(response)

    return response


# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
