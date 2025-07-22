from pprint import pprint

import numpy as np
import tabulate
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from model import InferenceModel

from inference.interfaces import PredictionResponse
from rl.environment.env import process_state
from rl.environment.protos.service_pb2 import EnvironmentState
from rl.utils import init_jax_jit_cache


app = FastAPI()
init_jax_jit_cache()


# Initialize the model
model = InferenceModel()


def pprint_nparray(arr: np.ndarray):
    print(np.array2string(arr, precision=3, suppress_small=True))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    data = await request.body()
    state = EnvironmentState.FromString(data)

    ts = process_state(state)
    response = await run_in_threadpool(model.predict, ts)

    pprint(ts.env.info)

    table = tabulate.tabulate(
        np.stack([response.logit, response.log_pi, response.pi]).clip(
            max=9.99, min=-9.99
        ),
        floatfmt=".2f",
    )
    print(table)
    pprint_nparray(np.array(response.v))
    pprint_nparray(np.array(response.action))

    return response


# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
