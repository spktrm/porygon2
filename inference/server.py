from pprint import pprint

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from model import InferenceModel

from inference.interfaces import PredictionResponse
from rlenv.env import process_state
from rlenv.protos.state_pb2 import State

app = FastAPI()


# Initialize the model
model = InferenceModel()


def pprint_nparray(arr: np.ndarray):
    print(np.array2string(arr, precision=3, suppress_small=True))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    data = await request.body()
    state = State.FromString(data)

    ex, hx = process_state(state)
    response = await run_in_threadpool(model.predict, ex, hx)
    pprint(state.info)

    pprint_nparray(np.array(ex.moveset[..., 0]))
    pprint_nparray(np.array(response.pi))
    pprint_nparray(np.array(response.logit))
    pprint_nparray(np.array(response.v))
    pprint_nparray(np.array(response.action))

    return response


# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
