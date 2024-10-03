from pprint import pprint

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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    data = await request.body()
    state = State.FromString(data)

    env_step = process_state(state, stage=0)
    response = await run_in_threadpool(model.predict, env_step)
    pprint(state.info)
    pprint(response.pi)

    if response.action == 4:
        env_step = process_state(state, stage=1)
        response = await run_in_threadpool(model.predict, env_step)
        pprint(response.pi)

    return response


# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
