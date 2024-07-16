import uvicorn

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

from inference.interfaces import PredictionResponse
from model import InferenceModel

from rlenv.env import Environment
from rlenv.protos.state_pb2 import State

app = FastAPI()

# Initialize the model
model = InferenceModel()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    data = await request.body()
    state = State.FromString(data)
    stage = 0

    while True:
        env_step = Environment.get_env_step(state, stage)
        # Make predictions using the model
        response = await run_in_threadpool(model.predict, env_step)

        if response.action == 4:
            print(response)
            stage = 1
        else:
            break

    return response


# Example usage
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8080)
