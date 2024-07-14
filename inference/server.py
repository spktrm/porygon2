import traceback
from fastapi import FastAPI, HTTPException, Request

from inference.interfaces import PredictionResponse
from model import InferenceModel

from rlenv.env import process_state
from rlenv.protos.state_pb2 import State

app = FastAPI()

# Initialize the model
model = InferenceModel()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    try:
        data = await request.body()
        state = State.FromString(data)
        env_step = process_state(state)
        return model.predict(env_step)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Example usage
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
