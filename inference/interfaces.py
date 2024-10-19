from typing import List

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    pi: List[float]
    v: float
    action: int
