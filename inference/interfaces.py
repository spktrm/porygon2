from typing import List

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    pi: List[float]
    logit: List[float]
    v: float
    action: int
