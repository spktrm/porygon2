from typing import List
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    data: bytes


class PredictionResponse(BaseModel):
    pi: List[float]
    v: float
    log_pi: List[float]
    logit: List[float]
    action: int
