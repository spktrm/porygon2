from pydantic import BaseModel


class ResetResponse(BaseModel):
    tokens: list[int]
    log_pi: list[float]
    entropy: list[float]
    key: list[int]
    v: list[float]


class StepResponse(BaseModel):
    pi: list[float]
    log_pi: list[float]
    logit: list[float]
    v: float
    action: int
