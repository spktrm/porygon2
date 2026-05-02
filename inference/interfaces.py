from pydantic import BaseModel


class ResetResponse(BaseModel):
    packed_team: list[int]
    v: float


class StepResponse(BaseModel):
    v_win: float
    log_prob: float
    entropy: float
    src: int
    tgt: int
