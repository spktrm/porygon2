from pydantic import BaseModel


class ResetResponse(BaseModel):
    tokens: list[int]
    log_pi: list[float]
    entropy: list[float]
    key: list[int]
    v: list[float]


class HeadOutput(BaseModel):
    logits: list[float]
    policy: list[float]
    log_policy: list[float]


class StepResponse(BaseModel):
    action_type_head: HeadOutput
    move_head: HeadOutput
    switch_head: HeadOutput
    v: float
    action_type: int
    move_slot: int
    switch_slot: int
