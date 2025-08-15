from pydantic import BaseModel


class ResetResponse(BaseModel):
    tokens: list[int]
    v: float


Logits = list[float] | list[list[float]]


class StepResponse(BaseModel):
    action_type_logits: Logits
    move_logits: Logits
    wildcard_logits: Logits
    switch_logits: Logits
    v: float
    action_type: int
    move_slot: int
    switch_slot: int
    wildcard_slot: int
