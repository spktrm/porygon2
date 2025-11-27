from pydantic import BaseModel


class ResetResponse(BaseModel):
    species_indices: list[int]
    packed_set_indices: list[int]
    v: float


class StepResponse(BaseModel):
    v: float
    action: int
    wildcard: int
