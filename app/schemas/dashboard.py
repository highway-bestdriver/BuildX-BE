from pydantic import BaseModel
from typing import List

class ModelResponse(BaseModel):
    id: int
    name: str
    last_modified: str

    class Config:
        from_attributes = True

class ModelDetailRecord(BaseModel):
    id: int
    epoch: int
    batch_size: int
    learning_rate: float
    loss: float
    accuracy: float
    date: str  # 문자열 or "YYYY.MM.DD"

    class Config:
        from_attributes = True

class ModelDetailResponse(BaseModel):
    id: int
    name: str
    date: str
    details: List[ModelDetailRecord]

    class Config:
        from_attributes = True