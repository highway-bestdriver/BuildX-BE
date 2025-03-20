from pydantic import BaseModel
from datetime import datetime

class ModelResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy 모델을 Pydantic 모델로 변환