from pydantic import BaseModel
from datetime import datetime

class ModelResponse(BaseModel):
    id: int
    name: str
    last_modified: str  # 여기서만 써야 함

    class Config:
        from_attributes = True