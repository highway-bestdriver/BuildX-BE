from pydantic import BaseModel
from datetime import datetime

class ModelResponse(BaseModel):
    id: int
    name: str
    last_modified: str

    class Config:
        from_attributes = True