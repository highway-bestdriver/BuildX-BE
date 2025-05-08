from pydantic import BaseModel
from app.schemas.generate import ModelRequest
from typing import Dict

class FeedbackRequest(BaseModel):
    model: ModelRequest
    metrics: Dict
