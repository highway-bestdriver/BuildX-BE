from pydantic import BaseModel
from app.schemas.generate import ModelRequest

class FeedbackRequest(BaseModel):
    model: ModelRequest
    accuracy: float
    loss: float
