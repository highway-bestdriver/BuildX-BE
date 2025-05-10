from pydantic import BaseModel
from app.schemas.generate import ModelRequest
from typing import Dict
class Metrics(BaseModel):
    epoch: int
    train_acc: float
    train_loss: float
    test_acc: float
    test_loss: float
    test_precision: float
    test_recall: float
    test_f1: float
class FeedbackRequest(BaseModel):
    model: ModelRequest
    metrics: Metrics
