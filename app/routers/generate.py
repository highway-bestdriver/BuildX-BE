from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.gpt import generate_model_code
from app.schemas.generate import Layer, Preprocessing, HyperParameters, ModelRequest

router = APIRouter()

@router.post("/generate")
def generate_model(request: ModelRequest):
    # GPT에게 코드 생성 요청
    generated_code = generate_model_code(
        model_name=request.model_name,
        layers=[layer.dict() for layer in request.layers],
        dataset=request.dataset,
        preprocessing=request.preprocessing.dict() if request.preprocessing else {},
        hyperparameters=request.hyperparameters.dict() if request.hyperparameters else {}
    )

    return {
        "model_name": request.model_name,
        "forms": request.hyperparameters,
        "code": generated_code
    }