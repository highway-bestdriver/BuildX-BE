from fastapi import APIRouter, Depends, HTTPException
from jose import jwt, JWTError
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.gpt import generate_model_code
from app.schemas.generate import LayerUnion, Preprocessing, HyperParameters, ModelRequest
import os
from app.auth.dependencies import oauth2_scheme_no_client

router = APIRouter()
# JWT 디코딩 정보
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = os.getenv("ALGORITHM")
def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/generate")
def generate_model(request: ModelRequest, token: str = Depends(oauth2_scheme_no_client)):
    user_name = decode_token(token)
    # GPT에게 코드 생성 요청
    generated_code = generate_model_code(
        model_name=request.model_name,
        layers=[layer.dict() for layer in request.layers],
        dataset=request.dataset,
        preprocessing=request.preprocessing.dict() if request.preprocessing else {},
        hyperparameters=request.hyperparameters.dict() if request.hyperparameters else {},
    )

    return {
        "model_name": request.model_name,
        "form": request.hyperparameters,
        "code": generated_code,
    }