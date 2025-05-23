from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.gpt import generate_model_code
from app.schemas.generate import LayerUnion, TransformUnion, HyperParameters, ModelRequest
import os
from app.services.validator import validate_code

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
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
def generate_model(request: ModelRequest, token: str = Depends(oauth2_scheme)):
    user_name = decode_token(token)
    # GPT에게 코드 생성 요청
    generated_code = generate_model_code(
        model_name=request.model_name,
        layers=[layer.dict() for layer in request.layers],
        dataset=request.dataset,
        preprocessing=[preprocessing.dict() for preprocessing in request.preprocessing],
        hyperparameters=request.hyperparameters.dict() if request.hyperparameters else {},
    )

    #코드 유효성 검증
    validation_result = validate_code(request, generated_code)

    response = {
        "model_name": request.model_name,
        "form": request.hyperparameters,
        "code": generated_code,
        "use_cloud": request.hyperparameters.use_cloud if request.hyperparameters else False,
        "error": validation_result["valid"]
    }

    if not validation_result["valid"]:
        response["error"] = validation_result

    return response