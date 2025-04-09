from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from app.schemas.feedback import FeedbackRequest
from app.services.gpt import generate_model_feedback
import os

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# JWT 디코딩 설정
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

@router.post("/feedback")
def feedback_from_gpt(request: FeedbackRequest, token: str = Depends(oauth2_scheme)):
    username = decode_token(token)
    feedback = generate_model_feedback(
        model=request.model.dict(),
        accuracy=request.accuracy,
        loss=request.loss
    )
    return {
        "username": username,
        "feedback": feedback
    }
