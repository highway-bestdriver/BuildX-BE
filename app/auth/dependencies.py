# 현재 로그인한 사용자를 가져오는 함수
from fastapi import Depends, HTTPException
from jose import jwt, JWTError
from app.auth.services import SECRET_KEY, ALGORITHM
from app.database import get_db
from sqlalchemy.orm import Session
from app.models.user import User
from fastapi.security import OAuth2PasswordBearer

# OAuth2PasswordBearer: JWT 토큰을 가져오기 위한 FastAPI 내장 기능
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """ JWT 토큰에서 현재 로그인한 유저 정보를 가져오는 함수 """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다.")
    except JWTError:
        raise HTTPException(status_code=401, detail="토큰 검증에 실패했습니다.")

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")

    return user