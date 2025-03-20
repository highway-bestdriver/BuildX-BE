from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.auth import schemas, services
from app.models.user import User
from datetime import timedelta
from jose import jwt
from app.auth.services import SECRET_KEY, ALGORITHM, JWTError
from app.auth.dependencies import get_current_user

router = APIRouter()

@router.post("/signup")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # 전화번호 포맷 정리 (하이픈과 공백 제거)
    normalized_phone = user.phone.replace("-", "").replace(" ", "")

    # 아이디 중복 확인 (대소문자 무시)
    existing_user_by_id = db.query(User).filter(User.username.ilike(user.username)).first()
    if existing_user_by_id:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")

    # 전화번호 중복 확인
    existing_user_by_phone = db.query(User).filter(User.phone == normalized_phone).first()
    if existing_user_by_phone:
        raise HTTPException(status_code=400, detail="이미 가입한 회원입니다.")

    # 비밀번호 해싱
    hashed_password = services.hash_password(user.password)

    # 유저 저장
    new_user = User(
        username=user.username,
        name=user.name,
        phone=normalized_phone,  # 정리된 전화번호 저장
        password_hash=hashed_password,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "회원가입 성공!"}

@router.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()

    if not db_user or not services.verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    # Access Token
    access_token = services.create_access_token(
        data={"sub": db_user.username},
        expires_delta=timedelta(minutes=30),
    )

    # Refresh Token
    refresh_token = services.create_refresh_token(
        data={"sub": db_user.username},
        expires_delta=timedelta(days=7),  # 유효기간 : 7일
    )

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

# Refresh Token을 사용하여 새로운 Access Token을 발급
@router.post("/refresh")
def refresh_token(request: schemas.RefreshTokenRequest):
    try:
        # Refresh Token 디코딩 및 검증
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")

        # 새로운 Access Token 발급 (만료 시간 설정)
        new_access_token = services.create_access_token(
            data={"sub": username},
            expires_delta=timedelta(minutes=30)  # Access Token 유효 기간 설정
        )

        return {"access_token": new_access_token}

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
@router.post("/change-password")
def change_password(
    request: schemas.ChangePasswordRequest,  # ✅ JSON Body로 받음
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_user = db.query(User).filter(User.username == current_user.username).first()

    if not db_user or not services.verify_password(request.current_password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="현재 비밀번호가 올바르지 않습니다.")

    # 새로운 비밀번호 해싱
    hashed_new_password = services.hash_password(request.new_password)

    # 비밀번호 업데이트
    db_user.password_hash = hashed_new_password
    db.commit()
    db.refresh(db_user)

    return {"message": "비밀번호가 성공적으로 변경되었습니다."}

# fastapi는 서버측에서 특별한 작업 없이 단순히 성공 메세지 반환해주면 클라이언트에서 저장된 토큰을 삭제하면 됨.
@router.post("/logout")
def logout():
    return {"message": "로그아웃 성공!"}
