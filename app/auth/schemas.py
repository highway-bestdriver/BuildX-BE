from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    username: str
    name: str
    phone: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class GoogleLoginRequest(BaseModel):
    token: str
    phone: Optional[str] = None  # 프론트에서 함께 보내주는 경우