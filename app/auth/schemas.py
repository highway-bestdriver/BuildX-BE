from pydantic import BaseModel

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