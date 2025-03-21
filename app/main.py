from fastapi import FastAPI
from app.database import engine, Base
import sys
import os
from app.auth.routes import router as auth_router
from app.routers import dashboard

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI()

# 라우터 등록
app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(dashboard.router, prefix='/dashboard', tags=["Dashboard"])

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/")
def home():
    return {"message": "FastAPI JWT 인증 API"}
