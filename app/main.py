from fastapi import FastAPI
from app.database import engine, Base
import sys
import os
from app.auth.routes import router as auth_router
from app.routers import dashboard, generate, runCode, ws_train
from fastapi.middleware.cors import CORSMiddleware
from app.routers import feedback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 미들웨어 등록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 프론트 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용 (POST, GET, OPTIONS 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# 라우터 등록
app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(dashboard.router, prefix='/dashboard', tags=["Dashboard"])
app.include_router(generate.router, prefix="/code", tags=["Model Generation"])
app.include_router(runCode.router, prefix="/code", tags=["RunningCode"])
app.include_router(ws_train.router)
app.include_router(feedback.router, prefix="/code", tags=["GPT Feedback"])

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# @app.get("/")
# def home():
#     return {"message": "FastAPI JWT 인증 API"}

