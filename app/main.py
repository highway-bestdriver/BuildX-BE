from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
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

# OpenAPI 문서 커스터마이징 함수 추가
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="BuildX API",
        version="1.0.0",
        description="API for BuildX",
        routes=app.routes,
    )

    # client_id, client_secret 없는 인증 방식만 명시적으로 정의
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": "/auth/login",
                    "scopes": {}
                }
            }
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# 커스터마이징 적용
app.openapi = custom_openapi

origins = [
    "http://localhost:3000",
]


# CORS 미들웨어 등록
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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