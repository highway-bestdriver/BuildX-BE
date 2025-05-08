from app.celery_worker import celery_app
from fastapi import APIRouter, WebSocket
from app.services.gpt import generate_model_code
from app.utils.pubsub import subscribe_log
from app.task.train import run_training
from jose import jwt, JWTError
import os, asyncio, json, builtins

router = APIRouter()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM  = os.getenv("ALGORITHM", "HS256")


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": payload["id"], "user_name": payload["name"]}
    except JWTError:
        raise ValueError("유효하지 않은 토큰입니다.")


@router.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    builtins.print("WebSocket 연결 완료")

    try:
        # 인증
        token = websocket.query_params.get("token")
        if not token:
            await websocket.send_json({"error": "Access token이 필요합니다."})
            return

        user_info = verify_token(token)
        user_id   = str(user_info["user_id"])
        builtins.print(f"인증된 유저 ID: {user_id}")

        # 학습 요청 수신
        data = await websocket.receive_json()
        builtins.print(f"학습 요청 수신: {data}")

        if "code" in data:# 직접 코드 전송
            model_code = data["code"]
            form       = data["form"]
            use_cloud  = data.get("use_cloud", False)
        else: # GPT 코드 생성
            model_code = generate_model_code(
                model_name    = data["model_name"],
                layers        = data["layers"],
                dataset       = data["dataset"],
                preprocessing = data.get("preprocessing", {}),
                hyperparameters = data["hyperparameters"],
            )

            form      = data["hyperparameters"]
            use_cloud = data.get("use_cloud", False)

        # Celery 태스크 발행
        try:
            res = run_training.delay(
                user_id,
                model_code,
                form["epochs"],
                form["batch_size"],
                form["learning_rate"],
                use_cloud,
            )
            builtins.print(f"[WS] Celery task queued, id = {res.id}")
        except Exception as exc:
            builtins.print(f"[WS] Celery publish ERROR → {exc}")
            await websocket.send_json({"error": f"Celery publish 실패: {exc}"})
            return

        # Redis 구독 설정
        try:
            pubsub = subscribe_log(f"user:{user_id}")
            builtins.print(f"Redis 채널 구독: user:{user_id}")
        except Exception as exc:
            builtins.print(f"Redis 구독 중 오류: {exc}")
            await websocket.send_json({"error": f"Redis 연결 실패: {exc}"})
            return

        # 실시간 로그 스트리밍
        while True:
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
            if msg:
                decoded = msg["data"].decode("utf-8")
                await websocket.send_text(decoded)
                builtins.print(f"로그 전송: {decoded}")
            await asyncio.sleep(0.5)

    except Exception as exc:
        builtins.print(f"[WS] 처리 중 오류: {exc}")
        # 소켓이 이미 닫혔을 수 있으므로 send 시도는 감싸둬야된대여
        try:
            await websocket.send_json({"error": str(exc)})
        except RuntimeError:
            pass

    finally:
        await websocket.close()
        builtins.print("WebSocket 종료")