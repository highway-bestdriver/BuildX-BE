from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.utils.pubsub import subscribe_log
from app.task.train import run_training
from jose import jwt, JWTError
import os
import json
import asyncio

router = APIRouter()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": payload["id"], "user_name": payload["name"]}
    except JWTError:
        raise ValueError("유효하지 않은 토큰입니다.")

@router.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결 수락 완료")

    try:
        token = websocket.query_params.get("token")
        print(f"토큰 확인: {token}")
        if not token:
            await websocket.send_json({"error": "Access token이 필요합니다."})
            await websocket.close()
            return

        user_info = verify_token(token)
        user_id = str(user_info["user_id"])
        print(f"인증된 유저: {user_id}")

        data = await websocket.receive_json()
        print(f"학습 요청 수신 완료! {data}")

        model_code = data["code"]
        form = data["form"]

        # Celery task 시작
        run_training.delay(
            user_id,
            model_code,
            form["epochs"],
            form["batch_size"],
            form["learning_rate"]
        )
        print("Celery 태스크 실행!")

        pubsub = subscribe_log(f"user:{user_id}")
        print(f"Redis 구독 시작: user:{user_id}")

        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
            if message:
                print(f"Redis 메시지 수신: {message}")
                await websocket.send_text(message["data"].decode("utf-8"))
            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"WebSocket 오류: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
        print("WebSocket 종료")