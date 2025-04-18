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

    try:
        token = websocket.query_params.get("token")
        if not token:
            await websocket.send_json({"error": "Access token이 필요합니다."})
            await websocket.close()
            return

        user_info = verify_token(token)
        user_id = str(user_info["user_id"])

        data = await websocket.receive_json()
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

        pubsub = subscribe_log(f"user:{user_id}")

        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
            if message:
                await websocket.send_text(message["data"].decode("utf-8"))
            await asyncio.sleep(0.5)

    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()