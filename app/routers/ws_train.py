from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import tensorflow as tf
import numpy as np
import requests
import json
from jose import jwt, JWTError
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os
from dotenv import load_dotenv
import textwrap
import asyncio
from starlette.websockets import WebSocketState

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
COLAB_API_URL = os.getenv("COLAB_API_URL")

router = APIRouter()

# JWT 디코딩 함수
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("id")
        user_name = payload.get("name")
        if not user_id or not user_name:
            raise ValueError("토큰 정보 부족")
        return {"user_id": user_id, "user_name": user_name}
    except JWTError:
        raise ValueError("유효하지 않은 토큰입니다.")

# TensorFlow 콜백
class WebSocketCallback(tf.keras.callbacks.Callback):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self.epoch_count = 0

    async def safe_send_json(self, data):
        try:
            if self.websocket.application_state == WebSocketState.CONNECTED:
                await self.websocket.send_json(data)
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):  # async 제거했으
        self.epoch_count += 1
        data = {
            "type": "epoch_log",
            "epoch": self.epoch_count,
            "accuracy": round(logs["accuracy"] * 100, 2),
            "loss": round(logs["loss"], 4)
        }
        # 이벤트 루프에서 비동기 함수 실행
        #asyncio.create_task(self.websocket.send_json(data))
        asyncio.create_task(self.safe_send_json(data))


@router.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()

    try:
        token = websocket.query_params.get("token")

        # 프론트에서 JSON 받기
        data = await websocket.receive_json()
        model_code = data.get("code")
        model_code = textwrap.dedent(model_code)
        dataset = data.get("dataset")   # 확인해야함
        form = data.get("form")
        model_name = data.get("model_name", "Unnamed Model")

        if not token:
            await websocket.send_json({"error": "Access token이 필요합니다."})
            await websocket.close()
            return

        user_info = verify_token(token)
        user_id = user_info["user_id"]
        user_name = user_info["user_name"]

        epochs = form["epochs"]
        batch_size = form["batch_size"]
        learning_rate = form["learning_rate"]

        exec_globals = {}

        try:
            exec(model_code, exec_globals)
            model = exec_globals["model"]
            x_train = exec_globals["x_train"]
            y_train = exec_globals["y_train"]
            x_test = exec_globals["x_test"]
            y_test = exec_globals["y_test"]
        except Exception as e:
            await websocket.send_json({"error": f"모델 실행 오류: {e}"})
            await websocket.close()
            return


        # 학습 시간 단축용
        x_train = x_train[:100]
        y_train = y_train[:100]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"]
        )

        await websocket.send_json({"status": "학습 시작", "model_name": model_name})
        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[WebSocketCallback(websocket)],
            verbose=0
        )

        y_pred_logits = model.predict(x_test)
        y_pred = np.argmax(y_pred_logits, axis=1)

        y_test = np.argmax(y_test, axis=1)

        precision = round(precision_score(y_test, y_pred, average="macro") * 100, 2)
        recall = round(recall_score(y_test, y_pred, average="macro") * 100, 2)
        f1 = round(f1_score(y_test, y_pred, average="macro") * 100, 2)

        loss, accuracy = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test), verbose=0)
        loss = round(loss, 4)
        accuracy = round(accuracy * 100, 2)

        await websocket.send_json({
            "type": "final_metrics",
            "accuracy": accuracy,
            "loss": loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"status": "학습 완료"})

    except WebSocketDisconnect:
        print(f"[{user_name}] WebSocket 연결 종료됨")
    except Exception as e:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"error": f"서버 오류: {e}"})
            await websocket.close()

    # 웹소켓 닫기
    if websocket.application_state == WebSocketState.CONNECTED:
        await websocket.close()