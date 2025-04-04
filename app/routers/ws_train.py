from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import tensorflow as tf
import numpy as np
import requests
import json
from jose import jwt, JWTError
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os
from dotenv import load_dotenv

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

    async def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        data = {
            "type": "epoch_log",
            "epoch": self.epoch_count,
            "accuracy": round(logs["accuracy"] * 100, 2),
            "loss": round(logs["loss"], 4)
        }
        await self.websocket.send_json(data)

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
        user_id = user_info["user_id"]
        user_name = user_info["user_name"]

        response = requests.get("http://localhost:8000/code/generate", headers={"Authorization": f"Bearer {token}"})
        if response.status_code != 200:
            await websocket.send_json({"error": "코드 생성 실패"})
            await websocket.close()
            return

        result = response.json()
        form = result["form"]
        model_code = result["code"]
        model_name = result.get("model_name", "Unnamed Model")

        

        epochs = form["epochs"]
        batch_size = form["batch_size"]
        learning_rate = form["learning_rate"]

        exec_globals = {}
        try:
            exec(model_code, exec_globals)
            model = exec_globals["model"]
        except Exception as e:
            await websocket.send_json({"error": f"모델 실행 오류: {e}"})
            await websocket.close()
            return

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        x = np.random.randn(1000, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(1000,)).astype(np.int32)

        await websocket.send_json({"status": "학습 시작", "model_name": model_name})
        model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[WebSocketCallback(websocket)],
            verbose=0
        )

        y_pred_logits = model.predict(x)
        y_pred = np.argmax(y_pred_logits, axis=1)

        precision = round(precision_score(y, y_pred) * 100, 2)
        recall = round(recall_score(y, y_pred) * 100, 2)
        f1 = round(f1_score(y, y_pred) * 100, 2)
        auc = round(roc_auc_score(y, y_pred) * 100, 2)

        await websocket.send_json({
            "type": "final_metrics",
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        })

        await websocket.send_json({"status": "학습 완료"})
        await websocket.close()

    except WebSocketDisconnect:
        print(f"[{user_name}] WebSocket 연결 종료됨")
    except Exception as e:
        await websocket.send_json({"error": f"서버 오류: {e}"})
        await websocket.close()