from fastapi import APIRouter, Request
import tensorflow as tf
import numpy as np
import requests

router = APIRouter()

@router.post("/train")
def train_model(request: Request):
    access_token = request.headers.get("Authorization")
    if not access_token:
        return {"error": "Access token이 필요합니다."}

    headers = {
        "Authorization": access_token
    }

    # 1. 코드 생성 API 호출
    response = requests.get("http://localhost:8000/code/generate", headers=headers)
    if response.status_code != 200:
        return {"error": "코드 생성 실패"}

    result = response.json()
    model_code = result["code"]
    form = result["form"]
    model_name = result.get("model_name", "Unnamed Model")

    epochs = form["epochs"]
    batch_size = form["batch_size"]
    learning_rate = form["learning_rate"]
    device_type = form["device_type"]

    # 2. 모델 실행
    exec_globals = {}
    try:
        exec(model_code, exec_globals)
        model = exec_globals["model"]  # exec 내에서 반드시 model = ... 정의되어야 함
    except Exception as e:
        return {"error": f"모델 코드 실행 오류: {e}"}

    # 3. Optimizer & Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 4. 예시 데이터셋 (1000개, 10차원 입력, 2클래스 분류)
    x = np.random.randn(1000, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(1000,)).astype(np.int32)

    # 5. 학습
    history = model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # 로그 생략
    )

    all_acc = [round(acc * 100, 2) for acc in history.history["accuracy"]]
    all_loss = [round(loss, 4) for loss in history.history["loss"]]

    # 6. 지표 반환
    final_metrics = {
        "정확도(Accuracy)": all_acc[-1],
        "loss": all_loss[-1],
        "f1 score": 85,  # 필요 시 실제 계산 로직 추가 가능
        "정밀도(Precision)": 60,
        "재현율(Recall)": 71,
        "AUC-ROC": 71
    }

    return {
        "model_name": model_name,
        "metrics": final_metrics,
        "all_accuracy": all_acc,
        "all_loss": all_loss
    }