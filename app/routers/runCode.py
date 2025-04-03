from fastapi import APIRouter, Request
import tensorflow as tf
import numpy as np
import requests
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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
        model = exec_globals["model"]
    except Exception as e:
        return {"error": f"모델 코드 실행 오류: {e}"}

    # 3. 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 4. 예시 데이터셋 생성
    x = np.random.randn(1000, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(1000,)).astype(np.int32)

    # 5. 학습 수행
    history = model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    all_acc = [round(acc * 100, 2) for acc in history.history["accuracy"]]
    all_loss = [round(loss, 4) for loss in history.history["loss"]]

    # 6. 메트릭 계산 (예측 기반)
    y_pred_logits = model.predict(x)
    y_pred = np.argmax(y_pred_logits, axis=1)

    try:
        precision = round(precision_score(y, y_pred) * 100, 2)
        recall = round(recall_score(y, y_pred) * 100, 2)
        f1 = round(f1_score(y, y_pred) * 100, 2)
        auc = round(roc_auc_score(y, y_pred) * 100, 2)
    except Exception as e:
        return {"error": f"메트릭 계산 오류: {e}"}

    final_metrics = {
        "정확도(Accuracy)": all_acc[-1],
        "loss": all_loss[-1],
        "f1 score": f1,
        "정밀도(Precision)": precision,
        "재현율(Recall)": recall,
        "AUC-ROC": auc
    }

    return {
        "model_name": model_name,
        "metrics": final_metrics,
        "all_accuracy": all_acc,
        "all_loss": all_loss
    }