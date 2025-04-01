import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi import APIRouter
import requests

router = APIRouter()

@router.post("/train")
def train_model():
    # 1. 백엔드 내부 요청 - GPT 코드 생성 API 호출
    response = requests.get("http://localhost:8000/code/generate")  # 주소는 실제 서버 기준
    if response.status_code != 200:
        return {"error": "코드 생성 실패"}

    result = response.json()
    model_code = result["code"]
    form = result["form"]

    epochs = form["epochs"]
    batch_size = form["batch_size"]
    learning_rate = form["learning_rate"]
    device_type = form["device_type"]

    # 2. GPU/CPU 선택
    device = torch.device("cuda" if device_type == "gpu" and torch.cuda.is_available() else "cpu")

    # 3. 모델 실행
    exec_globals = {}
    try:
        exec(model_code, exec_globals)
        model = exec_globals["model"]  # 코드 안에서 model 객체 만들어줘야 함
    except Exception as e:
        return {"error": f"모델 코드 실행 오류: {e}"}

    model.to(device)

    # 4. 예시 데이터셋
    x = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5. 학습 시작
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    all_acc = []
    all_loss = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

        acc = correct / total
        avg_loss = total_loss / len(dataloader)
        all_acc.append(round(acc * 100, 2))
        all_loss.append(round(avg_loss, 4))

    # 6. 최종 지표 예시
    final_metrics = {
        "정확도(Accuracy)": round(all_acc[-1], 2),
        "loss": round(all_loss[-1], 4),
        "f1 score": 85,
        "정밀도(Precision)": 60,
        "재현율(Recall)": 71,
        "AUC-ROC": 71
    }

    return {
        "metrics": final_metrics,
        "all_accuracy": all_acc,
        "all_loss": all_loss
    }