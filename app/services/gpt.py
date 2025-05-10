import openai
import os
import json
import re

os.environ["OLLAMA_NUM_GPU"] = "0"  # CPU 사용하도록 설정
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# test용으로 ollama 사용
#client = openai.OpenAI(
#    api_key="ollama",
#    base_url="http://localhost:11434/v1"
#)

# gpt 답변에서 코드만 추출하는 함수
def extract_code_only(text):
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def generate_model_code(model_name: str, layers: list, dataset: str, preprocessing: list, hyperparameters: dict):
    # GPT에게 줄 prompt 만들기
    layer_descriptions = json.dumps(layers, indent=2)
    dataset_description = json.dumps(dataset, indent=2)
    preprocessing_options = json.dumps(preprocessing, indent=2)
    hyperparam_settings = json.dumps(hyperparameters, indent=2)

    prompt = f"""
    아래 JSON 구조에 따라 PyTorch 모델 정의 및 데이터 전처리 코드를 작성해줘.

    요구사항:

    1. 모델 정의
    - `layers`: 필드는 사용자가 만든 모델 구조로, 각 항목은 하나의 레이어를 의미함.
    - `type`: 레이어 타입 (예: "Conv2d", "ReLU", "BatchNorm2d", ...)
    - `name`: 레이어 변수 이름 (예: self.conv1 에서 conv1)
    - `input`: 이 레이어가 어떤 이전 레이어의 출력을 입력으로 받는지를 나타냄 → 이걸 따라 forward() 내에서 연결해야 함
    - 그 외의 필드는 해당 레이어의 파라미터로, 반드시 torch.nn.{{type}} 생성자에 전달해야 함.

    2. 데이터 전처리
    - `preprocessing`은 torchvision.transforms.v2 기반의 전처리 리스트임
    - torchvision.transforms.v2 as transform를 사용하여 transforms.Compose([...]) 형식으로 구성하세요.
    - `SequentialTransform`이라는 이름의 클래스는 사용하지 마세요. 존재하지 않습니다.
    
    3. 데이터셋 및 로딩
    - `dataset` 필드는 어떤 데이터셋을 사용할지 나타냄. CIFAR10 또는 MNIST 등 torchvision.datasets 에서 불러와야 함
    - 로딩된 데이터에 `preprocessing`을 적용하고, `DataLoader`로 구성해야 함
    - `x_train`, `y_train`, `x_test`, `y_test`, `train_loader`, `test_loader`를 모두 정의해야 함

    4. 모델 학습 코드 포함
    - `model`, `criterion`, `optimizer` 정의
    - `train_loader`를 이용해 학습 루프 (`for epoch in range(...)`) 포함
    - 각 epoch마다 `loss`와 `accuracy`를 다음 형식으로 출력:
    print(f"epoch {{epoch}} | loss {{train_loss:.4f}} | acc {{train_acc:.4f}}")
    
    5. 테스트 평가 및 최종 출력
    학습 종료 후 `x_test` 또는 `test_loader`를 통해 `y_pred`를 구하고 `y_test`와 비교하여 다음 성능 지표를 계산:
    - test_acc, test_precision, test_recall, test_f1
    - 마지막에 다음 딕셔너리를 `print(json.dumps(...))` 형식으로 출력:

    ```python
    metrics = {{
        "type": "metric",
        "epoch": EPOCHS,
        "train_acc": ...,     # 마지막 epoch 기준
        "train_loss": ...,
        "test_acc": ...,
        "test_loss": ...,
        "test_precision": ...,
        "test_recall": ...,
        "test_f1": ...
    }}
    print(json.dumps(metrics, ensure_ascii=False))
    
    6. 실행 가능하게 작성
    - 전체 코드는 `exec()`로 실행 가능한 하나의 스크립트로 작성되어야 함
    - 모든 import, 모델 정의 (class Net(nn.Module)), 선언 포함해서 완전한 코드로 작성할 것
    - CUDA 사용이 가능한 경우에는 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 방식으로 모델과 입력을 이동
    - `model.to(device)` 및 `inputs.to(device)` 적용 필요
    
    주의: 사용자가 입력한 레이어 파라미터 값은 그대로 사용할 것. 잘못된 값이더라도 수정하거나 추측하지 말고, JSON의 값을 그대로 코드로 반영할 것.
    
    <모델 이름>
    {model_name}

    <레이어 구조>
    {layer_descriptions}

    <전처리 설정>
    {preprocessing_options}

    <데이터셋>
    {dataset_description}

    <하이퍼파라미터 설정>
    {hyperparam_settings}
    """

    # GPT 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes PyTorch code from model JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw_output = response.choices[0].message.content.strip()
        return extract_code_only(raw_output)

    except Exception as e:
        return f"# GPT 호출 실패: {str(e)}"


def generate_model_feedback(model: dict, metrics: dict) -> str:
    print(model.get("dateset"))
    prompt = f"""
    너는 딥러닝 전문가야. 아래는 한 초보자가 블록 기반 UI로 설계한 CNN 모델과 학습 결과야.
    사용자가 설계한 모델 구조, 전처리, 하이퍼파라미터, 성능 지표를 바탕으로 구조 개선 및 성능 향상에 대한 피드백을 제공해줘.
    목적: 초보자가 이해할 수 있도록 핵심 위주로 간단하고 직관적인 설명을 해줄 것.

    1. 모델 이름: {model.get("model_name")}

    2. 학습에 사용한 데이터셋: {model.get("dataset")}

    3. 모델 구조:
    {json.dumps(model.get("layers", []), indent=2, ensure_ascii=False)}

    4. 전처리 설정:
    {json.dumps(model.get("preprocessing", {}), indent=2, ensure_ascii=False)}

    5. 하이퍼파라미터:
    {json.dumps(model.get("hyperparameters", {}), indent=2, ensure_ascii=False)}

    6. 학습 및 테스트 성능 요약:
    - 학습 에포크: {metrics.get("epoch")}
    - 학습 정확도 (train_acc): {metrics.get("train_acc")}
    - 학습 손실값 (train_loss): {metrics.get("train_loss")}
    - 테스트 정확도 (test_acc): {metrics.get("test_acc")}
    - 테스트 정확도 (test_loss): {metrics.get("test_loss")}
    - 테스트 정밀도 (test_precision): {metrics.get("test_precision")}
    - 테스트 재현율 (test_recall): {metrics.get("test_recall")}
    - 테스트 F1 점수 (test_f1): {metrics.get("test_f1")}

    아래 내용을 중심으로 한국어 피드백 작성:
    1. 모델 구조 개선 팁 2~3가지 (예: 과적합 완화, 성능 향상 등)
    2. 하이퍼파라미터 조정 추천 (예: 학습률, 배치 크기, 에포크 수)
    3. 필요한 경우 추가하면 좋은 레이어 (예: Dropout, BatchNorm 등)
    4. 성능 결과에 기반한 원인 분석 (예: underfitting, 과적합, 모델 과단순 등)
    
    결과는 초보자가 이해할 수 있도록 간결하고 핵심만 알려줘.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 CNN 모델을 분석하고 개선을 제안하는 AI 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"# GPT 피드백 생성 실패: {str(e)}"

