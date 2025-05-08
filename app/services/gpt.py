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
    - `layers` 필드는 사용자가 만든 모델 구조로, 각 항목은 하나의 레이어를 의미함.
    - `type`: 레이어 타입 (예: "Conv2d", "ReLU", "BatchNorm2d", ...)
    - `name`: 레이어 이름 (코드에서 self.에 들어갈 이름)
    - `inputs`: 이전 레이어의 이름 (str 또는 List[str]) → 이걸 따라 forward() 내에서 연결해야 함
    - `SequentialLayer`는 내부에 하위 레이어를 나열함 (layers 필드 사용)

    2. 데이터 전처리
    - `preprocessing`은 torchvision.transforms.v2 기반의 전처리 리스트임
    - `SequentialTransform`은 Compose처럼 동작함
    - 최종적으로 transforms = v2.Sequential([...]) 형태로 생성해야 함

    3. 코드 형식
    - PyTorch `nn.Module` 클래스를 사용해서 `class Net(nn.Module):`으로 정의해
    - `__init__()`에서는 self.conv1 = ... 식으로 레이어 선언
    - `forward()`에서는 `inputs`를 따라 레이어를 연결해
    - 데이터셋은 torchvision.datasets 사용 가능
    - 출력은 실행 가능한 전체 코드로 작성 (exec 가능해야 함)
    - `model`, `x_train`, `y_train`, `x_test`, `y_test`는 모두 코드 내에서 정의되어야 함
    - 학습 (`fit`, `optimizer`, `loss`)은 작성하지 말 것

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


def generate_model_feedback(model: dict, accuracy: float, loss: float) -> str:
    print(model.get("dateset"))
    prompt = f"""
    너는 딥러닝 전문가야. 아래는 사용자가 설계한 CNN 모델 구조와 그 학습 결과를 정리 해놓은 거야.
    사용자는 딥러닝 초보이고, 모델 구조를 블록 형태로 쌓아 딥러닝 모델을 제작했어.
    이러한 초보자 맞춤 모델 개선 피드백을 부탁해.

    1. 모델 이름: {model.get("model_name")}

    2. 학습에 사용한 데이터셋: {model.get("dataset")}

    3. 모델 구조:
    {json.dumps(model.get("layers", []), indent=2)}

    4. 전처리 설정:
    {json.dumps(model.get("preprocessing", {}), indent=2)}

    5. 하이퍼파라미터:
    {json.dumps(model.get("hyperparameters", {}), indent=2)}

    6. 학습 성능:
    - Accuracy: {accuracy}
    - Loss: {loss}

    위 정보를 기반으로 모델의 성능을 분석해줘. 그리고,
    1. 구조 개선 팁 2~3가지 (과적합, 성능 향상 등)
    2. 하이퍼파라미터 추천
    3. 필요한 경우 추가할 수 있는 레이어
    4. 이 성능이 나온 원인 분석
    이런 내용을 한국어로 핵심만 간결하고 딥러닝 초보자가 이해하기 쉽게 알려줘.
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

