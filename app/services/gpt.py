import openai
import os
import json
import re

os.environ["OLLAMA_NUM_GPU"] = "0"  # CPU 사용하도록 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
# test용으로 ollama tinyllama 사용
client = openai.OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# gpt 답변에서 코드만 추출하는 함수
def extract_code_only(text):
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def generate_model_code(model_name: str, layers: list, dataset: str, preprocessing: dict, hyperparameters: dict):
    # GPT에게 줄 prompt 만들기
    layer_descriptions = json.dumps(layers, indent=2)
    dataset_description = json.dumps(dataset, indent=2)
    preprocessing_options = json.dumps(preprocessing, indent=2)
    hyperparam_settings = json.dumps(hyperparameters, indent=2)

    prompt = f"""
    너는 tensorflow 전문가야. 아래 JSON 정보를 기반으로,
    학습이 아니라 **모델 정의만 포함된 코드**를 `model = ...` 형식으로 작성해줘.
    ---
    <요구사항>
    1. 아래 JSON은 각 레이어의 블록 정보를 담고 있어. 각 레이어는 고유 id와 input 필드를 가지고 있어.
    2. 반드시 'input' 필드를 따라 레이어 간 연결을 정의해야 해.
    3. 모델은 반드시 `model = tf.keras.Model(inputs=..., outputs=...)` 형태로 작성할 것.
    4. 오직 모델 정의 코드만 작성하고 학습, 컴파일, 데이터 로딩은 포함하지 말 것.
    5. `import` 문은 있어도 되고 없어도 되고, 실행 가능한 수준만 되면 됨.
    6. **코드 외 설명은 절대 포함하지 말고** `model = ...` 포함한 코드만 반환할 것.
    7. 최종 모델 변수명은 반드시 `model`로 저장할 것.
    
    <JSON 구조>
    레이어 구조:
    {layer_descriptions}
    
    전처리 설정:
    {preprocessing_options}
    
    하이퍼파라미터:
    {hyperparam_settings}
    
    데이터셋:
    {dataset_description}
    """

    try:
        response = client.chat.completions.create(
            model="mistral",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Tensorflow code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        raw_output = response.choices[0].message.content.strip()
        return extract_code_only(raw_output)

    except Exception as e:
        return f"# GPT 호출 실패: {str(e)}"


def generate_model_feedback(model: dict, accuracy: float, loss: float) -> str:
    prompt = f"""
    너는 딥러닝 전문가야. 아래는 사용자가 설계한 CNN 모델 구조와 그 학습 결과를 정리 해놓은 거야.
    사용자는 딥러닝 초보이고, 모델 구조를 블록 형태로 쌓아 딥러닝 모델을 제작했어.
    이러한 초보자 맞춤 모델 개선 피드백을 부탁해.

    1. 모델 이름: {model.get("model_name")}

    2. 학습에 사용한 데이터셋: {model.get("dataset_name")}

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
            model="mistral",  # 또는 gpt-4o
            messages=[
                {"role": "system", "content": "당신은 CNN 모델을 분석하고 개선을 제안하는 AI 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"# GPT 피드백 생성 실패: {str(e)}"

