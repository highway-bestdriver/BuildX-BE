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

def generate_model_code(model_name: str, layers: list, dataset: str, preprocessing: dict, hyperparameters: dict):
    # GPT에게 줄 prompt 만들기
    layer_descriptions = json.dumps(layers, indent=2)
    dataset_description = json.dumps(dataset, indent=2)
    preprocessing_options = json.dumps(preprocessing, indent=2)
    hyperparam_settings = json.dumps(hyperparameters, indent=2)

    prompt = f"""
    너는 tensorflow 전문가야. 아래 JSON 정보를 기반으로, 아래 모델 구조에 기반하여 텐서플로우 학습 코드의 '데이터 로드, 전처리, 모델 정의' 부분을 작성해줘.
    1. 데이터 로딩
    - dataset에 사용된 데이터를 불러오고, 여기에 아래 작성된 전처리를 적용해줘.
    
    2. 모델 정의
    - layers 필드에 나열된 CNN 레이어들을 순서대로 조합해 모델을 정의해야 해.
    - 레이어 간 연결은 `input` 필드로 따라가며 연결해야 함.
    - 첫 번째 레이어가 "type": "Input"인 경우, dataset을 보고 input_shape를 추론해서 넣어줘.
    
    3. 예시
    다음과 같은 예시처럼 구조에 맞게 코드를 작성해줘
    # 데이터 로딩 및 전처리
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    # 모델 정의 (Functional API)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    출력 양식:
    - `x_train`, `y_train`, `x_test`, `y_test`, `model` 이라는 변수들을 모두 정의해야 함
    - 이 변수들이 그대로 exec() 환경에서 쓰일 수 있도록 코드 전체를 하나의 실행 스크립트로 작성해
    - compile과 fit 코드는 작성할 필요 없어. 
    
    <모델 구조>
    레이어 구조:
    {layer_descriptions}
    
    전처리 설정:
    {preprocessing_options}
    
    데이터셋:
    {dataset_description}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Tensorflow code."},
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

