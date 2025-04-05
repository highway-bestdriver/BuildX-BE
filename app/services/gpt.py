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
    레이어 구성:
    {layer_descriptions}
    
    전처리 정보:
    {preprocessing_options}

    요구사항:
    1. model = tf.keras.Sequential([...]) 또는 tf.keras.Model(...) 형태로 정의할 것
    2. 학습, 컴파일, 데이터 로드 등은 절대 포함하지 말 것
    3. 오직 모델 정의 코드만 반환할 것
    4. 반드시 model이라는 변수명으로 저장할 것
    5. import 문은 있어도 되고 없어도 되고, 실행에 지장 없게만 작성할 것
    6. 코드 외의 설명, 주석, 따옴표 등은 포함하지 말고 **오직 코드만** 반환할 것
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

