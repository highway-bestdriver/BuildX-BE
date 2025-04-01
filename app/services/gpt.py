import openai
import os
import json
import re

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
    너는 tensorflow 전문가야. 아래 JSON을 바탕으로 학습 전체 tensorflow 코드를 작성해줘. 
    
    모델 구조 (블록 구성):
    {layer_descriptions}
    
    클래스별 이미지 파일들:
    {dataset_description}
    
    전처리 옵션:
    {preprocessing_options}
    
    하이퍼파라미터:
    {hyperparam_settings}
    
    요구사항:
    - ImageDataGenerator 또는 tf.data 로 전처리
    - 학습 loop 포함
    - 정확도/손실 시각화 포함
    - 구조 혹은 매개변수에 오류가 있다면 설명없이 "Error"라는 메세지만 반환
    - 반드시 TensorFlow 2.x 기반 코드로 작성
    - Keras의 레이어는 정확한 이름만 사용 (예: Conv2D, MaxPooling2D, Dense, Dropout 등)
    - tf.keras.layers 에 없는 함수는 절대 쓰지 마
    - 설명 없이 파이썬 코드만 반환 (주석, 코드 블록 없이)
    """

    try:
        response = client.chat.completions.create(
            model="tinyllama",
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

