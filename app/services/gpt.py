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
    너는 tensorflow 전문가야. 아래 JSON 기반 정보를 바탕으로, 즉시 실행 가능한 학습 전체 코드를 작성해줘.

    ---
    모델 정보:
    {layer_descriptions}
    
    데이터셋:
    {dataset_description}
    
    전처리 설정:
    {preprocessing_options}
    
    하이퍼파라미터:
    {hyperparam_settings}
    ---
    
    요구사항:
    1. 데이터셋을 직접 로드하는 코드 포함 (예: keras.datasets.mnist 또는 사용자 경로)
    2. 전처리 설정에 따라 이미지 크기 조정, 정규화, 증강 적용
    3. 모델 정의 포함 (Sequential 또는 Model API)
    4. 컴파일 및 모델 학습 (fit) 포함
    5. 테스트셋 평가 포함
    6. 주석 없이 **오직 코드만** 반환할 것
    7. 함수 없이 파일 한 번에 실행되는 구조로 작성할 것 (ex. if __name__ == "__main__": 없이)
    8. 로드한 데이터셋에서 train,val,test로 데이터를 나누어 사용할 것
    
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

