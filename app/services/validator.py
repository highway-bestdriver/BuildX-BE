import re
from app.schemas.generate import ModelRequest
from typing import Dict

def parse_forward_connections(code: str):
    #forward 함수 내에서 self.XXX(x) 형태로 실행된 레이어 추출
    pattern = re.compile(r"x\s*=\s*self\.(\w+)\(x\)")
    found = pattern.findall(code)
    print("[DEBUG] 코드 내 forward() 레이어 실행 순서:", found)
    return found

def trace_json_connections(layers, start_id="input"):
    #ModelRequest 기반으로 연결된 순서대로 레이어 ID 추적. DFS를 통해 input → output 흐름 추적
    connections = []
    visited = set()

    def dfs(current_id):
        for layer in layers:
            inputs = layer.input if isinstance(layer.input, list) else [layer.input]
            if current_id in inputs and layer.id not in visited:
                visited.add(layer.id)
                connections.append(layer.id)
                print(f"[DEBUG] 연결됨: {current_id} → {layer.id}")
                dfs(layer.id)

    dfs(start_id)
    print("[DEBUG] JSON 기반 레이어 연결 순서:", connections)
    return connections

def parse_hyperparameters_from_code(code: str):
    import re

    variables = parse_variable_definitions(code)
    results = {}

    match_epoch = re.search(r"range\((\w+)\)", code)
    if match_epoch:
        name = match_epoch.group(1)
        if name in variables:
            results["epochs"] = variables[name]

    match_bs = re.search(r"batch_size\s*=\s*(\w+)", code)
    if match_bs:
        name = match_bs.group(1)
        if name in variables:
            results["batch_size"] = variables[name]

    match_lr = re.search(r"lr\s*=\s*(\w+)", code)
    if match_lr:
        name = match_lr.group(1)
        if name in variables:
            results["learning_rate"] = variables[name]

    return results

def validate_hyperparameters(expected: Dict, actual: Dict):
    mismatches = []
    for key in expected:
        if key in actual and expected[key] != actual[key]:
            mismatches.append({
                "parameter": key,
                "expected": expected[key],
                "actual": actual[key]
            })
    return mismatches

def parse_variable_definitions(code: str) -> dict:
    import re
    pattern = re.compile(r"(\w+)\s*=\s*([0-9.]+)")
    return {name: float(val) if '.' in val else int(val) for name, val in pattern.findall(code)}


def validate_code(model_request: ModelRequest, generated_code: str):
    """
       전체 검증 함수: JSON 연결 흐름 vs 코드 실행 흐름 비교
    """
    print("\n=========== 모델 코드 검증 시작 ===========\n")
    # 1. 실행 순서 검증
    json_layers = model_request.layers
    expected_flow = trace_json_connections(json_layers, start_id="input")
    code_flow = parse_forward_connections(generated_code)

    # 2. 하이퍼파라미터 검증
    expected_hyper = model_request.hyperparameters.dict()
    actual_hyper = parse_hyperparameters_from_code(generated_code)
    hyper_mismatches = validate_hyperparameters(expected_hyper, actual_hyper)

    result = {"valid": True}

    if expected_flow != code_flow:
        print("[ERROR] 레이어 실행 순서 불일치")
        print("         기대 순서:", expected_flow)
        print("         코드 순서:", code_flow)
        result["valid"] = False
        result["reason"] = "실행 순서 불일치"
        result["expected"] = expected_flow
        result["actual"] = code_flow

    if hyper_mismatches:
        print("[ERROR] 하이퍼파라미터 불일치")
        print("         기대 하이퍼파라미터:", expected_hyper)
        print("         코드 하이퍼파라미터:", actual_hyper)
        result["valid"] = False
        result["reason"] = "실행 순서 or 하이퍼파라미터 불일치"
        result["hyperparameter_errors"] = hyper_mismatches

    print("[SUCCESS] 일치")
    return result
