from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.gpt import generate_model_code
from app.utils.pubsub import subscribe_log
from app.task.train import run_training
from jose import jwt, JWTError
import os
import asyncio
import json

router = APIRouter()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")  # 기본값 추가

# JWT 검증 함수
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": payload["id"], "user_name": payload["name"]}
    except JWTError:
        raise ValueError("유효하지 않은 토큰입니다.")

@router.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결 완료")

    try:
        token = websocket.query_params.get("token")
        if not token:
            await websocket.send_json({"error": "Access token이 필요합니다."})
            await websocket.close()
            return

        user_info = verify_token(token)
        user_id = str(user_info["user_id"])
        print(f"인증된 유저 ID: {user_id}")

        # WebSocket으로 데이터 수신
        data = await websocket.receive_json()
        print(f"학습 요청 수신: {data}")

        # 직접 코드가 들어온 경우 (테스트용 PyTorch 등)
        if "code" in data:
            model_code = data["code"]
            form = data["form"]
            use_cloud = data.get("use_cloud", False)

        # GPT를 통해 코드 생성하는 경우
        else:
            model_code = generate_model_code(
                model_name=data["model_name"],
                layers=data["layers"],
                dataset=data["dataset"],
                preprocessing=data.get("preprocessing", {}),
                hyperparameters=data["hyperparameters"]
            )
            form = data["hyperparameters"]
            use_cloud = data.get("use_cloud", False)

        # Celery 태스크 실행
        run_training.delay(
            user_id,
            model_code,
            form["epochs"],
            form["batch_size"],
            form["learning_rate"],
            use_cloud
        )
        print("Celery 학습 태스크 실행됨")

        # Redis 로그 구독
        try:
            pubsub = subscribe_log(f"user:{user_id}")
            print(f"Redis 채널 구독: user:{user_id}")
        except Exception as e:
            print(f"Redis 구독 중 오류 발생: {e}")
        await websocket.send_json({"error": f"Redis 연결 실패: {str(e)}"})
        await websocket.close()
        return

        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
            if message:
                decoded = message["data"].decode("utf-8")
                await websocket.send_text(decoded)
                print(f"로그 전송: {decoded}")
            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"WebSocket 오류: {e}")
        await websocket.send_json({"error": str(e)})

    finally:
        await websocket.close()
        print("WebSocket 종료")