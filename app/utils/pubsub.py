# app/utils/pubsub.py
import redis
import json
import os
from dotenv import load_dotenv
load_dotenv()
# Redis 연결 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)

# 로그 publish 함수
def publish_log(channel: str, message: dict):
    redis_client.publish(channel, json.dumps(message))

# 로그 subscribe 함수 (WebSocket에서 사용)
def subscribe_log(channel: str):
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    return pubsub
