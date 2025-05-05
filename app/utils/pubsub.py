#app/utils/pubsub.py
import redis
import json
import os
from dotenv import load_dotenv
load_dotenv()

def get_redis_client():
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(REDIS_URL, decode_responses=False)

# 로그 publish 함수
def publish_log(channel: str, message: dict):
    redis_client = get_redis_client()
    redis_client.publish(channel, json.dumps(message))

# 로그 subscribe 함수 (WebSocket에서 사용)
def subscribe_log(channel: str):
    redis_client = get_redis_client()
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    return pubsub

def test_redis():
    try:
        client = get_redis_client()
        client.ping()
        print("Redis 연결 성공")
    except Exception as e:
        print("Redis 연결 실패:", e)