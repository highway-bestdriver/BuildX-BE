#app/utils/pubsub.py
import os, json, redis, builtins
from dotenv import load_dotenv; load_dotenv()

def get_redis_client():
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                                decode_responses=False)

def publish_log(channel: str, message: dict):
    try:
        redis_client = get_redis_client()
        redis_client.publish(channel, json.dumps(message, ensure_ascii=False))
    except Exception as e:
        builtins.print(f"[Redis Publish 오류] {e}")

def subscribe_log(channel):
    client = get_redis_client()
    p = client.pubsub(); p.subscribe(channel)
    return p