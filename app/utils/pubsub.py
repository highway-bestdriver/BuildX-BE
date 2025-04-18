# app/utils/pubsub.py
import redis
import os
import json

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
r = redis.Redis.from_url(REDIS_URL)

def publish_log(channel: str, data: dict):
    r.publish(channel, json.dumps(data))

def subscribe_log(channel: str):
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    return pubsub