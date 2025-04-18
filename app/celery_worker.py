# app/celery_worker.py
from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.task_routes = {
    "app.task.train.run_training": {"queue": "training"},
}

# 반드시 이 import가 있어야 task 등록
import app.task.train