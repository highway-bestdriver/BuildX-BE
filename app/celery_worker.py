# app/celery_worker.py
from celery import Celery
import os
from dotenv import load_dotenv
load_dotenv()

print("REDIS_URL:", os.getenv("REDIS_URL"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.task_routes = {
    "app.task.train.run_training": {"queue": "training"},
}

import app.task.train