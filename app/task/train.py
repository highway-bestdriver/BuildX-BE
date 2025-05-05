# app/task/train.py
from celery import shared_task
from app.utils.pubsub import publish_log
import builtins

@shared_task(name="app.task.train.run_training")
def run_training(user_id, model_code, *_, **__):
    orig_print = builtins.print
    def custom_print(*args, **kwargs):
        try:
            publish_log(f"user:{user_id}", {"type": "log", "message": " ".join(map(str,args))})
        except Exception as e:
            orig_print(f"[publish_log 오류] {e}")
        orig_print(*args, **kwargs)
    builtins.print = custom_print
    try:
        exec(model_code, {"__builtins__": builtins.__dict__})
        print("모델 실행 완료")
        return {"message": "done"}
    except Exception as e:
        print(f"오류 발생: {e}")
        publish_log(f"user:{user_id}", {"type":"error", "message":str(e)})
    finally:
        builtins.print = orig_print