# app/task/train.py
from celery import shared_task
from app.utils.pubsub import publish_log
import subprocess
import os
import builtins

@shared_task(name="app.task.train.run_training")
def run_training(user_id, model_code, epochs, batch_size, learning_rate, use_cloud=False):
    print("Celery 학습 태스크 시작")
    print(f"user={user_id}, cloud={use_cloud}")

    try:
        if use_cloud:
            ...
            return {"message": "Cloud training started"}

        # print 오버라이드
        import builtins
        builtins.__print = builtins.print
        def custom_print(*args, **kwargs):
            text = " ".join(str(arg) for arg in args)
            publish_log(f"user:{user_id}", {"type": "log", "message": text})
            builtins.__print(*args, **kwargs)
        builtins.print = custom_print

        print("로컬 학습 실행 중")

        # model_code 실행
        exec_globals = {"__builtins__": builtins.__dict__}
        exec(model_code, exec_globals)

        print("모델 실행 완료")
        return {"message": "done"}

    except Exception as e:
        print(f"오류 발생: {e}")
        publish_log(f"user:{user_id}", {
            "type": "error",
            "message": str(e)
        })
        return {"error": str(e)}

    finally:
        builtins.print = builtins.__print