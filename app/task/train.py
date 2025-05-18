# app/task/train.py
from celery import shared_task
from app.utils.pubsub import publish_log
from app.database import SessionLocal
from app.models.user import ModelDetail
from datetime import datetime
import builtins
import traceback

@shared_task(name="app.task.train.run_training")
def run_training(user_id, model_code, epochs, batch_size, learning_rate, use_cloud):
    orig_print = builtins.print
    def custom_print(*args, **kwargs):
        try:
            publish_log(f"user:{user_id}", {"type": "log", "message": " ".join(map(str,args))})
        except Exception as e:
            orig_print(f"[publish_log 오류] {e}")
        orig_print(*args, **kwargs)
    builtins.print = custom_print
    try:
        print("[Celery] >>> exec(model_code) 시작")
        # 학습 결과를 저장할 변수들
        final_loss = None
        final_accuracy = None
        
        # 모델 코드 실행 (model_id를 전역 변수로 전달)
        exec(model_code, {"__builtins__": builtins.__dict__, "model_id": user_id})
        print("모델 실행 완료")
        
        # 학습 결과 DB에 저장
        db = SessionLocal()
        try:
            model_detail = ModelDetail(
                model_id=user_id,  # user_id를 model_id로 사용
                epoch=epochs,  # 총 epoch 수
                batch_size=batch_size,
                learning_rate=learning_rate,
                loss=final_loss,
                accuracy=final_accuracy,
                created_at=datetime.now()
            )
            db.add(model_detail)
            db.commit()
            print("학습 결과 DB 저장 완료")
        except Exception as e:
            print(f"DB 저장 중 오류: {e}")
            db.rollback()
        finally:
            db.close()
            
        return {"message": "done"}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"오류 발생: {e}")
        publish_log(f"user:{user_id}", {"type":"error", "message":str(e), "traceback": tb})
        print("Redis에 publish 완료")

    finally:
        builtins.print = orig_print