from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.user import User, Model
from app.auth.dependencies import get_current_user
from app.schemas.dashboard import ModelResponse

router = APIRouter()

# 모델 목록 가져오기 (GET /dashboard)
@router.get("", response_model=List[ModelResponse])
def get_models(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    models = db.query(Model).filter(Model.user_id == current_user.id).all()
    if not models:
        raise HTTPException(status_code=404, detail="등록된 모델이 없습니다.")

    return [
        ModelResponse(
            id=model.id,
            name=model.name,
            last_modified=model.updated_at.strftime("%Y.%m.%d")
        )
        for model in models
    ]

# 모델 삭제 (DELETE /dashboard/{id})
@router.delete("/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    model = db.query(Model).filter(Model.id == model_id, Model.user_id == current_user.id).first()
    if not model:
        raise HTTPException(status_code=404, detail="해당 모델을 찾을 수 없습니다.")
    
    db.delete(model)
    db.commit()
    return {"message": "모델이 삭제되었습니다."}