from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float
from app.database import Base
from sqlalchemy.orm import relationship
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100))
    phone = Column(String(20))
    oauth_provider = Column(String(20), default="local")  # "local" 또는 "google"

    # 관계 설정
    models = relationship("Model", back_populates="user", cascade="all, delete")


class Model(Base):
    __tablename__ = "dashboards"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="models")

    # 모델 상세 기록과의 1:N 관계
    detail_records = relationship("ModelDetail", back_populates="model", cascade="all, delete")

# ModelDetail 테이블 (학습 상세 기록용)
class ModelDetail(Base):
    __tablename__ = "model_details"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("dashboards.id"), nullable=False)  # ✅ 모델과 연결
    epoch = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    loss = Column(Float)
    accuracy = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 역참조
    model = relationship("Model", back_populates="detail_records")