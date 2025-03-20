import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv  # ✅ 환경 변수 로드

# .env 파일에서 환경 변수 로드
load_dotenv()

# Alembic 설정 파일
config = context.config

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 데이터베이스 URL을 .env에서 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL 환경 변수가 설정되지 않았습니다!")

# Alembic이 사용할 DB URL 설정
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# ✅ 모델을 직접 import 해야 Alembic이 감지할 수 있음
from app.database import Base  # Base 모델 가져오기
from app.models import user  # 🚀 User 모델 직접 import

target_metadata = Base.metadata  # ✅ Alembic이 Base.metadata를 인식하도록 설정

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()