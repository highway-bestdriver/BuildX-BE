# 1. Python 기반 컨테이너 사용
FROM python:3.10

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 앱 파일 복사
COPY . .

# 5. 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
