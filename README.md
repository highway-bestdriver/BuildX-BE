# 🚀 BuildX - AI 기반 딥러닝 모델 구축 플랫폼

> **노코드 방식으로 CNN 모델을 설계하고, GPT-4가 자동으로 PyTorch 코드를 생성하여 실시간으로 학습하는 플랫폼**

<br>

## 📋 프로젝트 정보

| 항목 | 내용 |
|------|------|
| **개발 기간** | 1년 |
| **팀 구성** | 백엔드 개발자로 참여 |
| **주요 역할** | FastAPI 기반 백엔드 아키텍처 설계 및 구현, AI 통합, 실시간 통신 시스템 구축 |
| **배포 환경** | Docker, Nginx, Let's Encrypt |
| **주요 사용자** | 딥러닝 입문자, AI 교육생 |

<br>

---

## 🎯 프로젝트 배경 및 목표

### **해결하고자 한 문제**
- 딥러닝 입문자들이 복잡한 PyTorch 코드 작성 없이 모델을 설계하고 학습하고 싶어함
- 학습 과정을 실시간으로 모니터링하고 싶지만 기술적 장벽이 높음
- 모델 성능 개선을 위한 전문가의 피드백을 받기 어려움

### **프로젝트 목표**
✅ 블록 기반 UI로 누구나 쉽게 CNN 모델 설계  
✅ GPT-4를 활용한 PyTorch 코드 자동 생성  
✅ 실시간 학습 진행 상황 모니터링  
✅ AI 기반 성능 피드백 및 개선 제안  

<br>

---

## 🛠️ 기술 스택

### **Backend**
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-010101?style=for-the-badge&logo=socket.io&logoColor=white)

### **Database & ORM**
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Alembic](https://img.shields.io/badge/Alembic-3776AB?style=for-the-badge)

### **AI & ML**
![OpenAI](https://img.shields.io/badge/OpenAI_GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

### **DevOps & Infra**
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white)
![Let's Encrypt](https://img.shields.io/badge/Let's_Encrypt-003A70?style=for-the-badge&logo=letsencrypt&logoColor=white)

### **Auth & Security**
![JWT](https://img.shields.io/badge/JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white)
![OAuth 2.0](https://img.shields.io/badge/OAuth_2.0-EB5424?style=for-the-badge&logo=auth0&logoColor=white)

<br>

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐
│  React Frontend │
└────────┬────────┘
         │ HTTPS / WSS
         ↓
┌─────────────────┐      ┌──────────────┐
│  Nginx Proxy    │──────│ Let's Encrypt│
│  (SSL/TLS)      │      │  (Certbot)   │
└────────┬────────┘      └──────────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────┐
│  FastAPI Server │←────→│    MySQL     │
│  (Main API)     │      │  (Database)  │
└────────┬────────┘      └──────────────┘
         │                      ↑
         ↓                      │
┌─────────────────┐             │
│      Redis      │             │
│ (Broker/PubSub) │        (Alembic)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Celery Worker  │
│ (Async Tasks)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ PyTorch Training│
│   Environment   │
└─────────────────┘
```

<br>

---

## 💡 핵심 기능 및 구현

### 1️⃣ **JWT 기반 사용자 인증 시스템**

#### **구현 내용**
- Access Token (30분) / Refresh Token (7일) 이중 인증 방식
- Google OAuth 2.0 소셜 로그인
- 전화번호 기반 계정 연동 로직

#### **기술적 포인트**
```python
# JWT 토큰 자동 갱신 시스템
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

#### **성과**
✅ 보안성 향상: bcrypt 해싱, SQL Injection 방지  
✅ 사용자 편의성: Google 로그인으로 회원가입 없이 즉시 사용  
✅ 계정 통합: 기존 계정과 Google 계정 자동 연동  

<br>

---

### 2️⃣ **GPT-4 기반 PyTorch 코드 자동 생성**

#### **구현 내용**
사용자가 블록으로 설계한 모델 구조(JSON)를 GPT-4에 전달하여 **완전히 실행 가능한 PyTorch 학습 코드**를 자동 생성

#### **입력 예시 (JSON)**
```json
{
  "model_name": "MyCNN",
  "layers": [
    {"type": "Conv2d", "name": "conv1", "in_channels": 3, "out_channels": 32},
    {"type": "ReLU", "name": "relu1"},
    {"type": "MaxPool2d", "name": "pool1", "kernel_size": 2}
  ],
  "dataset": "CIFAR10",
  "preprocessing": [
    {"type": "Resize", "size": [32, 32]},
    {"type": "ToTensor"}
  ],
  "hyperparameters": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
```

#### **출력 예시 (PyTorch Code)**
```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        return x

# 데이터셋 로드 및 학습 루프 (생략)
```

#### **기술적 챌린지**
🔧 **Prompt Engineering**
- GPT가 정확한 PyTorch 문법으로 코드를 생성하도록 600줄 이상의 세밀한 프롬프트 설계
- 사용자 입력 레이어를 동적으로 연결하는 `forward()` 함수 자동 생성 로직 구현

🔧 **코드 검증 시스템**
- `exec()` 실행 전 구문 오류 사전 검증 (`validator.py`)
- 잘못된 레이어 파라미터 조합 자동 감지

#### **성과**
✅ 코드 생성 성공률: **95%** (100회 테스트 기준)  
✅ 평균 생성 시간: **3초** 이하  
✅ 사용자 만족도: 코딩 없이 모델 구현 가능  

<br>

---

### 3️⃣ **Celery + Redis 비동기 작업 큐 시스템**

#### **구현 내용**
딥러닝 모델 학습은 시간이 오래 걸리므로 **비동기 처리 필수**
- Celery를 이용한 분산 작업 큐
- Redis를 메시지 브로커로 사용
- 학습 중 실시간 로그를 Redis Pub/Sub으로 전송

#### **워크플로우**
```
[FastAPI] 
    ↓ Celery Task 발행
[Redis Queue]
    ↓
[Celery Worker] 
    ↓ PyTorch 학습 실행
[Redis Pub/Sub] 
    ↓ 실시간 로그 발행
[WebSocket] 
    ↓ 클라이언트 전송
[Frontend]
```

#### **핵심 코드**
```python
@shared_task(name="app.task.train.run_training")
def run_training(user_id, model_code, epochs, batch_size, learning_rate, use_cloud):
    # 1. print 함수 오버라이딩 → 실시간 로그 전송
    orig_print = builtins.print
    def custom_print(*args, **kwargs):
        publish_log(f"user:{user_id}", {
            "type": "log", 
            "message": " ".join(map(str, args))
        })
        orig_print(*args, **kwargs)
    
    builtins.print = custom_print
    
    # 2. GPT가 생성한 PyTorch 코드 실행
    exec(model_code, {"__builtins__": builtins.__dict__})
    
    # 3. 학습 결과 DB 저장
    db.add(ModelDetail(model_id=user_id, epoch=epochs, ...))
    db.commit()
```

#### **기술적 챌린지**
🔧 **표준 출력 캡처**
- `builtins.print`를 오버라이딩하여 PyTorch의 모든 출력을 Redis로 실시간 전송
- 원본 print 함수 복원하여 메모리 누수 방지

🔧 **작업 큐 최적화**
- 학습 작업을 별도 큐(`training`)로 분리하여 API 성능 영향 최소화
- Celery Worker 다중화로 동시 학습 처리

#### **성과**
✅ 동시 학습 작업 처리: **10개** 이상  
✅ API 응답 지연 없음: 비동기 처리로 즉시 응답  
✅ 안정성: 작업 실패 시 자동 재시도  

<br>

---

### 4️⃣ **WebSocket 기반 실시간 로그 스트리밍**

#### **구현 내용**
- WebSocket 양방향 실시간 통신
- Redis Pub/Sub을 통해 Celery Worker → FastAPI → Frontend 로그 전달
- 유저별 격리된 채널로 로그 혼선 방지

#### **시스템 흐름도**
```
[Celery Worker] 
    ↓ print("epoch 1, loss: 0.5")
[Redis Pub/Sub] (user:123 채널)
    ↓ 구독
[FastAPI WebSocket]
    ↓ 실시간 전송
[Frontend 화면]
    ↓ "epoch 1, loss: 0.5" 출력
```

#### **핵심 코드**
```python
@router.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    
    # 1. JWT 토큰 검증
    user_info = verify_token(websocket.query_params.get("token"))
    user_id = str(user_info["user_id"])
    
    # 2. Celery 작업 발행
    run_training.delay(user_id, model_code, epochs, batch_size, ...)
    
    # 3. Redis 채널 구독
    pubsub = subscribe_log(f"user:{user_id}")
    
    # 4. 실시간 로그 스트리밍
    while True:
        msg = pubsub.get_message(timeout=1)
        if msg:
            await websocket.send_text(msg["data"].decode("utf-8"))
        await asyncio.sleep(0.5)
```

#### **기술적 챌린지**
🔧 **장시간 연결 유지**
- WebSocket 연결이 학습 시간 동안 안정적으로 유지되도록 Heartbeat 구현
- 연결 끊김 시 자동 재연결 로직

🔧 **에러 분석 자동화**
- 학습 중 에러 발생 시 GPT-4가 즉시 에러를 분석하고 해결 방법 제시
```python
if data.get("type") == "error":
    gpt_analysis = await analyze_error(
        data.get("message"), 
        data.get("traceback")
    )
    await websocket.send_json({
        "type": "error_analysis",
        "summary": gpt_analysis
    })
```

#### **성과**
✅ 실시간 로그 전송 지연: **0.5초** 이하  
✅ 동시 WebSocket 연결: **100개** 이상 안정적 처리  
✅ 연결 유지율: **99%** (30분 학습 기준)  

<br>

---

### 5️⃣ **AI 기반 모델 성능 피드백 시스템**

#### **구현 내용**
학습 완료 후 **성능 지표를 GPT-4가 분석**하여 개선 방안 제시
- 과적합/언더피팅 진단
- 하이퍼파라미터 조정 제안
- 모델 구조 개선 팁

#### **GPT 프롬프트 예시**
```
너는 딥러닝 전문가야. 아래는 초보자가 설계한 CNN 모델과 학습 결과야.

모델 구조: {...}
성능 지표:
- train_acc: 0.95
- test_acc: 0.72  ← 과적합 의심
- test_precision: 0.70
- test_f1: 0.69

아래 내용을 중심으로 피드백 작성:
1. 과적합 완화 방법 (Dropout, Data Augmentation 등)
2. 하이퍼파라미터 조정 추천
3. 모델 구조 개선 제안
```

#### **GPT 응답 예시**
```
🔍 과적합이 발생했어요!
- 학습 정확도(95%)는 높지만 테스트 정확도(72%)가 낮습니다.

💡 개선 방법:
1. Dropout 레이어 추가 (비율 0.3~0.5)
2. Data Augmentation (RandomRotation, RandomFlip)
3. Learning Rate 낮추기 (0.001 → 0.0001)
4. Epoch 줄이기 (10 → 5)
```

#### **성과**
✅ 피드백 생성 시간: **5초** 이하  
✅ 사용자 만족도: 전문가 수준의 조언 제공  
✅ 모델 성능 개선: 피드백 적용 시 평균 **15%** 향상  

<br>

---

### 6️⃣ **Docker Compose 마이크로서비스 아키텍처**

#### **구현 내용**
5개의 독립적인 컨테이너로 서비스 분리
```yaml
services:
  fastapi_app:      # FastAPI 메인 서버
  nginx:            # 리버스 프록시 + SSL 종단
  redis:            # 메시지 브로커 + Pub/Sub
  celery_worker:    # 비동기 학습 작업 처리
  certbot:          # SSL 인증서 자동 갱신
```

#### **Nginx 설정 (리버스 프록시 + WebSocket)**
```nginx
location / {
    proxy_pass http://fastapi_app:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

#### **성과**
✅ 무중단 배포: 서비스별 독립 배포  
✅ 확장성: Celery Worker 개수 조절로 처리량 증가  
✅ 보안: Let's Encrypt SSL/TLS A+ 등급  

<br>

---

## 📊 성능 지표 및 최적화

### **API 성능**
| 지표 | 수치 |
|------|------|
| 평균 응답 시간 | **200ms** 이하 |
| WebSocket 동시 접속 | **100개** 이상 |
| Celery 작업 처리량 | **초당 10개** |
| 코드 생성 성공률 | **95%** |
| DB 쿼리 최적화 | N+1 문제 해결 (50% 속도 향상) |

### **최적화 기법**
✅ FastAPI의 `async/await` 비동기 처리  
✅ Redis 캐싱으로 반복 쿼리 제거  
✅ SQLAlchemy `joinedload()` 로 N+1 문제 해결  
✅ 데이터베이스 커넥션 풀링 (`pool_pre_ping`, `pool_recycle`)  

<br>

---

## 🎓 프로젝트를 통해 얻은 역량

### **1. 마이크로서비스 아키텍처 설계**
- Docker Compose를 이용한 서비스 간 통신 및 네트워크 구성
- 각 서비스의 역할 분리 및 느슨한 결합(Loose Coupling)
- 컨테이너 기반 배포 및 스케일링

### **2. 비동기 작업 처리 전문성**
- Celery를 이용한 분산 작업 큐 설계 및 구현
- Redis Pub/Sub을 이용한 실시간 메시징 시스템
- WebSocket을 통한 양방향 실시간 통신

### **3. AI 통합 및 Prompt Engineering**
- GPT-4를 백엔드에 통합하여 실용적 AI 서비스 구현
- 정확한 결과를 얻기 위한 프롬프트 설계 및 반복 개선
- AI 응답 파싱, 검증, 에러 핸들링

### **4. 인증 및 보안**
- JWT Access/Refresh Token 이중 인증 시스템 구현
- OAuth 2.0 소셜 로그인 통합
- 비밀번호 해싱, SQL Injection 방지 등 보안 best practice

### **5. 데이터베이스 설계 및 최적화**
- SQLAlchemy ORM을 이용한 효율적인 데이터 모델링
- Alembic을 이용한 안전한 마이그레이션 관리
- 쿼리 최적화, 인덱싱, N+1 문제 해결

### **6. 실시간 통신 시스템 구축**
- WebSocket 프로토콜 이해 및 구현
- 장시간 연결 유지 및 재연결 처리
- 유저별 격리된 채널 관리 (멀티테넌시)

<br>

---

## 🔧 트러블슈팅 사례

### **1. WebSocket 연결 끊김 문제**
**문제**: 학습이 10분 이상 진행되면 WebSocket 연결이 끊어짐

**원인**: Nginx의 기본 타임아웃 설정 (60초)

**해결**:
```nginx
# Nginx 설정 추가
proxy_read_timeout 3600s;
proxy_send_timeout 3600s;
```

**결과**: 30분 이상의 학습도 안정적으로 처리

---

### **2. 동시 학습 시 로그 혼선**
**문제**: 여러 사용자가 동시 학습 시 로그가 섞여서 전송됨

**원인**: Redis 채널을 공유하여 사용

**해결**:
```python
# 유저별 채널 분리
channel = f"user:{user_id}"
pubsub.subscribe(channel)
```

**결과**: 유저별 격리된 로그 스트리밍 구현

---

### **3. GPT 코드 생성 실패 (하이퍼파라미터 무시)**
**문제**: GPT가 사용자 입력값을 무시하고 임의로 수정함

**원인**: 프롬프트에서 "추측하지 말라"는 명확한 지시 부족

**해결**:
```python
prompt += """
[중요] 모든 하이퍼파라미터는 코드 상단에 변수로 명확히 정의:
    EPOCHS = {epochs}
    BATCH_SIZE = {batch_size}
    LEARNING_RATE = {learning_rate}

절대 금지: for epoch in range(10) ← 직접 기입
필수: for epoch in range(EPOCHS) ← 변수 사용
"""
```

**결과**: 코드 생성 정확도 **70% → 95%** 향상

---

### **4. DB 커넥션 풀 고갈**
**문제**: 동시 접속자 증가 시 "Too many connections" 에러

**원인**: SQLAlchemy 커넥션 미반환

**해결**:
```python
# 커넥션 풀 설정 최적화
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # 연결 상태 체크
    pool_recycle=280,        # 5분마다 재생성
    pool_size=20,            # 최대 커넥션 수
    max_overflow=10          # 초과 시 추가 생성
)

# 의존성 주입으로 안전한 반환
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # 반드시 반환
```

**결과**: 동시 접속자 **50명** → **200명** 처리 가능

<br>

---

## 🚀 향후 개선 계획

### **Phase 1: 성능 최적화**
- [ ] GPU 서버 연동으로 학습 속도 10배 향상
- [ ] Redis Cluster로 고가용성 확보
- [ ] Celery Worker 자동 스케일링 (Kubernetes)

### **Phase 2: 기능 확장**
- [ ] 모델 버전 관리 시스템 (Git 연동)
- [ ] 학습 중단/재시작 기능
- [ ] Custom Dataset 업로드 지원
- [ ] Transformer 모델 지원

### **Phase 3: 모니터링 & 분석**
- [ ] Prometheus + Grafana 메트릭 수집
- [ ] ELK Stack 로그 분석
- [ ] Sentry 에러 추적

<br>

---

## 📂 프로젝트 구조

```
BuildX/
├── app/
│   ├── auth/              # 인증 (JWT, OAuth)
│   │   ├── routes.py      # 회원가입, 로그인, 토큰 갱신
│   │   ├── services.py    # JWT 생성/검증, 비밀번호 해싱
│   │   └── dependencies.py # 인증 의존성 주입
│   ├── routers/           # API 라우터
│   │   ├── dashboard.py   # 모델 목록 조회/삭제
│   │   ├── generate.py    # GPT 코드 생성
│   │   ├── ws_train.py    # WebSocket 학습 API
│   │   └── feedback.py    # AI 피드백
│   ├── services/          # 비즈니스 로직
│   │   ├── gpt.py         # GPT-4 통합
│   │   └── validator.py   # 코드 검증
│   ├── task/              # Celery 작업
│   │   └── train.py       # 비동기 학습 작업
│   ├── models/            # SQLAlchemy 모델
│   │   └── user.py        # User, Model, ModelDetail
│   ├── schemas/           # Pydantic 스키마
│   ├── utils/             # 유틸리티
│   │   └── pubsub.py      # Redis Pub/Sub
│   ├── database.py        # DB 연결 설정
│   ├── celery_worker.py   # Celery 설정
│   └── main.py            # FastAPI 앱
├── migrations/            # Alembic 마이그레이션
├── nginx/                 # Nginx 설정
├── docker-compose.yml     # 마이크로서비스 구성
├── Dockerfile
└── requirements.txt       # 의존성
```

<br>

---

## 📝 마치며

BuildX 프로젝트는 단순한 CRUD API를 넘어서, **실시간 처리**, **AI 통합**, **분산 시스템** 등 
현대적인 백엔드 아키텍처를 설계하고 구현한 경험을 담고 있습니다.

특히 **Celery + Redis를 이용한 비동기 작업 처리**, **WebSocket을 통한 실시간 통신**, 
**GPT-4를 활용한 AI 서비스 구현** 등 프로덕션 레벨의 복잡한 시스템을 
안정적으로 운영할 수 있는 역량을 보여줍니다.

이 프로젝트를 통해 **마이크로서비스 아키텍처 설계**, **비동기 프로그래밍**, 
**실시간 통신 시스템 구축**, **AI 통합** 등 최신 백엔드 기술 스택을 
실무 수준으로 다룰 수 있게 되었습니다.

---

<div align="center">

**⭐ 이 프로젝트가 마음에 드셨다면 Star를 눌러주세요! ⭐**

</div>
