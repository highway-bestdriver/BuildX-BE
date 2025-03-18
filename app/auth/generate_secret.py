import secrets

# 안전한 랜덤 SECRET_KEY 생성
secret_key = secrets.token_urlsafe(32)  # 32바이트 길이의 랜덤 문자열 생성
print(secret_key)