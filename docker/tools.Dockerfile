# 데이터 생성, 인덱스 구축, 학습, 평가, 테스트용 (docker compose run --rm tools make train)
FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*
COPY requirements/ requirements/
RUN pip install --no-cache-dir --root-user-action=ignore --extra-index-url https://download.pytorch.org/whl/cpu -c requirements/constraints.txt -r requirements/dev.txt
COPY src/ src/
COPY configs/ configs/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY Makefile ./
