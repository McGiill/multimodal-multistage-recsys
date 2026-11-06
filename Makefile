.PHONY: install catalog data split train index search-eval-set eval-search eval-style-search eval eval-rec bench up down test lint docker-train docker-eval docker-test

install:        ## 로컬 개발 환경 설치
	pip install -r requirements/dev.txt

catalog:        ## H&M 원본 → 팀 공통 상품 카탈로그 (data/processed/products.parquet)
	python -m src.data.build_catalog

data:           ## 시뮬레이터로 상품/유저/로그 생성
	python -m src.simulator.run

split:          ## train/valid/test = 8/1/1 (시간 기반)
	python -m src.evaluation.split

index:          ## CLIP 임베딩 + FAISS 인덱스 구축
	python -m src.search.indexer

search-eval-set: ## 검색 평가셋(쿼리 → 정답 상품) 생성
	python -m src.evaluation.build_search_eval

eval-search:    ## 검색 평가 (CLIP vs BM25)
	python -m src.evaluation.eval_search

eval-style-search: ## 스타일 사진 검색 평가 (상품 사진 쿼리에서 기존 이미지 검색과 비교)
	python -m src.evaluation.eval_style_search

train:          ## Two-Tower → FAISS → DeepFM CVR → 세션 인코더 학습
	python -m src.recommendation.candidate.train
	python -m src.recommendation.candidate.build_index
	python -m src.recommendation.ranking.train
	python -m src.recommendation.session.train

eval:           ## 오프라인 지표 산출
	python -m src.evaluation.eval_search
	python -m src.evaluation.eval_recommend

bench:          ## 응답시간 p95 측정 (warm-up 10 + 100회)
	python -m src.evaluation.latency_bench --target search

up:
	docker compose up --build

down:
	docker compose down

docker-train:   ## Docker에서 추천 모델 학습 (결과: artifacts/models)
	docker compose run --rm --build tools make train

docker-eval:    ## Docker에서 오프라인 평가
	docker compose run --rm --build tools make eval

docker-test:    ## Docker(Python 3.10)에서 전체 테스트
	docker compose run --rm --build tools make test

test:
	pytest -q

lint:
	ruff check src tests

eval-rec:
	python -m src.evaluation.eval_recommend
