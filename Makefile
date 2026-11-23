.PHONY: install catalog data split train train-final-rank refit index search-eval-set eval-search eval-style-search index-tradeoff eval eval-rec eval-final eval-events experiment-images abtest stream monitor retrain registry bench bench-rec bench-feature up up-ct down test test-shop lint docker-train docker-eval docker-test

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

index-tradeoff: ## FAISS 인덱스 비교 (flat / HNSW / IVF / IVF+PQ의 속도·Recall·크기·검색 품질)
	python -m src.evaluation.index_tradeoff

train:          ## Two-Tower → FAISS → DeepFM → 세션 인코더 → 최종 순위 모델 → train+valid 재학습(서빙 묶음)
	python -m src.recommendation.candidate.train
	python -m src.recommendation.candidate.build_index
	python -m src.recommendation.ranking.train
	python -m src.recommendation.session.train
	python -m src.recommendation.final_rank.train
	python -m src.recommendation.refit

train-final-rank: ## 최종 순위 모델만 다시 학습 (선택용 묶음, valid 정답)
	python -m src.recommendation.final_rank.train $(ARGS)

refit:          ## 서빙 묶음만 다시 만들기 (train + valid 앞 7일로 Stage 1·2 재학습)
	python -m src.recommendation.refit

eval:           ## 오프라인 지표 산출 (검색, 추천 단계별, 최종 추천: 사용자 단위 주 지표)
	python -m src.evaluation.eval_search
	python -m src.evaluation.eval_recommend
	python -m src.evaluation.eval_pipeline

eval-final:     ## 사용자 단위 최종 추천 평가 (명세 판정, Redis와 분리, 새 시각별 결과 파일)
	python -m src.evaluation.eval_pipeline $(ARGS)

eval-events:    ## 보조 지표: 조회 1건 단위 조회·구매 평가 (세션 클릭 효과 확인)
	python -m src.evaluation.eval_pipeline --target both $(ARGS)

experiment-images: ## 동일 데이터·seed의 Two-Tower 이미지 특징 유무 비교 (별도 버전)
	python -m src.evaluation.experiment_recommend $(ARGS)

abtest:         ## 두 전략의 사용자 단위 A/B 시뮬레이션
	python -m src.abtest.simulate $(ARGS)

stream:         ## 실행 중인 API에 가상 고객 노출·행동 전송
	python -m src.simulator.stream $(ARGS)

monitor:        ## 영속 로그 CTR·추천 HitRate 임계치 점검
	python -m src.ct.monitor --once

retrain:        ## 새 로그 수·관찰창 확인 후 별도 모델 학습·검증 (자동 활성화 안 함)
	python -m src.ct.trigger --once $(ARGS)

registry:       ## 모델 버전·검증·활성 상태 조회
	python -m src.ct.registry list

bench:          ## 응답시간 p95 측정 (warm-up 10 + 100회)
	python -m src.evaluation.latency_bench --target search

bench-rec:      ## 추천 API 파싱·Redis 조회·직렬화·응답 등록 포함 p95
	python -m src.evaluation.latency_bench --target recommend

bench-feature:
	python -m src.evaluation.latency_bench --target feature

up:
	docker compose up --build

up-ct:
	docker compose --profile ct up --build

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

test-shop:
	node --test tests/shop_exposure.test.cjs

lint:
	ruff check src tests

eval-rec:
	python -m src.evaluation.eval_recommend
