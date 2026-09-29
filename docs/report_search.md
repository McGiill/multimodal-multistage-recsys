# 검색 실험 리포트

## 1. 데이터 분할 방식
- train / valid / test = 8 / 1 / 1 (시간 기반, 최근 데이터 → test)

## 2. 지표 정의 / 계산식
-

## 3. 재현용 시드 / 설정값
- random_seed = 42
- 설정 파일:

## 4. 실측 환경
- CPU / RAM / GPU:
- 측정 방법: warm-up 10회 후 100회 호출 p95, 단건 요청

## 5. 결과 (베이스라인 대비 개선율)
| 방법 | MRR (목표 ≥ 0.55) | NDCG@10 (목표 ≥ 0.50) | p95 latency (목표 < 200ms) |
|---|---|---|---|
| BM25 (베이스라인) | | | |
| CLIP text | | | |
| CLIP image | | | |
| CLIP hybrid | | | |

## 6. FAISS 인덱스 Trade-off (nprobe / HNSW 파라미터별 속도-Recall)
