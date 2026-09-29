# 추천 실험 리포트

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
| 방법 | Recall@300 (≥0.30) | HitRate@50 (≥0.20) | NDCG@50 (≥0.08) | Coverage (≥0.20) | p95 (<200ms) |
|---|---|---|---|---|---|
| 인기도 기반 | | | | | |
| Two-Tower 단독 | | | | | |
| Multi-Stage | | | | | |

- Ranking AUC (목표 ≥ 0.70):
- Stage별 latency (candidate / ranking / reranking):
- Feature Store 조회 p95 (목표 < 10ms):

## 6. 오프라인 vs 온라인 지표 불일치 분석
