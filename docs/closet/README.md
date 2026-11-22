# 옷장·콜드스타트 코드 정리

2026-11-20. 최신 옷장 기능을 팀 저장소의 패키지 구조에 배치했다.
기존 `closet-capstone/` 폴더를 대체하며, 데이터·모델·사용자 DB는 포함하지 않는다.

## 위치

| 내용 | 위치 |
|---|---|
| 실행기·코디 검색·데이터 경로 | `src/closet/` |
| 옷장 API·저장·취향 처리 | `src/closet/shop/` |
| 옷장·콜드스타트 화면 | `src/serving/shop/closet/` |
| 기존 검색 확인 화면 | `src/serving/shop/closet_retrieval/` |
| 추천 벡터 연결 어댑터 | `src/recommendation/preferences/adapter.py` |
| 포트·데이터·기록 경로 설정 | `configs/closet_demo.json` |
| 단위·실제 데이터 테스트 | `tests/closet/` |
| 인계 자료 | `docs/closet/` |

## 실행

최신 `closet-demo-60.zip`을 풀고 프로젝트 루트에서 실행한다.
아래 예시는 프로젝트와 데이터 폴더를 나란히 둔 경우다.

```bash
python -m pip install -r requirements/closet.txt
python -m src.closet.run --data-root "../closet-demo-data-60"
```

- 화면: http://127.0.0.1:8767/shop/#/mypage
- `configs/closet_demo.json`의 기본 데이터 경로는 프로젝트 내부 `data/closet-demo-data-60`이다. 그 위치에 풀었다면 `python -m src.closet.run`만 실행해도 된다.
- 포트가 겹치면 `--retrieval-port 8995 --shop-port 8997`을 추가한다.
- 기존 기록을 이어 쓰려면 `--state-dir "기존 profiles.sqlite3가 있는 폴더"`를 지정한다. 지정하지 않으면 `.local/closet`에서 새로 시작한다. 이전 DB를 자동 이동하거나 삭제하지 않는다.
- 종료는 Ctrl+C. 팀 API 주소는 `--team-url` 또는 설정 파일로 지정한다.
- 외부 데이터 내부의 `local_closet_demo/`는 기존 데이터 형식이므로 이름을 바꾸지 않는다.

## 확인

```bash
python -m unittest discover -s tests/closet/unit -t . -v
python -m src.closet.run --data-root "../closet-demo-data-60" --check --check-images
python -m src.closet.run --data-root "../closet-demo-data-60" --test
```

`make closet`, `make closet-check`, `make closet-test`, `make closet-unit`도 지원한다.
외부 데이터 경로는 `make closet DATA_ROOT="../closet-demo-data-60"`처럼 지정한다.
루트 `pytest`는 데이터가 필요 없는 옷장 단위 테스트만 수집한다. 데이터 버전별 통합 검사는 실행기의 `--test`로 별도 실행한다.

## 현재 동작과 통합 경계

남녀 각 30상품·318개 코디 연결, 저장/취소 및 취향 반영, 전체 보유 시 빈 유사상품 2개 보완까지 포함한다.
패키지 import로 변경하여 두 서버의 `server.py` 이름 충돌과 `sys.path` 조작을 제거했다.

이번 작업은 소스 구조와 실행 경로 정리다. 기존 팀 `src/serving/shop/`의 파일이나 FastAPI 라우터를 덮어쓰지 않았다.
옷장 화면은 기존처럼 8767, 코디 검색은 8765에서 작동하며 실행기 하나가 둘을 시작한다.
팀 회원 인증 통합과 Two-Tower 벡터 자체의 취향 반영은 별도 작업이다.
2026-11-22 기존 상품 추천 API의 후보·최종 정렬에 별도 취향 신호를 연결했다. 조회·구매 이력을 만들지 않으며, 실제 학습 모델을 붙인 품질·속도 확인은 남아 있다. [연결 방식과 실행 안내](RECOMMENDATION_CONNECTION.md).

## 추천 담당자 인계

- API: `GET /api/closet/preference-seed?user_id=...`
- 최신 선택·저장 코디에서 상품 ID와 상대 가중치를 생성하며 `operation=replace`로 이전 상태를 교체한다.
- 빈 `seed_products`는 이전 취향 벡터/후보 캐시를 제거하는 신호다. 실제 행동 이력은 유지한다.
- 현재 추천 모델과 같은 버전의 최종 64D 상품 벡터를 어댑터에 제공한다. 원시 FashionCLIP 512D 벡터를 직접 넣지 않는다.
- 자세한 계약: `preference_api.txt`, 저장 취향 규칙: `history/SAVED_PREFERENCE_NOTES.txt`.
- `history/` 문서는 이전 작업 기록이므로 그 안의 옛 실행 경로 대신 이 문서의 명령을 사용한다.

## GitHub 반영

이 폴더는 프로젝트 루트 구조다. 저장소 안에 이 폴더 자체를 한 겹 더 넣지 않는다.
PR에서 `src/closet/`, 화면 하위 폴더, 어댑터, 설정·요구사항·테스트·문서를 추가하고 기존 `closet-capstone/`를 제거한다.
`Makefile`, `README.md`, `.gitignore`는 옷장 관련 추가분만 병합한다.
제공된 ZIP 이후 팀원이 수정한 기존 코드를 이 작업본으로 통째로 덮어쓰지 않는다.

`file_mapping.json`에 이전 파일과 새 위치를 기록했다. 일회성 데이터 연구 스크립트는 실행 코드에 섞지 않았으며 원래 로컬 폴더에 보존되어 있다.
