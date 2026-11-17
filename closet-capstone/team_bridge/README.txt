콜드스타트 선택 → 추천 모델 전달 (2026-11-18)

현재 구현된 범위
- 무드보드 선택은 기존 데모의 코디 정렬에 바로 반영됩니다.
- GET /api/closet/preference-seed?user_id=<사용자ID> 로 별도 취향 스냅샷을 읽습니다.
- 모델을 호출하거나 팀 서버로 전송하지 않습니다. model_applied=false가 이 상태를 명시합니다.
- preference_adapter.py는 팀이 제공하는 학습된 64차원 상품 벡터를 받는 순수 함수입니다.
- 이번에 확인한 원본 (3) 폴더에는 이미지 특징을 추가하기 전 Two-Tower가 있습니다.
  최신 체크포인트·상품ID 대응표·FAISS 인덱스와의 실제 연결 및 품질 평가는 팀 코드에서 해야 합니다.

전달할 데이터
1. selected_outfit_ids: 원래 선택한 코디ID. 재현/변환에 사용.
2. effective_outfit_ids: 현재 검수 기준과 추천 범위를 통과한 선택.
   inactive_outfit_ids는 보관만 하며 seed_products에 반영하지 않습니다.
3. seed_products: 검수된 상품→코디 및 코디→상품 연결로 얻은 H&M 상품ID와 상대 가중치.
   product_id는 hm_0759118001 형태, source_article_id는 선행 0을 유지한 10자리 문자열.
   evidence에는 코디ID/영역ID/종류/유사도/연결 근거가 있습니다.
   실제 착용 상품이나 구매 의사가 확인된 상품이라는 의미가 아닙니다.
4. registration: 실제 입력한 성별·연령대. 미입력은 unknown.
   shopping_audience는 명시적으로 변경할 수 있는 쇼핑 범위이며 가입 성별과 분리됩니다.
5. dataset_version, selection_dataset_version, selection_revision, selected_at,
   last_action_at, snapshot_id: 선택 원본과 재처리 결과를 식별합니다.

저장 규칙
- operation=replace: 이 사용자의 이전 취향 스냅샷을 교체하세요. 재조회/재시도를 누적하지 마세요.
- status=not_started/selected/skipped를 구분합니다.
- 최초 건너뛰기: seed_products=[] → 기존 인기/가입정보 기반 기본 추천 유지.
- 재방문 건너뛰기: 기존 선택을 유지하며 retained_previous_selection=true.
  건너뛰기는 기존 취향을 삭제하는 동작이 아닙니다.
- 코디 다시 선택: 이전 선택을 교체. 범위 변경: 범위에 맞지 않는 선택은 비활성 처리.
- event_type=outfit_preference, signal_strength=weak.
- 코디를 고른 행위를 상품 클릭/장바구니/구매 또는 보유 이력으로 만들지 마세요.
- 이 로컬 API의 user_id는 시연용입니다. 실제 서비스의 인증 처리는 이번 범위에 없습니다.

가중치의 의미
- 선택한 각 코디에 같은 비중, 코디 안 각 의류 영역에 같은 비중을 줍니다.
- 한 영역의 유사상품 수가 많으면 해당 비중을 나눕니다. 전체 합은 1입니다.
- cosine similarity는 검수 근거이며 확률로 변환하지 않습니다.
- 이는 초기 휴리스틱이며 학습/검증으로 최적화된 선호 확률이 아닙니다.

추천 담당자 연결 순서
1. 위 API JSON을 읽고 사용자별 최신 스냅샷으로 보관합니다.
2. 팀 상품ID가 다르면 검증된 id_map = {데모상품ID:팀상품ID}를 만듭니다.
   hm_ 접두어를 지우거나 P00001 같은 순번을 임의로 붙여 매칭하지 마세요.
3. 현재 FAISS 인덱스와 동일한 체크포인트에서 만든 item_ids와 N×64 상품 벡터를 준비합니다.
   행 순서와 ID가 정확히 대응해야 합니다. PAD 행은 상품 목록에 넣지 않습니다.
   FashionCLIP 원본 512차원이나 학습 중간 32차원은 여기에 넣지 않습니다.
4. seed = build_seed_vector(payload, item_ids, item_vectors, id_map)
   매칭 안 된 상품은 제외되며 coverage_weight와 unmapped_product_ids로 확인할 수 있습니다.
   전부 미매칭/영벡터이면 vector=None이고 기존 추천으로 돌아갑니다.
5. 사용자의 유효한 64차원 query가 있으면:
   query, alpha = blend_user_query(query, seed, real_event_count=실제행동수)
   반환 query를 기존 FAISS 검색 직전에 사용합니다. 새로운 인덱스 생성은 불필요합니다.
   기본 alpha=0.2×매칭비중/(1+실제행동수/5)이며 평가가 필요한 보수적 시작값입니다.
6. 유효한 user query가 없는 신규회원은 rank_seed_candidates(seed, item_ids,
   item_vectors, allowed_mask, limit=20)로 별도 후보를 만들 수 있습니다.
   allowed_mask는 판매 가능/생성시점/추천범위 등 기존 상품 필터를 적용한 배열이어야 합니다.
   후보를 인기 후보 등과 합칠 때 중복 상품을 제거하고 기존 재정렬로 넘깁니다.
7. (3) 원본의 마지막 단계에는 이력 부족 시 인기 추천으로 대체하는 분기가 있습니다.
   앞 단계에 취향 후보를 넣더라도 이 분기가 덮어쓸 수 있으므로 최종 결과까지 확인해야 합니다.
   이력 부족 여부는 실제 행동만으로 판정하고 선택상품을 session_clicks/recent history에 넣지 마세요.
8. 실제 모델로 미매칭 비율, 결과 중복, 대상 범위, 취향 선택 전후 순위 변화를 평가합니다.
   이 묶음의 4개 어댑터 테스트는 합성 64차원 벡터로 계산/입력 계약을 확인할 뿐 추천 품질을 증명하지 않습니다.

예시와 검사
- examples/preference_seed.example.json: EXAMPLE_HANDOFF라는 별도 테스트 사용자의 실제 API 응답.
  M/20s는 예시값이며 실제 사용자 정보가 아닙니다. 원본 선택 코디와 연결 상품은 데모 데이터입니다.
- python tests/test_preference_adapter.py
- 통합 API 테스트: run_demo.py --data-root ../closet-demo-data-expanded --test
- 기존 데모의 코디 정렬은 태그 기반입니다. 64차원 어댑터를 추가했다고 데모 정렬이
  Two-Tower 기반으로 바뀐 것은 아닙니다.
