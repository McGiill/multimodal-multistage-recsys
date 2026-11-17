최신 확장 데모 (2026-11-18)
권장 데이터: closet-demo-data-expanded
옷장 상품37개 유지 / 코디204장 / 상품→코디233연결 / 상품사진236장.
이전131코디·150연결에서 73코디·83연결 추가. 26개 상품은 코디6~10개 확보.
6개 상품은5개, 3개는4개, 원피스2개는1개씩 유지. 부적합 후보를 억지로 추가하지 않았습니다.
성인으로 보이는 의류 코디 선별이며 실제 나이/개인의 성별 확인을 의미하지 않습니다.
의류 구성 기준 남성복54장 / 여성복62장 / 공용88장.
남성복/여성복 범위를 선택하면 해당 범위+공용만 남으므로 상품별 표시 수는 줄어듭니다.

실행 (코드와 데이터 폴더를 나란히 두고 코드 폴더에서):
  python -m pip install -r requirements.txt
  python run_demo.py --data-root "../closet-demo-data-expanded"
  http://127.0.0.1:8767/shop/#/mypage
  http://127.0.0.1:8767/shop/#/onboarding
포트가 사용 중이면 --retrieval-port 8905 --shop-port 8907 추가 후8907로 접속.
종료 Ctrl+C. 사용자 기록은 코드 .local에 저장됩니다. 전달 ZIP에는 DB가 없습니다.

변경 동작
- 상품 선택 → 같은 옷장 화면 아래 코디5개 → 코디 더 보기로 나머지 후보 표시.
- 추천 범위 필터/기존 취향 정렬을 먼저 적용한 뒤5개를 자릅니다.
- 취향조사는 전체204개 후보를 추천 범위로 거르고25개씩 표시합니다.
  사진1~5개 선택 또는 선택 없이 건너뛰기가 가능합니다. 5개를 채울 필요가 없습니다.
- 선택 코디ID/가입정보/추천범위/초기관심 상품ID/버전을 별도 API로 제공합니다.
  GET /api/closet/preference-seed?user_id=<사용자ID>
- team_bridge/README.txt: 추천 담당자 전달 계약과64차원 벡터 어댑터 사용법.
- 실제 화면의 코디 정렬은 기존 태그 기반입니다. 최신 팀 Two-Tower에 연결된 상태는 아닙니다.

검사
  python run_demo.py --data-root "../closet-demo-data-expanded" --check --check-images
  python run_demo.py --data-root "../closet-demo-data-expanded" --retrieval-port 8915 --test
  python tests/test_preference_adapter.py

GitHub에는 closet-capstone 코드만 올립니다. 이 데이터는 별도 공유합니다.
모델/특징값 불필요. 전체 데이터 실시간 검색, 임의 새 상품 추천, 공개 배포는 이번 범위가 아닙니다.

--- 이전 버전 설명 (최신 실행/규모는 위 내용 기준) ---
최신 전체 데이터 재선별 시연 (2026-11-17)
권장 데이터: closet-demo-data-full-reviewed
상품37개 / 코디131장 / 150연결. 전체 무신사8,626장의 기존 특징값에서 재검색 후 선별.
기존보다 코디42장·연결43개 추가. 모델/특징값 없이 실행하는 소형 모드입니다.
코드 폴더에서: python run_demo.py --data-root "../closet-demo-data-full-reviewed"
검사: python run_demo.py --data-root "../closet-demo-data-full-reviewed" --retrieval-port 8895 --test
새 데이터 전용14테스트와 실제 화면 확인 완료. 같은 실행기는 이전 small/full 묶음도 지원합니다.
아래89장/107연결/15MB 설명은 이전 closet-demo-data-small 버전 기준입니다.

콜드스타트·마이페이지 옷장 로컬 캡스톤 코드
작성:2026-11-12

이 폴더를 GitHub 브랜치에 올리면 됩니다.
사진,데이터셋,특징값,모델,선별 데이터JSON,사용자DB는 포함하지 않았습니다.
실제 사진이 있는 데모를 실행하려면 아래 별도 데이터 묶음이 필요합니다.

팀 시연용 권장:closet-demo-data-small (약15MB)
- 옷장37상품/코디89장/107연결,유사상품 포함147상품 사진만 사용합니다.
- 기존 전체 검색에서 검수한 결과를 저장한 시연 모드입니다.
- 취향 선택·조회에 따른 정렬,옷장 추가/삭제,성별 범위 필터는 계속 동작합니다.
- 전체10만 상품 새 검색,시연 범위 밖 입력 상품 추천은 지원하지 않습니다.
- 모델과 특징값 파일도 불필요합니다. 화면용 사진과 저장된 연결 결과만 씁니다.
- 반드시 소형 묶음 지원이 추가된 최신 코드와 함께 사용하세요.

코드 폴더와 작은 데이터 폴더를 나란히 두었을 때:
  python -m pip install -r requirements.txt
  python run_demo.py --data-root "../closet-demo-data-small"
  옷장:http://127.0.0.1:8767/shop/#/mypage
  취향조사:http://127.0.0.1:8767/shop/#/onboarding

위 명령은 이 README가 있는 코드 폴더에서 실행합니다.

현재 기능
- 기존 쇼핑몰 기반 코디 사진 무드보드 모달,배경 블러,최대5장 선택/건너뛰기
- 마이페이지 옷장:상의/하의/아우터/원피스/신발
- 상품 선택 시 같은 화면 아래 코디 표시
- 코디 상세의 유사상품,옷장 추가/삭제,보유 상품 제외
- 사용자별 선택/조회/옷장 저장 및 태그 기반 코디 정렬
- 선별 시연37상품/코디89장/107연결
- 팀 Two-Tower로 취향 전달,실제 회원 인증,라이브 배포는 미구현

1. 실행 환경
- Python3.10 이상 권장. 이 패키지는 Windows/Python3.12 환경에서 확인.
- 데모 실행에는 NumPy만 필요. GPU/PyTorch/모델 다운로드 불필요.
- Windows PowerShell에서 이 README가 있는 폴더로 이동한 뒤:

  py -3.12 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

macOS/Linux는 python3 -m venv .venv 생성 후 .venv/bin/python을 사용하면 됩니다.
해당 OS의 실제 실행 검증은 하지 않았습니다.

2. 별도 데이터 준비
팀에서 전달받은 데이터 묶음을 예를 들어 D:\closet-data에 둡니다.
화면 시연만 하면 closet-demo-data-small(약15MB)을 사용하세요.
전체 검색까지 필요하면 closet-demo-data(약4.42GB)를 사용합니다.
DATA_REQUIREMENTS.txt에 정확한 파일 목록과 폴더 구조가 있습니다.
현재 작성자 PC에서는 기존 연구 작업 루트가 그대로 --data-root가 됩니다.
새로 Hugging Face 원본만 다운로드하면 실행되는 구조가 아닙니다.
전체 검색 모드는 기존 추출 특징값도 필요합니다. 소형 시연 모드는 저장된 결과를 사용합니다.

데이터 준비 확인:
  .\.venv\Scripts\python.exe run_demo.py --data-root "D:\closet-data" --check

모든 이미지 경로까지 확인하려면(상품 이미지가 많아 시간이 걸릴 수 있음):
  .\.venv\Scripts\python.exe run_demo.py --data-root "D:\closet-data" --check --check-images

3. 데모 실행
  .\.venv\Scripts\python.exe run_demo.py --data-root "D:\closet-data"

브라우저:
  http://127.0.0.1:8767/shop/#/mypage

데모 이용:
- 처음에는 새 사용자 옷장이 비어 있습니다. 옷 추가에서 시연 묶음을 담습니다.
- 상의/하의/아우터/원피스/신발을 선택하고 상품을 누르면 아래 코디가 바뀝니다.
- 신규 사용자 홈으로 이동하면 첫 취향조사 모달을 확인할 수 있습니다.
- 원본 작성자의 옷장·회원 정보·조회 기록은 이 패키지에 포함하지 않았습니다.

종료:실행 터미널에서 Ctrl+C. 이번 실행이 만든 서버만 종료합니다.
신규 사용자DB는 .local/profiles.sqlite3에 생성되며 Git 업로드에서 제외됩니다.
원하는 위치에 보관하려면 --state-dir "D:\closet-local-state"를 추가합니다.

포트가 이미 사용 중이면:
  .\.venv\Scripts\python.exe run_demo.py --data-root "D:\closet-data" --retrieval-port 8875 --shop-port 8877
  브라우저:http://127.0.0.1:8877/shop/#/mypage

4. 팀 검색/추천 서버와 함께 사용
- 별도로 팀 API8000을 실행하면 기존 상품 검색·상품 추천 경로를 전달합니다.
- 다른 주소라면 --team-url http://127.0.0.1:8000을 변경합니다.
- 팀 API 없이도 옷장·코디·유사상품 데모는 동작합니다.
- 팀 API 미실행 시 팀 추천/검색은 사용 불가로 표시하며 결과를 만들어 넣지 않습니다.
- 코디 무드보드 선택이 팀 Two-Tower에 전달되는 것은 아닙니다.
- 이 패키지는 팀 src/serving/shop/을 자동 교체하지 않는 별도 로컬 실행본입니다.

5. 테스트
DATA_REQUIREMENTS.txt의 테스트용JSON2개도 준비한 뒤:
  .\.venv\Scripts\python.exe run_demo.py --data-root "D:\closet-data" --retrieval-port 8875 --test

검색 서버를 띄우고 임시 포트·임시DB로 통합26테스트를 실행한 후 종료합니다.
소형 묶음을 지정하면 해당 범위의 통합24테스트를 자동으로 실행합니다.
전체107입력/미검수 후보/제외된 원본 사진을 요구하는5개 검사는 소형 모드에서 제외하고,
전체 이미지 제공·제외 사진 차단·화면 파일과 신규 사용자 상태 검사3개를 추가했습니다.
데이터가 없으면 이 테스트는 실행할 수 없습니다.
데이터 없이 경로 처리와 유사상품 경로 회귀 테스트9개를 확인하려면:
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v

6. 폴더 안내
- run_demo.py:데이터 사전 점검,검색/화면 서버 실행,테스트,종료
- build_compact_data.py:기존 전체 데이터에서 검수된 시연 결과와 사진만 추리는 도구
- local_closet_demo/compact_catalog.py:소형 시연 데이터 로더,사진·결과 무결성 확인
- local_closet_demo/server.py:저장된 특징값을 이용한 검색API
- local_closet_demo/runtime_paths.py:외부 데이터 경로 연결
- local_closet_demo/shop_integration/shop/:콜드스타트·옷장 UI
- local_closet_demo/shop_integration/:로컬API,SQLite,취향정렬,검수필터,테스트
- local_closet_demo/team_integration/:코디 연결 조회 보조 코드
- research/:전체 전처리/자동 분류 원본 연구코드 보관 및 별도 재현 안내
- SOURCE_FILES.json:복사한 원본 코드의 경로·해시. 이식 수정 전 기준이며 변경점은 PORTING_NOTES.txt 참고.
- .gitignore:데이터/모델/사용자DB/임시 산출물 업로드 방지

7. GitHub에 올리는 방법
팀 저장소의 새 브랜치에서 이 폴더를 예를 들어 demos/closet-capstone/로 복사합니다.
기존 src/,configs/ 파일을 덮어쓰지 않습니다.

  git switch -c codex/cold-start-closet-ui
  git add demos/closet-capstone
  git diff --cached --stat
  git commit -m "Add local cold-start moodboard and closet demo"
  git push -u origin codex/cold-start-closet-ui

현재 이 폴더 자체에서는 브랜치 생성/커밋/업로드를 하지 않았습니다.
팀 저장소의 기존 데이터/코드와 합치는 작업은 검토 후 별도로 진행합니다.
