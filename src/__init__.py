"""프로젝트 루트 패키지.

macOS: pip로 설치한 torch와 faiss는 각자 OpenMP 런타임(libomp) 복사본을 갖고 있어서, 둘 다 스레드 풀을
만들면 segfault가 난다 (API 워커 스레드, torchvision 이미지 전처리 등에서 재현). 그래서 macOS에서는
OpenMP 스레드를 1개로 고정한다. 단건 요청 서빙은 이쪽이 오히려 빠르고(M1 Max CPU p95: text 22ms /
image 64ms / hybrid 91ms), 인덱싱은 GPU(MPS)가 주로 일해서 영향이 작다.
torch·faiss보다 먼저 실행되어야 하므로 모든 src.* 모듈이 거치는 여기에 둔다.
다른 값을 쓰려면 실행 전에 OMP_NUM_THREADS 환경변수를 직접 지정하면 된다.
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
