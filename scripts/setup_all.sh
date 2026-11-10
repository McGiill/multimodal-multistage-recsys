#!/usr/bin/env bash
# 처음부터 끝까지: 카탈로그 → 검색 인덱스 → 데이터 생성 → 분할 → 검색 평가셋 → 학습 → 평가
# 준비: H&M 원본(data/raw/hm)이 있어야 한다.
# 순서 이유: 시뮬레이터는 카탈로그와 검색 인덱스(configs/simulator.yaml search.engine: clip)를 쓰므로
#           데이터 생성보다 먼저 만든다. 검색 평가(make eval)는 검색 평가셋이 필요하다.
set -euo pipefail
make catalog
make index
make data
make split
make search-eval-set
make train
make eval
