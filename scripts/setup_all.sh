#!/usr/bin/env bash
# 처음부터 끝까지: 데이터 생성 → 분할 → 인덱스 → 학습 → 평가
set -euo pipefail
make data
make split
make index
make train
make eval
