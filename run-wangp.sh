#!/usr/bin/env bash
# WanGP 실행 — RTX 3060 12GB / 62GB RAM
cd "$(dirname "$0")"

# 시스템 CUDA 13.3 라이브러리가 torch 2.10+cu130 번들 cuBLAS(13.0)를 덮어써서
# CUBLAS_STATUS_NOT_INITIALIZED 를 유발한다. torch 자체 라이브러리를 쓰도록 제거.
unset LD_LIBRARY_PATH

# SageAttention2 빌드가 성공했으면 sage2, 아니면 sdpa
ATTN=sdpa
if .venv/bin/python -c "
from shared.sage2_core import is_sage2_supported
import sys; sys.exit(0 if is_sage2_supported() else 1)" 2>/dev/null; then
  ATTN=sage2
fi
echo "[run-wangp] attention=$ATTN"

exec .venv/bin/python wgp.py \
  --attention "$ATTN" \
  --profile 2 \
  --perc-reserved-mem-max 0.6 \
  --listen \
  --advanced \
  --open-browser \
  "$@"
