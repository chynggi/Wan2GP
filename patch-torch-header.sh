#!/usr/bin/env bash
# torch 2.10 헤더 1줄 패치 — CUDA 13.3 nvcc가 ATen/core/List_inl.h:202의
# `typename decltype(...)::difference_type` 를 처리하지 못해 CUDA 확장 빌드가 깨진다.
# std::vector<IValue>::difference_type == std::ptrdiff_t 이므로 의미는 동일하다.
# torch 재설치/업그레이드 후 SageAttention을 다시 빌드하려면 이 스크립트를 먼저 실행할 것.
set -e
H="$(dirname "$0")/.venv/lib/python3.11/site-packages/torch/include/ATen/core/List_inl.h"
[ -f "$H.orig" ] || cp "$H" "$H.orig"
python3 - "$H" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "  return {impl_->list.begin() + static_cast<typename decltype(impl_->list)::difference_type>(pos)};"
new = "  return {impl_->list.begin() + static_cast<std::ptrdiff_t>(pos)};"
if new in s:
    print("already patched")
elif old in s:
    open(p, "w").write(s.replace(old, new)); print("patched")
else:
    sys.exit("pattern not found - torch version changed, re-check manually")
PY
