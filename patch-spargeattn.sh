#!/usr/bin/env bash
# SpargeAttn 빌드 — CUDA 13.x 대응 1줄 패치 포함.
#
# 원인: csrc/mma.cuh 이 `namespace mma {` 안에서 <assert.h> 를 include 한다.
# 그러면 __assert_fail 이 mma 네임스페이스에 선언되고 glibc 의 _ASSERT_H_DECLS
# 가드가 켜져서, 이후 전역에서 <assert.h> 를 다시 include 해도 선언이 생기지 않는다.
# CUDA 13.x 의 cuda_fp8/fp6/fp4.hpp 는 host/device inline 함수에서 assert() 를
# 쓰기 때문에 `identifier "__assert_fail" is undefined` 로 빌드가 깨진다.
# 해결: <assert.h> include 를 namespace 밖(파일 상단)으로 옮긴다.
set -e
cd "$(dirname "$0")"
SRC=.build/SpargeAttn
if [ ! -d "$SRC/.git" ]; then
  mkdir -p .build
  git clone https://github.com/woct0rdho/SpargeAttn.git "$SRC"
fi
python3 - "$SRC/csrc/mma.cuh" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
if "#include <type_traits>\n#include <assert.h>\n" in s:
    print("already patched")
else:
    a = "#include <type_traits>\n\nnamespace mma{"
    b = "#else\n#include <assert.h>\n#define RUNTIME_ASSERT(x) assert(0 && x)"
    if a not in s or b not in s:
        sys.exit("pattern not found - upstream changed, re-check manually")
    s = s.replace(a, "#include <type_traits>\n#include <assert.h>\n\nnamespace mma{", 1)
    s = s.replace(b, "#else\n#define RUNTIME_ASSERT(x) assert(0 && x)", 1)
    open(p, "w").write(s); print("patched")
PY
exec uv pip install --no-build-isolation "$SRC"
