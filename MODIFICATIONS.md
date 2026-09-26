# Modifications

This is a modified fork of [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP),
maintained by @chynggi for a single machine: **RTX 3060 12GB / i5-10400 / 62GB RAM / Linux**.

Distributed under the WanGP Community License 2.0 for Free Use only.
This notice satisfies §4.1(d) of that license (statement of modification and dates).
No sponsorship, endorsement or affiliation with the upstream Licensor is implied.

Upstream base: `9ffd0a67` (2026-09-22)

---

## 2026-09-21 — `run-wangp.sh` (new)

Launcher that pins the environment this machine needs.

**`unset LD_LIBRARY_PATH`** is the important line. The system has CUDA **13.3**
on `LD_LIBRARY_PATH`, which overrides the cuBLAS **13.0** bundled with
torch 2.10.0+cu130 (`.venv/.../nvidia/cu13/lib/libcublasLt.so.13`). The mismatch
makes every `nn.Linear` on CUDA fail with:

```
CUDA error: CUBLAS_STATUS_NOT_INITIALIZED
  when calling `cublasLtMatmulAlgoGetHeuristic(...)`
```

Reproducible without WanGP:

```python
torch.nn.Linear(5120, 5376, dtype=torch.bfloat16, device='cuda')(x)   # fails
```

Text encoding survives it because the quanto/comfy-kitchen INT8 and GGUF paths
barely touch cuBLAS; the DiT's first plain Linear is where it surfaces.
Not reproducible under torch 2.12.1+cu130 — this is specific to torch 2.10.

The script also auto-selects `sage2` when the build is importable, else `sdpa`.

## 2026-09-21 — `patch-torch-header.sh` (new)

One-line patch to `torch/include/ATen/core/List_inl.h` in the venv, required to
build SageAttention 2.2 CUDA extensions with CUDA 13.3's nvcc.

`nvcc` cannot parse `static_cast<typename decltype(impl_->list)::difference_type>(pos)`
and reports "need 'typename'" on a line that already has it. Replaced with
`static_cast<std::ptrdiff_t>(pos)` — `std::vector<IValue>::difference_type` *is*
`std::ptrdiff_t`, so the semantics are unchanged.

Not a GCC issue: g++-13, g++-14 and g++-15 all compile the header fine as plain C++.
Re-run this script after any torch reinstall, before rebuilding SageAttention.

## 2026-09-21 — `fetch-models.sh` (new)

Pre-downloads DiT weights by explicit filename with `hf`, instead of letting the
UI resolve them. WanGP stores the per-model text-encoder / VAE choice in the
`config` field of `settings/<model>_settings.json` (see `shared/config_groups.py`);
when that field is empty, "Default" can fall back to the BF16 variant and pull
tens of GB that will not fit this machine's RAM. Naming files directly avoids it.

## 2026-09-22 — `patch-spargeattn.sh` (new)

Builds SpargeAttn with a one-line fix for CUDA 13.x.

`csrc/mma.cuh` includes `<assert.h>` *inside* `namespace mma`, which declares
`__assert_fail` in that namespace and trips glibc's `_ASSERT_H_DECLS` guard, so
later global includes are no-ops. CUDA 13.x's `cuda_fp8/fp6/fp4.hpp` call
`assert()` from host/device inline functions and fail with
`identifier "__assert_fail" is undefined`. The include is moved to file scope.

## 2026-09-22 — `.gitignore`

Added: third-party plugins cloned into `plugins/` (each carries its own `.git`),
`/.build/`, `/workspaces/`, and `*.bak*`.

Note the upstream patterns `/plugins/wan2gp-*/` and `/plugins/wangp-*/` are
lowercase and do not match mixed-case directory names on a case-sensitive
filesystem.

## 2026-09-22 — prompt enhancer output budget

Raised `*_prompt_enhancer_max_tokens*` for MiniMax H3 and LTX-2.

`shared/prompt_enhancer/prompt_enhance_utils.py` makes a single `generate()`
call with one `max_new_tokens`, so when thinking is enabled (`"K"` in the
enhancer mode, `wgp.py:6449`) the chain-of-thought and the final prompt share
that one budget. There is no separate reasoning budget. Running out mid-thought
truncates or loses the prompt.

(`llm_budget: 18000` in `shared/prompt_enhancer/assets.py` is unrelated — it
becomes an mmgp VRAM budget in MB, not a token count.)

| file | key | before | after |
|---|---|---:|---:|
| `models/minimax_h3/minimax_h3_handler.py:322` | `text_..._max_tokens` | 1024 | 3072 |
| `models/minimax_h3/minimax_h3_handler.py:323` | `text_..._max_tokens1` | 2048 | 4096 |
| `models/minimax_h3/minimax_h3_handler.py:514-515` | `text_/video_..._max_tokens` | 2048 / 1024 | 4096 / 3072 |
| `models/ltx2/ltx2_handler.py:617-619` | `text_/video_/image_..._max_tokens1` | 1536 | 3072 |
| `models/ltx2/ltx2_handler.py:682-683` | `text_/video_..._max_tokens1` | 1024 | 3072 |

H3 needs the room: its prompt format is three sections
(`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`).
The Ref2VA path keeps the larger budget upstream already gave it.

`models/ltx2/ltx_audio_tts_handler.py` (768 / 1024) is left unchanged — TTS
prompts are short.

`max_new_tokens` is only a cap; generation still stops at EOS, so short outputs
stay short. Thinking models do tend to think longer when given room, so
enhancement takes more wall time.

## 2026-09-23 — MiniMax H3 W4A8 `s_rel` fp8 fix (sm_86)

`shared/qtypes/asym_w4a8_int8.py` — added `_fp8_e4m3_supported()` and cast
`weight_s_rel` to `bfloat16` in `convert_to_quanto` when the target GPU cannot
run fp8e4nv kernels.

W4A8 community checkpoints (e.g. `10Eros_Max_h3_TURBO-hybrid_beta5_w4a8`) store
the relative scale as `float8_e4m3fn`. `_decode_w4a8_triton` loads it inside a
Triton kernel, and Triton refuses to codegen fp8e4nv on pre-Ada GPUs:

```
CompilationError: type fp8e4nv not supported in this architecture.
The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
```

The existing backend probe (`quanto_int8_triton._runtime_compatible`) only checks
`cc_major >= 8`, which is true for the RTX 3060 (sm_86), so it picks Triton and
the kernel fails at compile time. The official DeepBeepMeep int8 checkpoints use
a F32 `weight_scale` (`int8_convrot`) and never hit this.

Cast is lossless (every e4m3 value is exactly representable in bf16) and is
skipped on Ada/Hopper (sm >= 8.9), where the fp8 path still works.

## 2026-09-23 — 10Eros-Max H3 finetune presets (new)

`finetunes/minimax_h3_ref2va_pruned_10eros_max_turbo_hybrid.json` and
`finetunes/minimax_h3_fl2va_pruned_10eros_max_turbo_hybrid.json`.

Register `TenStrip/10Eros-Max` as a local-path TURBO finetune for both the pruned
Ref2VA and FL2VA architectures. The checkpoint's 474 weight bases match
`MiniMax-H3-*_pruned` in shape exactly, so it is a pruned (20B) model, not the
full 33B. The TURBO delta is baked into the weights, so no turbo LoRA is listed.
Defaults: `num_inference_steps=6`, `guidance_scale=1.0`,
`sample_solver=res_multistep`.

The non-Turbo pair
`finetunes/minimax_h3_{ref2va,fl2va}_pruned_10eros_max_hybrid.json` registers
`ckpts/10Eros_Max_h3_hybrid_beta5_int8.safetensors` (same model without the baked
turbo delta) under the same two architectures and inherits the H3 full-step
defaults (`euler`, 20 steps).

## 2026-09-23 — comfy-kitchen native sm_86 build

Rebuilt the `comfy-kitchen` CUDA extension natively for sm_86 instead of relying
on PTX JIT:

```
env -u LD_LIBRARY_PATH COMFY_CUDA_ARCHS=86 \
  uv pip install --python /media/chynggi/Data-S/Wan2GP/.venv/bin/python \
  -e /media/chynggi/EXTRA/Linux-ComfyUI/comfy-kitchen/. --no-build-isolation
```

`nanobind` was missing from the venv (required because `--no-build-isolation`);
system cmake 4.2.3 and ninja are used. The source tree also held a stale
`_C.abi3.so` from a 3.12+ build; `comfy_kitchen/backends/cuda/__init__.py` loads
`_C.abi3.so` first and it failed with `undefined symbol: PyObject_GetTypeData`,
hiding the freshly built `_C.cpython-311-...so`. Renamed it to
`_C.abi3.so.py312-stale-bak` (safe to delete). `_probe_kitchen()` now returns
`available` and `resolve_backend("auto") == "kitchen"`.

## 2026-09-25 — 10Eros-Max H3 TURBO presets: W4A8 → INT8

`finetunes/minimax_h3_ref2va_pruned_10eros_max_turbo_hybrid.json` and
`finetunes/minimax_h3_fl2va_pruned_10eros_max_turbo_hybrid.json` now register
`ckpts/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors` instead of the previous
W4A8 checkpoint, which has been deleted (`ckpts/` and its HuggingFace download
cache entries). Names, descriptions and info panels were updated from W4A8 to
INT8; the sampling defaults are unchanged (`res_multistep`, 6 steps,
`guidance_scale=1.0`).

The matching saved presets in `settings/*10eros_max_turbo*_settings.json` had
their display `type` string updated to the new name. The INT8 checkpoint uses a
standard INT8 format rather than the W4A8 `weight_s_rel` fp8 layout, so it does
not need the sm_86 cast from the 2026-09-23 W4A8 fix.

## 2026-09-25 — vast.ai 이미지: 커스텀 deps 이미지 → vast 공식 파생 구조

`Dockerfile.vastai` 는 이전에 `nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04` 기반의
**의존성 전용 이미지**였고, 앱 코드를 런타임에 클론해 `/workspace/entrypoint.sh`
로 실행해야 했다. 그래서 Instance Portal / Caddy / Supervisor / Jupyter / SSH
같은 vast 공식 `vastai/wan2gp` 동작이 전혀 없었다.

이제 vast 공식 파생 이미지(`vast-ai/base-image` →
`derivatives/pytorch/derivatives/wan2gp`)와 동일한 구조로 재작성했다:

- 베이스: `vastai/pytorch:2.10.0-cuda-13.0.3-py311-24.04-2026-09-08`
  (torch 2.10.0+cu130, CUDA 13.0.3, cuDNN 9.14, Python 3.11, `/venv/main`,
  `ENTRYPOINT=/opt/instance-tools/bin/entrypoint.sh`). 이미지가 torch 를
  제공하므로 명시적 torch 설치와 `/opt/venv`·uv python 설치 단계는 제거했다.
- `ROOT/` (신규): 공식 파생 이미지의 오버레이를 그대로 vendoring.
  `etc/supervisor/conf.d/wan2gp.conf`, `opt/supervisor-scripts/wan2gp.sh`,
  `etc/vast_boot.d/05-caddy-localhost-ports.sh`,
  `etc/vast_capabilities.d/50-wan2gp.yaml`, `etc/vast_agents/wan2gp.md`,
  `LICENSES.md`. `wan2gp.sh` 에만 `unset LD_LIBRARY_PATH` 를 추가했다 — 베이스가
  `/usr/local/cuda/lib64` 를 `LD_LIBRARY_PATH` 에 넣어 torch cu130 번들 cuBLAS 를
  가리고 `CUBLAS_STATUS_NOT_INITIALIZED` 를 유발하기 때문(`run-wangp.sh` 와 동일한
  이유).
- 앱 코드는 `WAN2GP_REPO`/`WAN2GP_REF` 로 `/opt/workspace-internal/Wan2GP` 에
  클론한다(기본 포크 `main`). 베이스 이미지가 첫 부팅에 `/workspace/Wan2GP` 로
  복사하고, Supervisor 서비스 `wan2gp` 가 `wgp.py --server-port 7860` 을 상시
  실행한다. `entrypoint.sh` 는 이미지에서 더 이상 사용하지 않는다.
- `.github/workflows/docker-vastai.yml` 에 `wan2gp_ref` 입력과 `WAN2GP_REPO`/
  `WAN2GP_REF` build-arg 를 추가. `.dockerignore` 를 새로 추가해 빌드 컨텍스트를
  `ROOT/` + Dockerfile 만 남긴다.

커널 빌드 단계(SageAttention v2/v3, FlashAttention 2.7.2.post1, comfy-kitchen
sm120, Lightx2v/Nunchaku 휠)는 그대로 유지하되 모든 `RUN` 을 `/venv/main`
활성화 후 실행하도록 바꿨다(베이스의 `python3` 는 시스템 3.12 이므로). 커널
휠이 cp311 이라 베이스의 Python 3.11 과 맞고, torch 계열 핀은 requirements 에서
제거해 베이스 torch 를 단일 소스로 둔다.

## 2026-09-26 — 10Eros-Max H3 finetune presets: HuggingFace URL 추가

`finetunes/minimax_h3_{ref2va,fl2va}_pruned_10eros_max{,_turbo}_hybrid.json`
네 개가 체크포인트를 로컬 경로(`ckpts/...`)로만 참조하고 있었다. 그래서
`ckpts/` 가 비어 있는 원격 인스턴스에서 모델을 선택하면

```
Model 'ckpts/10Eros_Max_h3_hybrid_beta5_int8.safetensors' was not found locally
and no URL was provided to download it. Please add an URL in the model definition file.
```

로 실패한다 (`wgp.py` `download_models()`, http 가 아니면 즉시 예외). 이제 네
파일 모두 `TenStrip/10Eros-Max` 의 HuggingFace URL 을 가리킨다
(`10Eros_Max_h3_hybrid_beta5_int8.safetensors`,
`10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors`).

로컬 설치는 영향 없다: `shared/utils/files_locator.py` 의 `locate_file()` 이
URL 이면 basename 으로 `ckpts/` 를 먼저 검색하므로 기존 파일을 그대로 쓴다.
파일 정의는 `finetunes/*.json` 을 매번 다시 읽는 `refresh_model_defs()` 로
갱신되므로, JSON 수정 후 UI 의 모델 목록 새로고침(↻ / Alt+R)만으로 반영되고
프로세스 재시작은 필요 없다.

## 2026-09-26 — vast.ai: FlashAttention prebuilt 휠로 교체

`Dockerfile.vastai` 의 FlashAttention 소스 빌드(2.7.2.post1 sdist + setup.py
gencode 패치 + bundled cutlass CUDA 13 패치 + `FLASH_ATTENTION_FORCE_BUILD`
빌드)를 prebuilt 휠 설치로 교체했다:

`flash_attn-2.8.3+cu130torch2.10-cp311-cp311-linux_x86_64.whl`
(mjun0812/flash-attention-prebuild-wheels v0.9.0, vast.ai RTX 5090 검증 완료)

- 휠은 sm_80/90/100/120 fatbin 을 포함해 sm_120 커널을 제공하고, `Requires-Dist`
  가 `torch`/`einops` 무버전뿐이라 베이스 torch 2.10.0+cu130 과 충돌하지 않는다.
  `patch_flash_setup.py` 헤로독과 sdist 다운로드/빌드 단계가 통째로 사라진다.
- 휠이 함께 싣는 `hopper`(FA3 소스) top-level 패키지는 Wan2GP 가 참조하지 않고
  (FA3 는 `flash_attn_interface` 로만 취급 — `shared/attention.py`,
  `preprocessing/sam3/perflib/fa3.py`) site-packages 네임스페이스만 점유하므로
  설치 후 제거한다. 최종 검증 RUN 이 `hopper` 부재와 `flash_attn.__version__
  == '2.8.3'` 을 assert 한다.
- Wan2GP 호출부와 2.8.3 API 호환 확인: `_flash_attn_forward` 시그니처는
  `models/hyvideo/modules/attenion.py` 의 `>= 2.7.0` 분기
  (`window_size_left/right`, `softcap`, `alibi_slopes`)와 정확히 일치하고,
  `flash_attn_varlen_func`/`flash_attn_with_kvcache` 는 반환값 단일 tensor 유지.
  Windows 설치는 이미 2.8.3 휠(`setup_config.json` flash v210)을 쓰고 있어
  버전 축이 플랫폼 간에 일치한다.
