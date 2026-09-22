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
