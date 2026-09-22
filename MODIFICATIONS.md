# Modifications

This is a modified fork of [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP),
maintained by @chynggi for a single machine: **RTX 3060 12GB / i5-10400 / 62GB RAM / Linux**.

Distributed under the WanGP Community License 2.0 for Free Use only.
This notice satisfies §4.1(d) of that license (statement of modification and dates).
No sponsorship, endorsement or affiliation with the upstream Licensor is implied.

Upstream base: `5b9bc5c` (2026-09-21)

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
