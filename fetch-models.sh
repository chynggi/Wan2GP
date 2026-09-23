#!/usr/bin/env bash
# 선택 모델 DiT 본체 사전 다운로드 (85.75 GiB)
# 텍스트 인코더/VAE 등 부속은 WanGP가 첫 모델 로드 시 자동으로 받는다.
set -euo pipefail
CK="$(dirname "$0")/ckpts"
mkdir -p "$CK"

get() {  # repo file
  echo "--- $2"
  hf download "$1" "$2" --local-dir "$CK"
}

get DeepBeepMeep/LTX-2  ltx-2.3-22b-distilled-Q4_K_M_light.gguf
get DeepBeepMeep/LTX-2  ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors
get DeepBeepMeep/Wan2.2 wan2.2_text2video_14B_high_quanto_mbf16_int8.safetensors
get DeepBeepMeep/Wan2.2 wan2.2_text2video_14B_low_quanto_mbf16_int8.safetensors
get DeepBeepMeep/Wan2.2 wan2.2_image2video_14B_high_quanto_mbf16_int8.safetensors
get DeepBeepMeep/Wan2.2 wan2.2_image2video_14B_low_quanto_mbf16_int8.safetensors

echo "=== 완료 ==="
df -h "$CK" | tail -1
