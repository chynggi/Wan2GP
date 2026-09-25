# WanGP 실행 (Windows) — RTX 3060 12GB / 62GB RAM
Set-Location $PSScriptRoot

# Windows에서는 torch가 import 시 자체 torch\lib의 cuBLAS/cudart(13.0)를 먼저 로드하므로
# run-wangp.sh의 LD_LIBRARY_PATH 해제에 해당하는 조치는 필요 없다.
$Python = Join-Path $PSScriptRoot 'venv-windows\Scripts\python.exe'

# SageAttention2 빌드가 성공했으면 sage2, 아니면 sdpa
$Attn = 'sdpa'
& $Python -c @'
from shared.sage2_core import is_sage2_supported
import sys; sys.exit(0 if is_sage2_supported() else 1)
'@ 2>$null
if ($LASTEXITCODE -eq 0) { $Attn = 'sage2' }
Write-Host "[run-wangp] attention=$Attn"

& $Python wgp.py `
  --attention $Attn `
  --profile 2 `
  --perc-reserved-mem-max 0.6 `
  --listen `
  --advanced `
  --open-browser `
  @args
exit $LASTEXITCODE
