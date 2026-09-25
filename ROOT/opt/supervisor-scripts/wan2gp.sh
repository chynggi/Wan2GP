#!/bin/bash

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/cleanup_generic.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Wan2GP"

echo "Starting Wan2GP"

. /venv/main/bin/activate

# Wait for provisioning to complete
while [ -f "/.provisioning" ]; do
    echo "$PROC_NAME startup paused until instance provisioning has completed (/.provisioning present)"
    sleep 10
done

# Required for SDL audio (Wan2GP imports pygame/SDL on startup)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}

# The base image exports /usr/local/cuda/lib64 on LD_LIBRARY_PATH. That system
# cuBLAS/cuBLASLt shadows the CUDA 13 libraries bundled inside torch 2.10.0+cu130
# and every nn.Linear fails with CUBLAS_STATUS_NOT_INITIALIZED. Let torch use its
# own libraries (same fix as run-wangp.sh; see MODIFICATIONS.md).
unset LD_LIBRARY_PATH

cd "${WORKSPACE}/Wan2GP"

pty python wgp.py --server-port ${WAN2GP_PORT:-7860} ${WAN2GP_ARGS:-} 2>&1
