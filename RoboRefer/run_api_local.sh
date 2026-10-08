#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin="${repo_dir}/../.conda-roborefer/bin/python"
vlm_path="${repo_dir}/models/RoboRefer-2B-SFT"
depth_path="${repo_dir}/models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth"

cd "${repo_dir}/API"
exec env \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${python_bin}" -s api.py \
    --port "${ROBOREFER_PORT:-25547}" \
    --depth_model_path "${depth_path}" \
    --vlm_model_path "${vlm_path}"
