#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin="${repo_dir}/../.conda-roborefer/bin/python"

image_path="${1:-${repo_dir}/assets/tabletop.jpg}"
prompt="${2:-Pick the apple in front of the logo side of the leftmost cup.}"
output_path="${3:-${repo_dir}/assets/my_tabletop_result_rgbd_test.jpg}"

cd "${repo_dir}/API"
exec env PYTHONNOUSERSITE=1 "${python_bin}" -s use_api.py \
    --image_path "${image_path}" \
    --prompt "${prompt}" \
    --output_path "${output_path}" \
    --url "http://127.0.0.1:${ROBOREFER_PORT:-25547}"
