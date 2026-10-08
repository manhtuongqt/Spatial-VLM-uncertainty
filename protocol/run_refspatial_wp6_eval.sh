#!/usr/bin/env bash
# Run the fixed primary challenge with the identical RGB-D protocol used in WP5.
set -euo pipefail

cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$PWD/.conda-roborefer/lib:${LD_LIBRARY_PATH:-}"

for model in b0 b1; do
  ./.conda-roborefer/bin/python -s protocol/refspatial_b0_b1_eval.py \
    --model-id "$model" \
    --provenance results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/primary_challenge.jsonl \
    --out-root results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/evaluation
done
