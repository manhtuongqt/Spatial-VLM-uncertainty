#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seed in 24082026 24082027 24082028; do
  "${SCRIPT_DIR}/run.sh" train \
    --config "new/configs/v3_seed_${seed}.json" \
    --run-name "pcrau_target_v3_seed_${seed}" \
    "$@"
done
