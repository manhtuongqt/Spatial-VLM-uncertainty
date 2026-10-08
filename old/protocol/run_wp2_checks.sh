#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_ROOT="${1:-${PROJECT_ROOT}/datasets/roborefer_dataset_v1_prototype_20260818_155606}"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"
unset PYTHONNOUSERSITE || true
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=""

set +u
source /opt/ros/humble/setup.bash
source "${PROJECT_ROOT}/install/setup.bash"
set -u
cd "${PROJECT_ROOT}"

/usr/bin/python3 -m pytest --color=no -q \
  protocol/test_wp2_dataset.py \
  protocol/test_wp1_depth_sensitivity.py \
  ur3/ur3_perception/test/test_projection.py \
  ur3/ur3_perception/test/test_depth_component_gate_v2.py \
  ur3/ur3_perception/test/test_roborefer_adapter.py \
  ur3/ur3_perception/test/test_roborefer_dimension_comparator.py \
  ur3/ur3_perception/test/test_roborefer_pilot.py \
  ur3/ur3_moveit_control/test/test_pose_generation.py \
  ur3/ur3_moveit_control/test/test_d435i_description.py \
  2>&1 | tee "${LOG_DIR}/wp2_pytest.log"

/usr/bin/python3 protocol/wp2_validate.py \
  --dataset-root "${DATASET_ROOT}" --selection smoke \
  2>&1 | tee "${LOG_DIR}/wp2_smoke_final_validate.log"

/usr/bin/python3 protocol/wp2_validate.py \
  --dataset-root "${DATASET_ROOT}" --selection all \
  2>&1 | tee "${LOG_DIR}/wp2_full_final_validate.log"

/usr/bin/python3 protocol/wp2_leakage_validator.py \
  --dataset-root "${DATASET_ROOT}" --selection all \
  2>&1 | tee "${LOG_DIR}/wp2_leakage.log"

/usr/bin/python3 protocol/wp2_replay_validator.py \
  --dataset-root "${DATASET_ROOT}" --selection all \
  2>&1 | tee "${LOG_DIR}/wp2_replay.log"

echo "WP2_CHECKS_PASS dataset=${DATASET_ROOT}"
