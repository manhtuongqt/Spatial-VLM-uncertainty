#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PILOT_ROOT="${PROJECT_ROOT}/results/roborefer_pilot_v0_20260813_173305"
PILOT_RELATIVE="results/roborefer_pilot_v0_20260813_173305"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=""

set +u
source /opt/ros/humble/setup.bash
set -u
cd "${PROJECT_ROOT}"

pilot_digest() {
  (
    cd "${PROJECT_ROOT}"
    find "${PILOT_RELATIVE}" -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

START_EPOCH="$(date +%s)"
PILOT_DIGEST_BEFORE="$(pilot_digest)"

colcon build \
  --base-paths ur3 \
  --packages-up-to ur3_perception \
  --allow-overriding ur_controllers ur_dashboard_msgs \
  --cmake-clean-cache \
  --cmake-args \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  --event-handlers console_cohesion+ \
  2>&1 | tee "${LOG_DIR}/wp0_build.log"

set +u
source "${PROJECT_ROOT}/install/setup.bash"
set -u

/usr/bin/python3 - <<'PY' | tee "${LOG_DIR}/wp0_interface_import.log"
from ur3_perception_interfaces.msg import ObjectObservation
print("INTERFACE_IMPORT_OK", ObjectObservation.__name__)
PY

/usr/bin/python3 -m pytest --color=no -q \
  ur3/ur3_perception/test/test_projection.py \
  ur3/ur3_perception/test/test_depth_component_gate_v2.py \
  ur3/ur3_perception/test/test_roborefer_adapter.py \
  ur3/ur3_perception/test/test_roborefer_dimension_comparator.py \
  ur3/ur3_perception/test/test_roborefer_pilot.py \
  ur3/ur3_moveit_control/test/test_pose_generation.py \
  ur3/ur3_moveit_control/test/test_d435i_description.py \
  2>&1 | tee "${LOG_DIR}/wp0_pytest.log"

/usr/bin/python3 "${SCRIPT_DIR}/demo_wp0_reason_codes.py"

PILOT_DIGEST_AFTER="$(pilot_digest)"
END_EPOCH="$(date +%s)"

/usr/bin/python3 - \
  "${START_EPOCH}" "${END_EPOCH}" \
  "${PILOT_DIGEST_BEFORE}" "${PILOT_DIGEST_AFTER}" \
  "${SCRIPT_DIR}/core_check_result.json" <<'PY'
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

start, end = int(sys.argv[1]), int(sys.argv[2])
before, after, output = sys.argv[3], sys.argv[4], pathlib.Path(sys.argv[5])
root = output.parent
pytest_text = (root / "logs" / "wp0_pytest.log").read_text(encoding="utf-8")
build_text = (root / "logs" / "wp0_build.log").read_text(encoding="utf-8")
match = re.search(r"(\d+) passed(?:, (\d+) warnings?)? in ([0-9.]+)s", pytest_text)
summary = next(
    (line.strip() for line in reversed(build_text.splitlines()) if line.startswith("Summary:")),
    "",
)
payload = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "command": "./protocol/run_wp0_checks.sh",
    "environment": {
        "python": "/usr/bin/python3",
        "python_no_user_site": True,
        "pytest_plugin_autoload": False,
    },
    "build": {"passed": "packages finished" in summary, "summary": summary},
    "interface_import": {"passed": True, "type": "ObjectObservation"},
    "pytest": {
        "passed": bool(match),
        "passed_count": int(match.group(1)) if match else None,
        "warning_count": int(match.group(2) or 0) if match else None,
        "duration_s": float(match.group(3)) if match else None,
    },
    "pilot_lock": {
        "before_sha256": before,
        "after_sha256": after,
        "unchanged": before == after,
    },
    "elapsed_s": end - start,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not payload["build"]["passed"] or not payload["pytest"]["passed"] or before != after:
    raise SystemExit(1)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
