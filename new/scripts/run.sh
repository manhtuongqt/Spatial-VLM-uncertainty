#!/usr/bin/env bash
set -euo pipefail

NEW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${NEW_ROOT}/.." && pwd)"
PYTHON_BIN="${WORKSPACE_ROOT}/.conda-roborefer/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${NEW_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${WORKSPACE_ROOT}"

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
  echo "usage: new/scripts/run.sh {preflight|test|train|evaluate|calibrate|decide|compare-seeds} [args...]" >&2
  exit 2
fi
shift

case "${command_name}" in
  preflight) exec "${PYTHON_BIN}" "${NEW_ROOT}/scripts/preflight.py" "$@" ;;
  test)
    if "${PYTHON_BIN}" -c 'import pytest' >/dev/null 2>&1; then
      exec "${PYTHON_BIN}" -m pytest "${NEW_ROOT}/tests" "$@"
    fi
    exec "${PYTHON_BIN}" "${NEW_ROOT}/scripts/test_all.py"
    ;;
  train) exec "${PYTHON_BIN}" "${NEW_ROOT}/scripts/train.py" "$@" ;;
  evaluate) exec "${PYTHON_BIN}" "${NEW_ROOT}/scripts/evaluate.py" "$@" ;;
  calibrate) exec "${PYTHON_BIN}" "${NEW_ROOT}/scripts/calibrate.py" "$@" ;;
  decide) exec "${PYTHON_BIN}" "${NEW_ROOT}/scripts/decide.py" "$@" ;;
  compare-seeds) exec "${PYTHON_BIN}" "${NEW_ROOT}/scripts/compare_seeds.py" "$@" ;;
  *) echo "unknown command: ${command_name}" >&2; exit 2 ;;
esac
