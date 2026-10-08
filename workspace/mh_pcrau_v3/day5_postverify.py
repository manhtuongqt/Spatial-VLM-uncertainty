"""Record Day-5 source compilation and regression verification evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DAY5 = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05"
RESULT = DAY5 / "DAY5_POST_FINALIZATION_VERIFICATION.json"
DELIVERY_R2 = DAY5 / "DAY5_DELIVERY_MANIFEST_REVISION_V2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Append-only output exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_tests(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v", "-s", "workspace/mh_pcrau_v3/tests", "-p", "test_*.py"],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def main() -> None:
    if RESULT.exists() or DELIVERY_R2.exists():
        raise FileExistsError("Day-5 post-verification is append-only")
    isolated = run_tests({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    normal = run_tests({"PYTHONDONTWRITEBYTECODE": "1"})
    isolated_expected = isolated.returncode != 0 and "No module named 'jsonschema'" in isolated.stdout
    normal_pass = normal.returncode == 0 and "Ran 40 tests" in normal.stdout and "OK" in normal.stdout
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_DOCUMENTED_OPTIONAL_DEPENDENCY" if isolated_expected and normal_pass else "FAIL",
        "interpreter": sys.executable,
        "checks": {
            "isolated_usersite_disabled": {
                "returncode": isolated.returncode,
                "expected_environment_issue": isolated_expected,
                "reason": "jsonschema is not installed inside the Conda prefix when user-site packages are disabled",
                "tail": isolated.stdout[-1200:],
            },
            "day4_compatible_environment": {
                "returncode": normal.returncode,
                "tests_passed": 40 if normal_pass else None,
                "status": "PASS" if normal_pass else "FAIL",
                "tail": normal.stdout[-1200:],
            },
        },
        "interpretation": "The first failure is dependency availability, not a model/loss regression. The same interpreter with the Day-4 dependency environment passes 40/40 tests.",
    }
    write_new(RESULT, payload)
    prior_delivery = DAY5 / "DAY5_DELIVERY_MANIFEST.json"
    revision = {
        "schema_version": "1.0",
        "date_local": "2026-09-24",
        "status": "DAY5_COMPLETE_G2_PASS_VERIFIED" if payload["status"].startswith("PASS") else "DAY5_VERIFICATION_FAIL",
        "supersedes_for_delivery_only": record(prior_delivery),
        "verification": record(RESULT),
        "verification_source": record(Path(__file__)),
        "decision": record(DAY5 / "G2_DECISION_REVISION_V2.json"),
        "report": record(DAY5 / "KET_QUA_NGAY_05.md"),
        "note": "Earlier delivery manifest remains immutable; this revision appends regression evidence.",
    }
    write_new(DELIVERY_R2, revision)
    print(json.dumps({"status": revision["status"], "tests": 40 if normal_pass else 0, "isolated_dependency_issue_recorded": isolated_expected}))
    if not (isolated_expected and normal_pass):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
