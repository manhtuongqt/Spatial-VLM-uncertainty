"""Verify and finalize Day-6 S1a evidence, hashes and gate ledger."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DAY6 = ROOT / "ketqua1/07_huan_luyen/ngay_06"
LEDGER_DIR = ROOT / "ketqua1/00_quan_tri_khoa/ngay_06"
CACHE_DIR = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Append-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    decision_path = DAY6 / "S1A_DECISION.json"
    decision = json.loads(decision_path.read_text())
    if decision["outcome"] != "S1A_COMPLETE" or not decision["day7_oof_s1b_authorized"]:
        raise RuntimeError("Day 6 cannot be finalized without S1A_COMPLETE")
    if not all(decision["checks"].values()):
        raise RuntimeError("At least one locked S1a completion check failed")
    cache_qc = json.loads((CACHE_DIR / "S1A_CACHE_QC.json").read_text())
    if cache_qc["status"] != "PASS":
        raise RuntimeError("Day-6 cache QC is not PASS")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONNOUSERSITE", None)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v", "-s", "workspace/mh_pcrau_v3/tests", "-p", "test_*.py"],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    tests_pass = tests.returncode == 0 and "Ran 40 tests" in tests.stdout and "OK" in tests.stdout
    regression = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if tests_pass else "FAIL", "tests_passed": 40 if tests_pass else None,
        "interpreter": sys.executable,
        "environment_note": "Day-4-compatible dependency environment; jsonschema is available from user site",
        "output_tail": tests.stdout[-5000:],
    }
    regression_path = DAY6 / "DAY6_REGRESSION_TEST_REPORT.json"
    write_new(regression_path, regression)
    if not tests_pass:
        raise RuntimeError("Workspace regression tests failed")

    source_paths = [
        ROOT / "workspace/mh_pcrau_v3/day6_prepare.py",
        ROOT / "workspace/mh_pcrau_v3/day6_cache.py",
        ROOT / "workspace/mh_pcrau_v3/day6_prepare_r2.py",
        ROOT / "workspace/mh_pcrau_v3/day6_train.py",
        ROOT / "workspace/mh_pcrau_v3/plot_day6.py",
        ROOT / "workspace/mh_pcrau_v3/day6_finalize.py",
        ROOT / "workspace/mh_pcrau_v3/multihead_v3.py",
        ROOT / "workspace/mh_pcrau_v3/loss_v3.py",
    ]
    source_manifest = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Day-6 source identity after S1a completion",
        "files": [record(path) for path in source_paths],
    }
    source_manifest_path = DAY6 / "SOURCE_MANIFEST.json"
    write_new(source_manifest_path, source_manifest)

    artifact_paths = [
        ROOT / "protocol/MH_PCRAU_V3_DAY6_S1A_RUN_LOCK.json",
        ROOT / "protocol/MH_PCRAU_V3_DAY6_S1A_RUN_LOCK_R2.json",
        DAY6 / "S1A_INPUT_MANIFEST.jsonl",
        DAY6 / "S1A_SUPERVISION_STORE.jsonl",
        DAY6 / "S1A_SPLIT_AUDIT.json",
        CACHE_DIR / "S1A_FEATURE_CACHE_MANIFEST.json",
        CACHE_DIR / "S1A_CACHE_QC.json",
        CACHE_DIR / "S1A_CACHE_EXTRACTION_TIMING.csv",
        DAY6 / "DAY6_ATTEMPT_01_PREFLIGHT_FAILURE.md",
        DAY6 / "S1A_TRAIN_LOG.jsonl",
        DAY6 / "S1A_EPOCH_METRICS.csv",
        DAY6 / "S1A_CHECKPOINT_MANIFEST.json",
        DAY6 / "S1A_RESOURCE_REPORT.json",
        DAY6 / "BACKBONE_FREEZE_AUDIT.json",
        decision_path,
        DAY6 / "S1A_LEARNING_CURVES.png",
        DAY6 / "KET_QUA_NGAY_06.md",
        regression_path,
        source_manifest_path,
        ROOT / "ketqua1/09_danh_gia/figures/ngay_06/S1A_LEARNING_CURVES.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_06/S1A_PER_CLASS_RECALL.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_06/S1A_RESOURCE_AND_GRADIENT.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_06/S1A_HEAD_SUPPORT.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_06/FIGURE_INDEX.json",
        ROOT / "ketqua1/09_danh_gia/tables/ngay_06/S1A_BEST_CHECKPOINT_SUMMARY.csv",
        ROOT / "ketqua1/09_danh_gia/metrics/ngay_06/S1A_DEVELOPMENT_METRICS.json",
    ]
    checkpoint_manifest = json.loads((DAY6 / "S1A_CHECKPOINT_MANIFEST.json").read_text())
    checkpoint_paths = [ROOT / item["path"] for item in checkpoint_manifest["files"]]
    cache_manifest = json.loads((CACHE_DIR / "S1A_FEATURE_CACHE_MANIFEST.json").read_text())
    cache_paths = sorted({ROOT / item["feature_path"] for item in cache_manifest["records"]})
    artifact_manifest = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": "S1A_COMPLETE",
        "artifact_count_excluding_feature_shards_and_checkpoints": len(artifact_paths),
        "feature_shard_count": len(cache_paths), "checkpoint_count": len(checkpoint_paths),
        "files": [record(path) for path in artifact_paths],
        "feature_shards": [record(path) for path in cache_paths],
        "checkpoints": [record(path) for path in checkpoint_paths],
    }
    artifact_manifest_path = DAY6 / "ARTIFACT_MANIFEST.json"
    write_new(artifact_manifest_path, artifact_manifest)

    g2 = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/G2_DECISION_REVISION_V2.json"
    ledger = {
        "schema_version": "1.0", "date_local": "2026-09-24",
        "input_gate": {**record(g2), "outcome": "G2_PASS"},
        "day6": {**record(decision_path), "outcome": "S1A_COMPLETE"},
        "attempt_history": [
            {"attempt": 1, "status": "PREFLIGHT_STOP", "optimizer_steps": 0, "evidence": record(DAY6 / "DAY6_ATTEMPT_01_PREFLIGHT_FAILURE.md")},
            {"attempt": 2, "status": "S1A_COMPLETE", "best_epoch": 54, "last_epoch": 79, "evidence": record(decision_path)},
        ],
        "authorized_next_work": ["Day 7 family-disjoint OOF generation", "Day 7 S1b confidence-only training from OOF correctness"],
        "must_remain_masked": ["reasoning-depth without certified labels", "uncertainty-source without certified labels"],
        "still_frozen_until_oof": ["confidence head"],
        "g3": "NOT_OPENED",
        "still_sealed": ["Calibration", "IID Test", "OOD Test", "robot scientific evaluation"],
        "artifact_manifest": record(artifact_manifest_path),
    }
    ledger_path = LEDGER_DIR / "GATE_LEDGER_DAY_06.json"
    write_new(ledger_path, ledger)

    delivery = {
        "schema_version": "1.0", "date_local": "2026-09-24",
        "status": "DAY6_COMPLETE_S1A_COMPLETE_VERIFIED",
        "report": record(DAY6 / "KET_QUA_NGAY_06.md"),
        "decision": record(decision_path),
        "regression": record(regression_path),
        "source_manifest": record(source_manifest_path),
        "artifact_manifest": record(artifact_manifest_path),
        "gate_ledger": record(ledger_path),
        "figure_index": record(ROOT / "ketqua1/09_danh_gia/figures/ngay_06/FIGURE_INDEX.json"),
    }
    delivery_path = DAY6 / "DAY6_DELIVERY_MANIFEST.json"
    write_new(delivery_path, delivery)
    print(json.dumps({
        "status": delivery["status"], "regression_tests": 40,
        "source_files": len(source_paths), "artifacts": len(artifact_paths),
        "feature_shards": len(cache_paths), "checkpoints": len(checkpoint_paths),
    }))


if __name__ == "__main__":
    main()
