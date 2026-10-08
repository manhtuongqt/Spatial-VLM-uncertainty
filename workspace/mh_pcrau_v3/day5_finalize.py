"""Finalize the Day-5 append-only gate ledger and hash manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DAY5 = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05"
LEDGER_DIR = ROOT / "ketqua1/00_quan_tri_khoa/ngay_05"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Append-only output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    g1 = ROOT / "ketqua1/00_quan_tri_khoa/ngay_03/G1_DECISION_REVISION_V3.json"
    day4 = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_04/DAY4_DECISION.json"
    g2_r1 = DAY5 / "G2_DECISION.json"
    g2_r2 = DAY5 / "G2_DECISION_REVISION_V2.json"
    cache_qc = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_QC.json"
    overfit_r2 = DAY5 / "revision_02/MINIBATCH_OVERFIT.json"
    decision = json.loads(g2_r2.read_text(encoding="utf-8"))
    if decision["outcome"] != "G2_PASS" or not decision["day6_s1a_authorized"]:
        raise RuntimeError("Cannot finalize Day 5 without G2_PASS")
    if json.loads(g2_r1.read_text(encoding="utf-8"))["outcome"] != "G2_STOP":
        raise RuntimeError("Attempt-01 negative evidence was not preserved")
    if json.loads(cache_qc.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Cache QC did not pass")
    if json.loads(overfit_r2.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Revision-02 overfit did not pass")

    source_paths = [
        ROOT / "workspace/mh_pcrau_v3/day5_prepare.py",
        ROOT / "workspace/mh_pcrau_v3/day5_cache.py",
        ROOT / "workspace/mh_pcrau_v3/day5_train.py",
        ROOT / "workspace/mh_pcrau_v3/day5_prepare_r2.py",
        ROOT / "workspace/mh_pcrau_v3/day5_train_r2.py",
        ROOT / "workspace/mh_pcrau_v3/plot_day5.py",
        ROOT / "workspace/mh_pcrau_v3/multihead_v3.py",
        ROOT / "workspace/mh_pcrau_v3/loss_v3.py",
    ]
    source_manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Day-5 source identity after evidence generation",
        "files": [record(path) for path in source_paths],
    }
    source_path = DAY5 / "SOURCE_MANIFEST.json"
    write_new(source_path, source_manifest)

    artifact_paths = [
        ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK.json",
        ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK_R2.json",
        ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_INPUT_MANIFEST.jsonl",
        ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/SUPERVISION_STORE.jsonl",
        ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/FEATURE_CACHE_MANIFEST.json",
        cache_qc,
        ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_EXTRACTION_TIMING.csv",
        ROOT / "ketqua1/06_ham_mat_mat/ngay_05/GRADIENT_SCALE.csv",
        ROOT / "ketqua1/06_ham_mat_mat/ngay_05/LOSS_WEIGHT_DECISION.json",
        DAY5 / "MINIBATCH_TRAIN_LOG.csv",
        DAY5 / "MINIBATCH_OVERFIT.json",
        DAY5 / "ATTEMPT_01_VARIANCE_COLLAPSE.md",
        g2_r1,
        DAY5 / "revision_02/MINIBATCH_TRAIN_LOG.csv",
        overfit_r2,
        g2_r2,
        DAY5 / "G2_LEARNING_CURVES.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_05/G2_LEARNING_CURVES.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_05/G2_VARIANCE_ATTEMPT_COMPARISON.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_05/GRADIENT_SCALE_CANDIDATES.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_05/CACHE_EXTRACTION_TIMING.png",
        ROOT / "ketqua1/09_danh_gia/figures/ngay_05/FIGURE_INDEX.json",
        ROOT / "ketqua1/09_danh_gia/tables/ngay_05/DAY5_OBSERVED_SUMMARY.csv",
        DAY5 / "KET_QUA_NGAY_05.md",
        source_path,
    ]
    shards = sorted((ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/cache").glob("*.pt"))
    artifact_manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": "G2_PASS",
        "artifact_count_excluding_cache_shards": len(artifact_paths),
        "cache_shard_count": len(shards),
        "files": [record(path) for path in artifact_paths],
        "cache_shards": [record(path) for path in shards],
    }
    artifact_path = DAY5 / "ARTIFACT_MANIFEST.json"
    write_new(artifact_path, artifact_manifest)

    ledger = {
        "schema_version": "1.0",
        "date_local": "2026-09-24",
        "input_gates": [
            {**record(g1), "outcome": "G1_PASS"},
            {**record(day4), "outcome": "DAY4_IMPLEMENTATION_SMOKE_PASS"},
        ],
        "attempt_history": [
            {**record(g2_r1), "outcome": "G2_STOP", "reason": "log-variance reached -8 clamp"},
            {**record(g2_r2), "outcome": "G2_PASS", "passed_at_step": 90},
        ],
        "effective_gate": "G2_PASS",
        "authorized_next_work": [
            "Day 6 S1a development-only train/Val manifest and run lock",
            "Day 6 S1a trunk plus supported non-confidence heads",
        ],
        "must_remain_masked": ["reasoning-depth without certified labels", "uncertainty-source without certified labels"],
        "deferred": ["confidence training until OOF/S1b"],
        "still_sealed": ["Calibration", "IID Test", "OOD Test", "robot scientific evaluation"],
        "evidence": [record(cache_qc), record(overfit_r2), record(artifact_path)],
    }
    ledger_path = LEDGER_DIR / "GATE_LEDGER_DAY_05.json"
    write_new(ledger_path, ledger)

    delivery = {
        "schema_version": "1.0",
        "date_local": "2026-09-24",
        "status": "DAY5_COMPLETE_G2_PASS",
        "report": record(DAY5 / "KET_QUA_NGAY_05.md"),
        "decision": record(g2_r2),
        "source_manifest": record(source_path),
        "artifact_manifest": record(artifact_path),
        "gate_ledger": record(ledger_path),
        "figure_index": record(ROOT / "ketqua1/09_danh_gia/figures/ngay_05/FIGURE_INDEX.json"),
    }
    delivery_path = DAY5 / "DAY5_DELIVERY_MANIFEST.json"
    write_new(delivery_path, delivery)
    print(json.dumps({"status": delivery["status"], "source_files": len(source_paths), "artifacts": len(artifact_paths), "cache_shards": len(shards)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
