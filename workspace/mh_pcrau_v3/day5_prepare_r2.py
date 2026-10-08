"""Freeze the variance-stability remediation before Day-5 attempt 02."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK1 = ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK.json"
LOCK2 = ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK_R2.json"
CACHE_QC = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05/CACHE_QC.json"
OVERFIT1 = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/MINIBATCH_OVERFIT.json"
DECISION1 = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/G2_DECISION.json"
WEIGHTS1 = ROOT / "ketqua1/06_ham_mat_mat/ngay_05/LOSS_WEIGHT_DECISION.json"
DIAGNOSIS = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_05/ATTEMPT_01_VARIANCE_COLLAPSE.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def main() -> None:
    if LOCK2.exists():
        raise FileExistsError(LOCK2)
    lock1 = json.loads(LOCK1.read_text())
    overfit1 = json.loads(OVERFIT1.read_text())
    decision1 = json.loads(DECISION1.read_text())
    weights1 = json.loads(WEIGHTS1.read_text())
    if overfit1["status"] != "FAIL" or decision1["outcome"] != "G2_STOP":
        raise RuntimeError("Attempt-01 negative evidence not present")
    if overfit1["final"]["log_variance_min"] != -8.0:
        raise RuntimeError("Unexpected remediation cause")
    if weights1["selected_candidate"] != "W2_SPA050":
        raise RuntimeError("Loss-weight identity drift")
    lock2 = {
        "schema_version": "1.0", "method_id": "MH-PCRA-U-v3",
        "status": "FROZEN_BEFORE_DAY5_ATTEMPT_02",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": lock1["seed"],
        "inherited_lock": reference(LOCK1),
        "prerequisites": [reference(CACHE_QC), reference(OVERFIT1), reference(DECISION1),
                          reference(WEIGHTS1), reference(DIAGNOSIS)],
        "data_and_features_unchanged": True,
        "mini_batch_sample_ids": lock1["selection"]["mini_batch_sample_ids"],
        "selected_loss_weights": weights1["selected_weights"],
        "optimizer_revision": {
            "optimizer": "AdamW", "main_learning_rate": 0.0005,
            "log_variance_head_learning_rate": 0.000005,
            "weight_decay": 0.0, "maximum_gradient_norm": 5.0,
            "maximum_steps": 1200, "minimum_steps": 50,
            "evaluation_interval": 10, "full_batch": 16, "dropout": 0.10
        },
        "pass_thresholds_unchanged": lock1["mini_overfit"]["pass_thresholds"],
        "remediation": "Preserve Gaussian NLL and architecture; reduce global LR 4x, variance-head LR 100x vs main, clip global gradient norm. Restart from identical seed; no checkpoint continuation.",
        "access_boundary": "Development-only rerun. No Val/Calibration/Test/robot access."
    }
    LOCK2.write_text(json.dumps(lock2, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": lock2["status"], "sha256": sha256(LOCK2)}))


if __name__ == "__main__":
    main()
