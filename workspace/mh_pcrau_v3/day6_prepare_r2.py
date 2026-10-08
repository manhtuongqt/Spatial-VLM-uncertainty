"""Freeze Day-6 attempt-02 source and deterministic-environment amendment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK1 = ROOT / "protocol/MH_PCRAU_V3_DAY6_S1A_RUN_LOCK.json"
LOCK2 = ROOT / "protocol/MH_PCRAU_V3_DAY6_S1A_RUN_LOCK_R2.json"
CACHE_MANIFEST = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_FEATURE_CACHE_MANIFEST.json"
CACHE_QC = ROOT / "ketqua1/03_backbone_h_spatial/ngay_06/S1A_CACHE_QC.json"
FAILURE = ROOT / "ketqua1/07_huan_luyen/ngay_06/DAY6_ATTEMPT_01_PREFLIGHT_FAILURE.md"
TRAIN = ROOT / "workspace/mh_pcrau_v3/day6_train.py"
MODEL = ROOT / "workspace/mh_pcrau_v3/multihead_v3.py"
LOSS = ROOT / "workspace/mh_pcrau_v3/loss_v3.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    if LOCK2.exists():
        raise FileExistsError("Day-6 attempt-02 lock is append-only")
    base = json.loads(LOCK1.read_text())
    cache_qc = json.loads(CACHE_QC.read_text())
    if base["status"] != "FROZEN_BEFORE_DAY6_CACHE_OR_TRAINING" or cache_qc["status"] != "PASS":
        raise RuntimeError("Attempt-02 prerequisites are not eligible")
    lock = dict(base)
    lock.update({
        "status": "FROZEN_BEFORE_DAY6_TRAIN_ATTEMPT_02",
        "revision": "append_only_r2_after_pre_optimizer_cublas_preflight_stop",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inherited_lock": ref(LOCK1),
        "cache_manifest": ref(CACHE_MANIFEST),
        "cache_qc": ref(CACHE_QC),
        "attempt_01_failure": ref(FAILURE),
        "source_identity": {
            "train_runner": ref(TRAIN),
            "multihead": ref(MODEL),
            "masked_loss": ref(LOSS),
        },
        "deterministic_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TF32": False,
            "deterministic_algorithms": True,
        },
        "amendment_scope": "Environment/source identity only; data, cache, seed, architecture, loss weights, optimizer, selection and completion checks unchanged",
    })
    LOCK2.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "lock_sha256": sha256(LOCK2), "train_source_sha256": sha256(TRAIN)}))


if __name__ == "__main__":
    main()
