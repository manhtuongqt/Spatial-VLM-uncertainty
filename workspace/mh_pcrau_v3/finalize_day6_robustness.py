#!/usr/bin/env python3
"""Create and verify an append-only delivery manifest for Day-6 amendments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/ROBUSTNESS_AMENDMENTS_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError("Robustness manifest is append-only")
    patterns = [
        "ketqua1/07_huan_luyen/ngay_06/KET_QUA_NGAY_06_ROBUSTNESS_AMENDMENTS.md",
        "ketqua1/07_huan_luyen/ngay_06/amendment_0*/*",
        "ketqua1/09_danh_gia/metrics/ngay_06/amendment_0*/*",
        "ketqua1/09_danh_gia/tables/ngay_06/amendment_0*/*",
        "ketqua1/09_danh_gia/figures/ngay_06/amendment_0*/*",
        "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view/*.json",
        "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view/*.jsonl",
        "ketqua1/03_backbone_h_spatial/ngay_06/amendment_02_paired_view/*.csv",
        "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned/*.json",
        "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned/*.jsonl",
        "ketqua1/03_backbone_h_spatial/ngay_06/amendment_03_prompt_aligned/*.csv",
        "protocol/MH_PCRAU_V3_*SHORTCUT*.json",
        "protocol/MH_PCRAU_V3_DAY6_PAIRED_VIEW*.json",
        "protocol/MH_PCRAU_V3_DAY6_PROMPT_ALIGNED*.json",
        "protocol/MH_PCRAU_V3_DAY6_ROBUST_CANDIDATE*.json",
        "workspace/mh_pcrau_v3/day6_*shortcut*.py",
        "workspace/mh_pcrau_v3/day6_paired_view*.py",
        "workspace/mh_pcrau_v3/day6_prompt_aligned*.py",
        "workspace/mh_pcrau_v3/day6_robust_candidate*.py",
        "workspace/mh_pcrau_v3/plot_day6_robust_candidate.py",
        "workspace/mh_pcrau_v3/checkpoints/day06_robust_candidate*/s1a_robust_fixed_epoch54.pt",
    ]
    paths = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    if len(paths) < 35: raise RuntimeError(f"Unexpectedly small amendment artifact set: {len(paths)}")
    records = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
               for path in sorted(paths)]
    payload = {
        "schema_version": "1.0", "status": "COMPLETE",
        "outcome": "S1A_ENGINEERING_COMPLETE_GENERALIZATION_HOLD",
        "robust_candidate": "ROBUST_CANDIDATE_READY_AWAITING_FRESH_VAL",
        "day7_oof_s1b_authorized": False,
        "artifact_count": len(records), "artifacts": records,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
    failures = [row["path"] for row in loaded["artifacts"] if sha256(ROOT / row["path"]) != row["sha256"]]
    if failures: raise RuntimeError(f"Manifest verification failed: {failures}")
    print(json.dumps({"status": "PASS", "artifacts": len(records), "manifest_sha256": sha256(OUTPUT)}))


if __name__ == "__main__": main()
