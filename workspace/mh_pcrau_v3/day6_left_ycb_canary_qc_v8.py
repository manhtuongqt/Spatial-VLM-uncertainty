#!/usr/bin/env python3
"""Measure v8 semantic masks and lock one passing geometry per pair."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import cv2, yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v8"
CAPTURE = OUT / "capture_attempt_01"

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main() -> None:
    ann=yaml.safe_load((OUT/"annotations.yaml").read_text())["scenes"]
    rows=[json.loads(x) for x in (CAPTURE/"input_manifest.jsonl").read_text().splitlines() if x]
    obs=defaultdict(list)
    for row in rows:
        sid=row["scene_id"]; a=ann[sid]; c=a["projection_calibration"]
        p=CAPTURE/sid/"evaluator/semantic_labels.png"; labels=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        if labels is None: raise RuntimeError(f"missing {p}")
        n=int((labels==int(a["target_label"])).sum()); o=int((labels==int(a["occluder_labels"][0])).sum())
        obs[(c["box"],c["fruit"])].append({"scene_id":sid,"signed_lateral_offset_m":float(c["signed_lateral_offset_m"]),
          "forward_separation_m":float(c["forward_separation_m"]),"target_visible_pixels":n,"occluder_visible_pixels":o,
          "pass_1_to_119":1<=n<120,"semantic_labels_sha256":sha256(p)})
    selected={}; results=[]
    for (box,fruit), vals in sorted(obs.items()):
        valid=[v for v in vals if v["pass_1_to_119"]]
        choice=min(valid,key=lambda v:(abs(v["target_visible_pixels"]-60),abs(v["signed_lateral_offset_m"]))) if valid else None
        if choice: selected.setdefault(box,{})[fruit]=choice
        results.append({"box":box,"fruit":fruit,"observations":len(vals),"minimum_pixels":min(v["target_visible_pixels"] for v in vals),
                        "maximum_pixels":max(v["target_visible_pixels"] for v in vals),"valid_count":len(valid),"selected":choice})
    passed=len(results)==15 and all(r["valid_count"] for r in results)
    qc={"schema_version":1,"status":"PASS" if passed else "FAIL_TARGETED_RESCAN_REQUIRED","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "world_revision":"v5_contact_exact","lemon_excluded":True,"captured_scenes":len(rows),"pair_count":len(results),
        "passing_pair_count":sum(bool(r["valid_count"]) for r in results),"pass_rule":"all 15 pairs have 1<=target_visible_pixels<120",
        "pair_results":results,"capture_manifest_sha256":sha256(CAPTURE/"capture_manifest.json")}
    q=OUT/"CANARY_QC.json"; q.write_text(json.dumps(qc,indent=2)+"\n")
    (OUT/"SELECTED_PAIR_GEOMETRY.json").write_text(json.dumps({"schema_version":1,"status":"LOCKED" if passed else "INCOMPLETE",
        "pairs":selected,"canary_qc_sha256":sha256(q)},indent=2)+"\n")
    print(json.dumps({"status":qc["status"],"passing_pairs":qc["passing_pair_count"],"pairs":len(results)},indent=2))
    if not passed: raise SystemExit(2)

if __name__=="__main__": main()
