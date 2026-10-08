#!/usr/bin/env python3
"""Match every derived RefSpatial record to the pinned official Simulator metadata."""

import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path

from refspatial_build_supplement import json_array


ROOT = Path(__file__).resolve().parents[1]
REVISION = "519a6fd43aee2d0ed1366776e50f9456d212f94f"
DATASETS = (
    "RefSpatial-Tabletop-Large",
    "RefSpatial-Tabletop",
    "RefSpatial-Handle-Occlusion",
    "RefSpatial-Cup-Occlusion",
    "RefSpatial-WristLike-Cup-Fruit-Occlusion",
)
RAW = ROOT / "datasets/refspatial_supplement_v1_20260907/source/Simulator/metadata.json"
RAW_PROVENANCE = RAW.with_name(RAW.name + ".provenance.json")
OUT = ROOT / "results/spatial_vlm_refspatial_v1/wp1_source_audit/pinned_source_lineage.json"


def fingerprint(record):
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def scene_id(record):
    images = record.get("image", [])
    return Path(images[0]).stem if len(images) == 1 else None


def main():
    provenance = json.loads(RAW_PROVENANCE.read_text())
    if provenance.get("revision") != REVISION or not provenance.get("sha256_verified_against_lfs"):
        raise RuntimeError("Official raw metadata is not verified at the required revision")

    wanted = {}
    dataset_stats = {}
    for name in DATASETS:
        path = ROOT / "datasets" / name / "metadata.jsonl"
        total = 0
        fingerprints = set()
        scenes = set()
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                fp = fingerprint(record)
                total += 1
                fingerprints.add(fp)
                sid = scene_id(record)
                if sid:
                    scenes.add(sid)
                entry = wanted.setdefault(fp, {"datasets": set(), "id": record.get("id"), "scene_id": sid})
                entry["datasets"].add(name)
        dataset_stats[name] = {
            "derived_records": total,
            "derived_unique_record_fingerprints": len(fingerprints),
            "derived_scenes": len(scenes),
            "fingerprints": fingerprints,
        }

    matched = set()
    raw_fingerprints = set()
    raw_scenes = set()
    raw_records = 0
    raw_duplicate_fingerprints = 0
    for raw_records, record in enumerate(json_array(RAW), 1):
        fp = fingerprint(record)
        if fp in raw_fingerprints:
            raw_duplicate_fingerprints += 1
        else:
            raw_fingerprints.add(fp)
        sid = scene_id(record)
        if sid:
            raw_scenes.add(sid)
        if fp in wanted:
            matched.add(fp)
        if raw_records % 25000 == 0:
            print(f"raw_records={raw_records} matched={len(matched)}/{len(wanted)}", flush=True)

    per_dataset = {}
    for name, stats in dataset_stats.items():
        expected = stats.pop("fingerprints")
        missing = expected - matched
        per_dataset[name] = {
            **stats,
            "matched_unique_record_fingerprints": len(expected & matched),
            "missing_unique_record_fingerprints": len(missing),
            "all_records_exactly_in_pinned_raw": not missing,
            "missing_examples": [
                {"fingerprint": fp, "id": wanted[fp]["id"], "scene_id": wanted[fp]["scene_id"]}
                for fp in sorted(missing)[:20]
            ],
        }

    missing_all = set(wanted) - matched
    result = {
        "status": "PINNED_METADATA_LINEAGE_PASS" if not missing_all else "PINNED_METADATA_LINEAGE_FAIL",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "official_source": {
            "repository": provenance["repo"],
            "revision": provenance["revision"],
            "path": provenance["path"],
            "bytes": provenance["bytes"],
            "sha256": provenance["sha256"],
            "official_lfs_sha256": provenance["official_lfs_sha256"],
            "sha256_verified_against_lfs": provenance["sha256_verified_against_lfs"],
        },
        "official_raw_scan": {
            "records": raw_records,
            "unique_record_fingerprints": len(raw_fingerprints),
            "duplicate_record_fingerprints": raw_duplicate_fingerprints,
            "scenes": len(raw_scenes),
        },
        "derived_union": {
            "unique_record_fingerprints": len(wanted),
            "matched_unique_record_fingerprints": len(matched),
            "missing_unique_record_fingerprints": len(missing_all),
            "exact_subset_of_pinned_official_metadata": not missing_all,
        },
        "per_dataset": per_dataset,
        "scope": {
            "metadata_lineage_verified": not missing_all,
            "media_archive_lineage_verified": False,
            "original_filter_code_recovered": False,
            "original_filter_reproduced": False,
            "label_semantics_certified_by_this_check": False,
            "training_eligible": False,
            "b2_open": False,
            "sam2_used": False,
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "official_raw_records": raw_records,
        "derived_unique_records": len(wanted),
        "matched": len(matched),
        "missing": len(missing_all),
        "output": str(OUT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
