"""Fail-closed prerelease exclusion check for *future* v3 Calibration/Test.

This is not an authorization to create or inspect sealed splits. It validates
a candidate manifest only after a later G4 protocol has been approved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .audit_development import ROOT, sha256


REGISTRY = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/HISTORICAL_EXCLUSION_REGISTRY_V4.jsonl"
EXPECTED_REGISTRY_SHA256 = "fc834db6d3db325f98032fde8185088e6291ef7364e86f915f7e61fb0af4ce17"
ALLOWED_SPLITS = ("calibration", "test_iid", "test_ood")
REQUIRED = ("split", "scene_id", "family_id", "seed", "layout_id", "signature",
            "physical_signature", "rgb_path", "rgb_sha256")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check(candidate_path: Path) -> dict:
    if sha256(REGISTRY) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("Historical registry hash drift; abort")
    historical = _read_jsonl(REGISTRY)
    candidate = _read_jsonl(candidate_path)
    errors: list[str] = []
    old = {key: {r.get(key) for r in historical if r.get(key) is not None}
           for key in ("scene_id", "effective_family_id", "layout_id", "signature",
                       "physical_signature", "seed", "rgb_sha256")}
    seen = {key: set() for key in ("scene_id", "family_id", "layout_id", "signature",
                                  "physical_signature", "seed", "rgb_sha256")}
    old_images = []
    for row in historical:
        p = ROOT / row["rgb_path"]
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            errors.append(f"historical RGB unavailable: {row['rgb_path']}")
            continue
        old_images.append((row, im.shape,
                           cv2.resize(im, (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)))
    for i, row in enumerate(candidate):
        prefix = f"candidate[{i}]"
        missing = [key for key in REQUIRED if key not in row or row[key] is None]
        if missing:
            errors.append(f"{prefix}: missing {missing}")
            continue
        split = row["split"]
        if split not in ALLOWED_SPLITS:
            errors.append(f"{prefix}: invalid split")
            continue
        ns = f"mh_v3_{split}/"
        if not row["family_id"].startswith(ns) or not row["scene_id"].startswith(f"mh_v3_{split}_"):
            errors.append(f"{prefix}: family/scene namespace mismatch")
        seed = row["seed"]
        if type(seed) is not int or not (2**32 <= seed < 2**63):
            errors.append(f"{prefix}: seed must be in preregistered 64-bit high namespace")
        for key in seen:
            value = row[key]
            if value in seen[key]:
                errors.append(f"{prefix}: within-candidate duplicate {key}")
            seen[key].add(value)
        for key, old_key in (("scene_id", "scene_id"), ("family_id", "effective_family_id"),
                             ("layout_id", "layout_id"), ("signature", "signature"),
                             ("physical_signature", "physical_signature"), ("seed", "seed"),
                             ("rgb_sha256", "rgb_sha256")):
            if row[key] in old[old_key]:
                errors.append(f"{prefix}: historical {key} collision")
        rgb = ROOT / row["rgb_path"]
        try:
            rgb.resolve().relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"{prefix}: RGB path escapes workspace")
            continue
        if not rgb.is_file() or sha256(rgb) != row["rgb_sha256"]:
            errors.append(f"{prefix}: RGB missing or hash mismatch")
            continue
        image = cv2.imread(str(rgb), cv2.IMREAD_GRAYSCALE)
        if image is None:
            errors.append(f"{prefix}: RGB cannot decode")
            continue
        thumb = cv2.resize(image, (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
        for old_row, shape, old_thumb in old_images:
            if image.shape != shape or float(np.mean(np.abs(thumb - old_thumb))) >= 0.06:
                continue
            other = cv2.imread(str(ROOT / old_row["rgb_path"]), cv2.IMREAD_GRAYSCALE)
            if other is None:
                errors.append(f"{prefix}: historical RGB disappeared")
                continue
            difference = cv2.absdiff(image, other)
            if float(np.mean(difference)) < 0.05 and float(np.mean(difference > 3)) < 0.002:
                errors.append(f"{prefix}: historical near-RGB duplicate {old_row['scene_id']}")
    return {"status": "PASS" if not errors and candidate else "FAIL_CLOSED",
            "candidate_count": len(candidate), "historical_count": len(historical),
            "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_jsonl", type=Path)
    args = parser.parse_args()
    result = check(args.candidate_jsonl)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
