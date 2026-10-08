#!/usr/bin/env python3
"""Run locked pilot QC with one narrowly scoped, auditable validator correction.

The capture-time validator is never modified.  Amendment 01 only handles the
inverse-relation counterfactual of a `too_small_or_out_of_view` family: the
original invisible target becomes the counterfactual anchor, so invisibility of
that anchor is expected evidence rather than a capture defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import dataset_v2_pilot_validate as locked


LOCKED_VALIDATOR_SHA256 = "e48c825bb9b9e5307b1066456b549c0444b12edf846a73ed2dd34ade0c59734c"
AMENDMENT_PATH = Path(__file__).with_name("dataset_v2_pilot_validator_amendment_01.json")
_locked_mask_and_ontology_qc = locked.mask_and_ontology_qc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_and_ontology_qc_amended(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    try:
        return _locked_mask_and_ontology_qc(root, record)
    except locked.PilotValidationError as exc:
        expected_error = f"INSUFFICIENT anchor not visible: {record['sample_id']}"
        evaluator = record["evaluator_only"]
        uncertainty = evaluator["uncertainty_label"]
        if (
            str(exc) != expected_error
            or record["variant"] != "relation_counterfactual"
            or uncertainty["state"] != "INSUFFICIENT_EVIDENCE"
            or uncertainty["state_submode"] != "too_small_or_out_of_view"
        ):
            raise

        masks = evaluator["masks"]
        arrays = {
            name: locked.read_mask(root, masks[name])
            for name in ("target", "target_interior", "graspable", "reachable", "valid_depth")
        }
        anchor_arrays = [locked.read_mask(root, item) for item in masks["anchor"]]
        target = arrays["target"] > 0
        interior = arrays["target_interior"] > 0
        graspable = arrays["graspable"] > 0
        reachable = arrays["reachable"] > 0
        spatial = evaluator["spatial_label"]
        anchor_pixels = [int(np.count_nonzero(value)) for value in anchor_arrays]

        # The amendment accepts only the exact intervention it documents.  All
        # remaining ontology/action invariants are checked here because the
        # locked function raised immediately before those final checks.
        if (
            len(spatial["valid_target_ids"]) != 1
            or uncertainty["answerable"]
            or uncertainty["expected_intervention"] != "REOBSERVE"
            or not anchor_pixels
            or not any(value == 0 for value in anchor_pixels)
            or int(np.count_nonzero(graspable))
            or int(np.count_nonzero(reachable))
        ):
            raise locked.PilotValidationError(
                f"amendment-01 scope/invariant failed: {record['sample_id']}"
            ) from exc

        labels = cv2.imread(
            str(locked.safe_path(root, evaluator["semantic_instance_labels"]["path"])),
            cv2.IMREAD_UNCHANGED,
        )
        if labels is None or labels.shape != target.shape:
            raise locked.PilotValidationError(
                f"semantic labels unavailable for amended mask QC: {record['sample_id']}"
            ) from exc
        label_ids = evaluator["object_oracle"]["label_ids"]
        candidate_pixel_counts = {
            object_id: int(np.count_nonzero(labels == int(label_ids[object_id])))
            for object_id in spatial["candidate_target_ids"]
            if object_id in label_ids
        }
        return {
            "sample_id": record["sample_id"],
            "state": uncertainty["state"],
            "state_submode": uncertainty["state_submode"],
            "target_pixels": int(np.count_nonzero(target)),
            "interior_pixels": int(np.count_nonzero(interior)),
            "anchor_pixel_counts": anchor_pixels,
            "candidate_pixel_counts": candidate_pixel_counts,
            "graspable_pixels": int(np.count_nonzero(graspable)),
            "reachable_pixels": int(np.count_nonzero(reachable)),
            "valid_depth_fraction": float(
                np.count_nonzero(arrays["valid_depth"]) / arrays["valid_depth"].size
            ),
            "validator_amendment": "AMENDMENT_01_EXPECTED_INVISIBLE_INVERSE_ANCHOR",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    locked_path = Path(locked.__file__).resolve()
    if sha256_file(locked_path) != LOCKED_VALIDATOR_SHA256:
        raise SystemExit("locked validator hash differs; refusing amended QC")
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment["locked_validator_sha256"] != LOCKED_VALIDATOR_SHA256:
        raise SystemExit("amendment declaration does not match locked validator")

    locked.mask_and_ontology_qc = mask_and_ontology_qc_amended
    try:
        report = locked.validate(root, run_replay=not args.skip_replay)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "protocol_id": locked.PROTOCOL_ID,
            "passed": False,
            "decision": "FIX_DATASET_V2_GENERATOR_OR_CAPTURE_PIPELINE_FIRST",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "validator_amendment": amendment,
            "training_performed": False,
        }
        (root / "PILOT_QC_REPORT.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"DATASET_V2_PILOT_QC_FAIL {type(exc).__name__}: {exc}")
        return 2

    amendment_evidence = dict(amendment)
    amendment_evidence["amended_validator_sha256"] = sha256_file(Path(__file__).resolve())
    amendment_evidence["declaration_sha256"] = sha256_file(AMENDMENT_PATH)
    report["validator_amendments"] = [amendment_evidence]
    (root / "PILOT_QC_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (root / "PILOT_QC_REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Validator amendment\n\n"
            "`AMENDMENT_01_EXPECTED_INVISIBLE_INVERSE_ANCHOR` was applied to the "
            "locked validator without modifying it. The amendment accepts only a "
            "relation-counterfactual anchor inherited from the deliberately "
            "too-small/out-of-view target; all other checks remain unchanged.\n"
        )
    print(
        "DATASET_V2_PILOT_QC_PASS "
        f"decision={report['decision']} families={report['family_count']} "
        f"records={report['sample_count']} captures={report['raw_capture_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
