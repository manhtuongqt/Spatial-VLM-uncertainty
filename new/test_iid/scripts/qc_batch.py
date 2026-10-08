#!/usr/bin/python3
"""Run the qualified observed RGB-D QC on a Test-IID capture batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "old" / "protocol"))
import dataset_v2_1_development_batch_qc as qualified  # noqa: E402


PROTOCOL_ID = "roborefer_dataset_v2_1_test_iid_capture_200"
DECISION = "GO_TEST_IID_CAPTURE_CANARY"
qualified.PROTOCOL_ID = PROTOCOL_ID
qualified.DECISION = DECISION


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    dataset = ROOT / "new/test_iid/dataset"
    report = qualified.run_qc(
        ROOT,
        dataset,
        ROOT / "new/test_iid/protocol/test_iid_capture_plan.json",
        ROOT / "new/test_iid/protocol/execution_lock.json",
        ROOT / "new/test_iid/protocol/test_iid_manifest.json",
        args.batch_id,
    )
    report["decision"] = (
        "GO_NEXT_TEST_IID_BATCH" if report["passed"] and args.batch_id != "batch_002"
        else "GO_TEST_IID_MATERIALIZATION" if report["passed"]
        else "FIX_CURRENT_TEST_IID_BATCH"
    )
    report["test_inference_performed"] = False
    path = dataset / f"report_assets/checkpoints/batch_qc/{args.batch_id}.json"
    qualified.atomic_json(path, report)
    print(json.dumps({
        "batch": args.batch_id, "passed": report["passed"],
        "decision": report["decision"], "captures": report["valid_capture_count"],
        "relation_checks": report["relation_checks"], "errors": report["errors"],
    }, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
