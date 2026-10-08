"""Run and materialize the Day-4 implementation smoke evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import tempfile
import time
import unittest

import jsonschema
import torch

from workspace.mh_pcrau_v3.multihead_v3 import (
    ANSWERABILITY_CLASSES, REASONING_DEPTH_CLASSES, RELATION_CLASSES, SOURCE_CLASSES,
    MHMultiHeadConfig, build_seeded_model, to_development_records,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_04"
LOSS_DIR = ROOT / "ketqua1/06_ham_mat_mat/ngay_04"
SCHEMA = ROOT / "ketqua1/05_dau_ra_tong_hop/schema/inference_record.schema.json"
SEED = 24092026


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def run_suite(suite: unittest.TestSuite) -> tuple[list[str], float]:
    ids = [test.id() for test in flatten(suite)]
    start = time.perf_counter()
    result = unittest.TestResult()
    suite.run(result)
    elapsed = time.perf_counter() - start
    if not result.wasSuccessful():
        details = [f"{test.id()}: {trace}" for test, trace in result.failures + result.errors]
        raise RuntimeError("Day-4 tests failed\n" + "\n".join(details))
    if result.testsRun != len(ids):
        raise RuntimeError("Test inventory drift")
    return ids, elapsed


def run_tests() -> tuple[list[str], float]:
    return run_suite(unittest.defaultTestLoader.loadTestsFromName(
        "workspace.mh_pcrau_v3.tests.test_multihead_v3"
    ))


def module_inventory(model) -> list[dict]:
    rows = []
    for name, module in model.named_children():
        rows.append({
            "module": name,
            "parameters": sum(p.numel() for p in module.parameters()),
            "trainable_s1a": sum(p.numel() for p in module.parameters() if p.requires_grad),
        })
    return rows


def loss_truth_table() -> str:
    rows = [
        ("relation", "relation_mask AND class in core-4", "cross_entropy", "valid relation count", "enabled"),
        ("reasoning", "reasoning_mask AND certified depth 0..2", "cross_entropy", "valid certified reasoning count", "enabled but Day-3 support=0"),
        ("spatial_coordinate_and_variance", "spatial_mask AND answerability=FOUND AND normalized target_uv", "diagonal_gaussian_nll", "valid FOUND spatial count", "enabled"),
        ("uncertainty_source", "source_mask AND certified single source in core-5", "cross_entropy", "valid certified source count", "enabled but Day-3 support=0"),
        ("answerability", "answerability_mask AND state in core-4", "cross_entropy", "valid answerability count", "enabled"),
        ("confidence", "OOF correctness label only", "binary_cross_entropy_with_logits", "valid OOF count", "weight=0 and parameters frozen in S1a"),
        ("language_modeling", "S2 authorization only", "not implemented in head-only module", "0", "weight=0 in S1a/S1b"),
    ]
    header = "task,active_condition,loss,denominator,s1a_status\n"
    return header + "".join(
        ",".join('"' + cell.replace('"', '""') + '"' for cell in row) + "\n"
        for row in rows
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    config = MHMultiHeadConfig()
    config.validate()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    test_ids, elapsed = run_tests()
    regression_ids, regression_elapsed = run_suite(unittest.defaultTestLoader.discover(
        str(ROOT / "workspace/mh_pcrau_v3/tests"), pattern="test*.py"
    ))
    model = build_seeded_model(SEED).eval()
    model.configure_trainable("s1a")
    inventory = module_inventory(model)
    total = sum(row["parameters"] for row in inventory)
    trainable = sum(row["trainable_s1a"] for row in inventory)
    h = torch.linspace(-1, 1, 3 * 1536, dtype=torch.float32).reshape(3, 1536)
    with torch.no_grad():
        before = model(h)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    state_bytes = stream.getvalue()
    with tempfile.TemporaryDirectory(prefix="mh_pcrau_v3_day4_") as directory:
        checkpoint = Path(directory) / "smoke_state.pt"
        checkpoint.write_bytes(state_bytes)
        restored = build_seeded_model(0).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        with torch.no_grad():
            after = restored(h)
    output_fields = (
        "relation_logits", "relation_probabilities", "reasoning_logits",
        "reasoning_probabilities", "mu_uv", "log_variance_uv", "source_logits",
        "source_probabilities", "answerability_logits", "answerability_probabilities",
        "confidence_logit", "raw_safe_score", "z_spatial",
    )
    max_difference = max(float((getattr(before, key) - getattr(after, key)).abs().max())
                         for key in output_fields)
    records = to_development_records(
        after, [f"synthetic-{i}" for i in range(3)], model_id="day4-untrained-smoke",
        source_hash=sha256(Path(__file__).parent / "multihead_v3.py"),
    )
    validator = jsonschema.Draft202012Validator(schema)
    schema_errors = [error.message for record in records for error in validator.iter_errors(record)]
    if total != 1_062_421 or trainable != 1_061_908 or max_difference != 0 or schema_errors:
        raise RuntimeError("Final Day-4 invariant failed")
    created = datetime.now(timezone.utc).isoformat()
    source_paths = [
        Path(__file__).parent / "multihead_v3.py",
        Path(__file__).parent / "loss_v3.py",
        Path(__file__).parent / "tests/test_multihead_v3.py",
        Path(__file__), SCHEMA,
    ]
    source_manifest = {
        "schema_version": "1.0", "created_at_utc": created,
        "files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                   "bytes": path.stat().st_size} for path in source_paths],
    }
    model_config = {
        "schema_version": "1.0", "architecture": config.to_dict(), "initialization_seed": SEED,
        "relation_class_order": list(RELATION_CLASSES),
        "reasoning_class_order": list(REASONING_DEPTH_CLASSES),
        "source_class_order": list(SOURCE_CLASSES),
        "answerability_class_order": list(ANSWERABILITY_CLASSES),
        "default_stage": "s1a", "confidence_trainable_s1a": False,
        "language_modeling_loss_s1a": 0, "calibration_status": "NOT_FIT",
        "robot_action_status": "UNAUTHORIZED_BEFORE_CALIBRATION",
    }
    inventory_report = {
        "schema_version": "1.0", "created_at_utc": created,
        "stage": "s1a", "modules": inventory, "total_parameters": total,
        "trainable_parameters": trainable, "frozen_parameters": total - trainable,
        "confidence_parameters": 513,
    }
    head_ids = [name for name in test_ids if ".HeadTests." in name]
    loss_ids = [name for name in test_ids if ".LossTests." in name]
    common = {
        "created_at_utc": created, "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_available_but_not_required_for_cpu_smoke": torch.cuda.is_available(),
        }, "elapsed_seconds_all_16_tests": elapsed,
    }
    head_report = {
        "schema_version": "1.0", "status": "PASS", **common,
        "tests_run": len(head_ids), "tests_passed": len(head_ids),
        "tests": [{"id": name, "status": "PASS"} for name in head_ids],
        "batches_tested": [1, 3, 4, 7], "nonfinite_outputs": 0,
        "serialization": {"temporary_checkpoint_only": True,
                          "state_dict_sha256": hashlib.sha256(state_bytes).hexdigest(),
                          "max_absolute_output_difference": max_difference},
        "development_records_schema_validated": len(records),
        "schema_validation_errors": schema_errors,
    }
    loss_report = {
        "schema_version": "1.0", "status": "PASS", **common,
        "tests_run": len(loss_ids), "tests_passed": len(loss_ids),
        "tests": [{"id": name, "status": "PASS"} for name in loss_ids],
        "nan_or_inf_failures": 0, "all_masked_returns_connected_zero": True,
        "denominator_uses_valid_count": True,
        "nonfound_coordinate_logvariance_gradient_sum": 0.0,
        "s1a_confidence_and_lm_weight_enforced_zero": True,
    }
    json_write(MODEL_DIR / "MODEL_CONFIG.json", model_config)
    json_write(MODEL_DIR / "HEAD_PARAMETER_INVENTORY.json", inventory_report)
    json_write(MODEL_DIR / "HEAD_TEST_REPORT.json", head_report)
    json_write(MODEL_DIR / "SOURCE_MANIFEST.json", source_manifest)
    json_write(MODEL_DIR / "REGRESSION_TEST_REPORT.json", {
        "schema_version": "1.0", "status": "PASS", "created_at_utc": created,
        "tests_run": len(regression_ids), "tests_passed": len(regression_ids),
        "elapsed_seconds": regression_elapsed,
        "scope": "All workspace/mh_pcrau_v3/tests, including locked adapter and G1 regressions",
        "tests": [{"id": name, "status": "PASS"} for name in regression_ids],
    })
    (LOSS_DIR / "LOSS_MASK_TRUTH_TABLE.csv").write_text(loss_truth_table(), encoding="utf-8")
    json_write(LOSS_DIR / "LOSS_TEST_REPORT.json", loss_report)
    artifact_paths = [
        MODEL_DIR / "MODEL_CONFIG.json", MODEL_DIR / "HEAD_PARAMETER_INVENTORY.json",
        MODEL_DIR / "HEAD_TEST_REPORT.json", MODEL_DIR / "REGRESSION_TEST_REPORT.json",
        MODEL_DIR / "SOURCE_MANIFEST.json",
        LOSS_DIR / "LOSS_MASK_TRUTH_TABLE.csv", LOSS_DIR / "LOSS_TEST_REPORT.json", SCHEMA,
    ]
    artifact_manifest = {
        "schema_version": "1.0", "created_at_utc": created,
        "files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                   "bytes": path.stat().st_size} for path in artifact_paths],
    }
    json_write(MODEL_DIR / "ARTIFACT_MANIFEST.json", artifact_manifest)
    decision = {
        "schema_version": "1.0", "date_local": "2026-09-24",
        "outcome": "DAY4_IMPLEMENTATION_SMOKE_PASS", "g2_outcome": "NOT_YET_EVALUATED",
        "day5_authorized": True, "tests": f"{len(test_ids)}/{len(test_ids)} PASS",
        "workspace_regression_tests": f"{len(regression_ids)}/{len(regression_ids)} PASS",
        "total_parameters": total, "s1a_trainable_parameters": trainable,
        "confidence_frozen_parameters": 513, "nonfinite_count": 0,
        "save_load_max_absolute_difference": max_difference,
        "source_manifest_sha256": sha256(MODEL_DIR / "SOURCE_MANIFEST.json"),
        "artifact_manifest_sha256": sha256(MODEL_DIR / "ARTIFACT_MANIFEST.json"),
        "scope": "Synthetic CPU implementation smoke only; no real-data training, no accuracy claim, no Calibration/Test access",
    }
    json_write(MODEL_DIR / "DAY4_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
