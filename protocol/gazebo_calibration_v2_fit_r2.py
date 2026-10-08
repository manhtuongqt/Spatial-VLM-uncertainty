#!/usr/bin/env python3
"""Fit/freeze wrapper for Calibration-v2 materialization revision r3.

Only the materialization-lock reference differs from the preregistered v2
wrapper. The inherited one-temperature algorithm, gates, and threshold rule
are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gazebo_calibration_v1_fit as impl


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v2"
AMENDMENT = ROOT / "protocol/gazebo_calibration_v2_fit_r2_amendment.json"


def configure() -> None:
    values = {
        "DATA": ROOT / "datasets/Gazebo_calibration_v2",
        "RESULT": RESULT,
        "PREDICTIONS": RESULT / "b0_predictions/predictions.jsonl",
        "PREDICTION_RUN": RESULT / "b0_predictions/run.json",
        "CONTRACT": ROOT / "protocol/GAZEBO_CALIBRATION_V2_CONTRACT_LOCK.json",
        "MATERIALIZATION_LOCK": ROOT / "protocol/gazebo_calibration_v2_materialization_r3_lock.json",
        "FIT_LOCK": ROOT / "protocol/GAZEBO_CALIBRATION_V2_FIT_INPUT_R2_LOCK.json",
        "CALIBRATOR": RESULT / "temperature_calibrator.json",
        "METRICS": RESULT / "GAZEBO_CALIBRATION_METRICS.json",
        "REPORT": RESULT / "GAZEBO_CALIBRATION_RESULT.md",
        "FINAL_LOCK": RESULT / "CALIBRATOR_THRESHOLD_LOCK.json",
    }
    for name, value in values.items():
        setattr(impl, name, value)


def lock_inputs() -> None:
    configure()
    impl.lock_fit_inputs()
    path = impl.FIT_LOCK
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(protocol_id="gazebo_calibration_v2", fit_revision="r2")
    value["source_artifact_sha256"].update({
        str(Path(__file__).resolve().relative_to(ROOT)): impl.sha(Path(__file__).resolve()),
        str(AMENDMENT.relative_to(ROOT)): impl.sha(AMENDMENT),
    })
    impl.dump(path, value)
    print(json.dumps({"status": "LOCKED_FINAL_FIT_R2", "sha256": impl.sha(path)}, indent=2))


def fit() -> None:
    configure()
    impl.fit()
    for path in (impl.CALIBRATOR, impl.METRICS):
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(protocol_id="gazebo_calibration_v2", fit_revision="r2")
        impl.dump(path, value)
    report = impl.REPORT.read_text(encoding="utf-8")
    report = report.replace("Calibration v1", "Calibration v2").replace("Calibration-v1", "Calibration-v2")
    impl.REPORT.write_text(report, encoding="utf-8")


def freeze() -> None:
    configure()
    impl.freeze()
    value = json.loads(impl.FINAL_LOCK.read_text(encoding="utf-8"))
    value.update(
        protocol_id="gazebo_calibration_v2",
        fit_revision="r2",
        fit_wrapper_sha256=impl.sha(Path(__file__).resolve()),
        fit_amendment_sha256=impl.sha(AMENDMENT),
    )
    impl.dump(impl.FINAL_LOCK, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock-fit-inputs", "fit", "freeze"))
    command = parser.parse_args().command
    if command == "lock-fit-inputs":
        lock_inputs()
    elif command == "fit":
        fit()
    else:
        freeze()


if __name__ == "__main__":
    main()
