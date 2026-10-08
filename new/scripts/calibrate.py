#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcrau.calibration import fit_calibrator, read_jsonl
from pcrau.utils import atomic_json, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the independent post-freeze grounding-risk calibrator")
    parser.add_argument("--config")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    predictions = Path(args.predictions).resolve()
    rows = read_jsonl(predictions)
    config = load_config(args.config)
    expected = int(config["locked_counts"]["calibration_samples"])
    if len(rows) != expected:
        raise ValueError(f"Expected all {expected} independent calibration rows, got {len(rows)}")
    output = Path(args.output).resolve() if args.output else predictions.parent / "calibrator.json"
    if output.exists():
        raise FileExistsError(output)
    result = fit_calibrator(rows, config)
    result["predictions_path"] = str(predictions)
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "risk_policy": result["risk_policy"], "metrics": result["crossfit_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
