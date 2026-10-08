#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcrau.calibration import read_jsonl
from pcrau.policy import decide
from pcrau.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply calibrated selective-action policy")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    calibrator = read_json(Path(args.calibrator).resolve())
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(decide(row, calibrator), ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "decisions": len(rows)}))


if __name__ == "__main__":
    main()
