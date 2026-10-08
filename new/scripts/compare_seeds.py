#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen best metrics across V3 seeds")
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        best = json.loads((run.resolve() / "best.json").read_text())
        dev = best["metrics"]["dev"]
        rows.append({
            "run": run.name,
            "epoch": best["epoch"],
            "dev_total_loss": dev["loss"]["total"],
            "grounding_accuracy": dev["grounding_accuracy"],
            "answerability_macro_f1": dev["answerability"]["macro_f1"],
            "source_macro_f1": dev["source"]["macro_f1"],
            "relation_edge_accuracy": dev["relation_edge_accuracy"],
        })
    numeric = [key for key in rows[0] if key not in {"run", "epoch"}]
    summary = {
        key: {"mean": float(np.mean([row[key] for row in rows])),
              "sample_std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0}
        for key in numeric
    }
    print(json.dumps({"runs": rows, "aggregate": summary}, indent=2))


if __name__ == "__main__":
    main()
