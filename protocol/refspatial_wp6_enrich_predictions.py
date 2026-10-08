#!/usr/bin/env python3
"""Attach immutable WP6 challenge metadata to already-generated predictions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/primary_challenge.jsonl"
EVAL = ROOT / "results/spatial_vlm_refspatial_v1/wp6_unbiased_challenge/evaluation"


def main():
    source = {row["sample_id"]: row for row in (json.loads(x) for x in CHALLENGE.read_text().splitlines() if x)}
    for model in ("b0", "b1"):
        path = EVAL / model / "predictions.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x]
        if {row["sample_id"] for row in rows} != set(source) or len(rows) != 180:
            raise ValueError(f"{model}: prediction IDs do not exactly match the fixed challenge manifest")
        for row in rows:
            fixed = source[row["sample_id"]]
            row["challenge_stratum"] = fixed["challenge_stratum"]
            row["challenge_role"] = fixed["challenge_role"]
            row["label_status"] = fixed["label_status"]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"{model}: enriched {len(rows)} predictions")


if __name__ == "__main__":
    main()
