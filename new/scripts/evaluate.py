#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset, FamilyBatchSampler, collate_samples
from pcrau.engine import evaluate
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.utils import atomic_json, config_path, load_config, read_json, seed_everything, sha256_file, workspace_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen target-architecture checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev", "calibration", "test_iid"], required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--calibrator", help="optional post-freeze calibrator with source thresholds")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--acceptance-ranker", help="optional frozen experimental acceptance scorer")
    parser.add_argument("--support-ranker", help="optional frozen support scorer on the answerability adapter")
    args = parser.parse_args()
    if args.acceptance_ranker and args.support_ranker:
        parser.error("Choose one experimental scorer")
    config_file = config_path(args.config)
    config = load_config(config_file)
    seed_everything(int(config["seed"]))
    profile = args.split if args.split in {"calibration", "test_iid"} else "development"
    dataset = ArchivedPCRAUDataset(config, args.split, profile=profile)
    sampler = FamilyBatchSampler(dataset, int(config["optimization"]["families_per_batch"]), int(config["seed"]), False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCRAUTargetV2(config).to(device)
    metadata = load_model_checkpoint(model, args.checkpoint, sha256_file(config_file))
    if (config["model"].get("answerability_evidence_adapter") or {}).get("include_detail", False):
        config["runtime_checkpoint_sha256"] = sha256_file(Path(args.checkpoint))
    if args.acceptance_ranker:
        from pcrau.acceptance_ranker import load_acceptance_scorer
        model = load_acceptance_scorer(model, Path(args.acceptance_ranker).resolve(), sha256_file(Path(args.checkpoint)))
    if args.support_ranker:
        from pcrau.support_evidence import load_support_scorer
        model = load_support_scorer(model, Path(args.support_ranker).resolve(), sha256_file(Path(args.checkpoint)))
    # Loss weighting is part of the frozen training objective.  Never derive
    # weights from dev/calibration labels, otherwise standalone loss is not
    # comparable with checkpoint-selection loss.
    weight_dataset = ArchivedPCRAUDataset(config, "train", profile="development")
    answer_weights, source_weights = class_weights(weight_dataset, device)
    rr_path = workspace_path(config["paths"]["dev_roborefer_points"]) if args.split == "dev" else None
    source_thresholds = None
    if args.calibrator:
        calibrator = read_json(Path(args.calibrator).resolve())
        # Older frozen V2 calibrators contain risk/conformal calibration but
        # predate per-source threshold fitting.  In that case retain the
        # preregistered 0.5 source threshold; never infer thresholds on Test.
        if "source_thresholds" in calibrator:
            source_thresholds = {
                name: float(row["threshold"])
                for name, row in calibrator["source_thresholds"].items()
            }
    metrics, predictions = evaluate(
        model, loader, device, config, answer_weights, source_weights, rr_points_path=rr_path,
        source_thresholds=source_thresholds,
        max_batches=1 if args.smoke else None,
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(args.checkpoint).resolve().parents[2] / "evaluation" / args.split
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "metrics.json", {"checkpoint": metadata, "split": args.split, **metrics})
    with (output_dir / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "samples": metrics["samples"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
