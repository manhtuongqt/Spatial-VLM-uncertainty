#!/usr/bin/env python3
"""Correct paired reporting from saved train/dev traces, without model execution.

The original run, dev selections and checkpoints remain immutable. The sole
metric correction restores the protocol's same-cache, visible pair definition.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrau.anchor_shadow_pilot import aggregate, family_bootstrap, locked_gates, pair_rows, paired_changes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def jsonl(path, values):
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("new/outputs/pcrau_s1_anchor_shadow_pilot_20261005"))
    args = parser.parse_args()
    root = args.run.resolve()
    original = read(root / "summary.json")
    provenance = read(root / "provenance.json")
    snapshots = read(root / "source_snapshot/manifest.json")
    inputs = [root / name for name in ("summary.json", "provenance.json", "freeze_lock.json",
        "evaluator_rows.jsonl", "selected_inference_predictions.jsonl", "selected_spatial_logits.npz",
        "initial_baseline_metrics.json", "epoch_history.json", "latency.json")]
    for arm in ("M1", "M2"):
        inputs += [root / arm / "checkpoints/best" / name for name in ("mlp.safetensors", "metadata.json")]
    hashes = {str(p): sha(p) for p in inputs}
    # Preserve the sources that actually trained these checkpoints, and check
    # all other protected files against the original run's freeze audit.
    changed_sources = []
    for path, digest in provenance["protected_sha256"].items():
        if sha(path) != digest:
            saved = snapshots.get(path)
            assert saved and saved["sha256"] == digest and sha(root / saved["snapshot"]) == digest, path
            changed_sources.append(path)
    freeze = read(root / "freeze_lock.json")
    for arm in ("M1", "M2"):
        assert sha(root / original["arms"][arm]["selected_checkpoint"]) == freeze["checkpoints"][arm]
    rows = [json.loads(line) for line in (root / "evaluator_rows.jsonl").read_text().splitlines()]
    groups = {s: aggregate([r for r in rows if r["split"] == s]) for s in ("train", "dev")}
    assert groups["dev"] == original["groups"]["dev"], "Dev/selection changed"
    assert original["groups"]["train"]["matched_swap_pairs"] == 50
    assert groups["train"]["matched_swap_pairs"] == 51
    assert groups["train"]["models"] == original["groups"]["train"]["models"]
    pairs = pair_rows(rows)
    corrected_pair = next(p for p in pairs if p["family_id"] == "v211dev_family_000050")
    assert corrected_pair["eligible"] and not corrected_pair["anchor_phrase_changed"]
    assert not any(m["both_hit"] for m in corrected_pair["models"].values())
    for arm in ("M1", "M2"):
        assert groups["dev"]["models"][arm] == read(root / arm / "checkpoints/best/metadata.json")["dev"]
    contrasts = (("M1", "M0"), ("M2", "M0"), ("M1", "M2"))
    deltas, bootstrap = {}, {}
    for split in ("train", "dev"):
        subset = [r for r in rows if r["split"] == split]
        deltas[split] = {f"{a}_vs_{b}": paired_changes(subset, a, b) for a, b in contrasts}
        bootstrap[split] = {f"{a}_vs_{b}": family_bootstrap(subset, a, b) for a, b in contrasts}
    assert deltas == original["paired_changes"]
    assert bootstrap["dev"] == original["family_bootstrap"]["dev"]
    gates = locked_gates([r for r in rows if r["split"] == "dev"],
        original["gates"]["gates"]["technical"], original["gates"]["gates"]["latency_le_20_percent"])
    assert gates == original["gates"]
    stratified = {}
    for field in ("variant", "truth"):
        stratified[field] = {f"{split}/{v}": aggregate([r for r in rows if r["split"] == split and r[field] == v])
            for split, v in sorted({(r["split"], r[field]) for r in rows})}
    phrases = defaultdict(list)
    for row in rows:
        phrases[f"{row['split']}/{row['parsed']['anchors'][0]['text']}"].append(row)
    stratified["anchor_phrase"] = {k: aggregate(v) for k, v in sorted(phrases.items())}
    family_deltas = []
    for split, family in sorted({(r["split"], r["family_id"]) for r in rows}):
        subset = [r for r in rows if r["split"] == split and r["family_id"] == family]
        metrics = aggregate(subset)
        for a, b in contrasts:
            delta = paired_changes(subset, a, b)
            family_deltas.append({"split": split, "family_id": family, "contrast": f"{a}_vs_{b}",
                "visible": metrics["visible"], "found": metrics["found"], "matched_pairs": metrics["matched_swap_pairs"],
                **delta, "found_both_delta": metrics["models"][a]["found_both_hits"] - metrics["models"][b]["found_both_hits"],
                "swap_both_delta": metrics["models"][a]["swap_both_hits"] - metrics["models"][b]["swap_both_hits"]})
    destination = root / "paired_report_r2"
    destination.mkdir(exist_ok=False)
    summary = {**original, "groups": groups, "gates": gates, "paired_changes": deltas,
        "family_bootstrap": bootstrap, "stratified": stratified,
        "artifact_root": str(root), "report_revision": "paired_report_r2",
        "correction": {"reason": "Remove unintended noun-phrase-change pair filter; protocol uses visible and same-cache",
            "original_train_pairs": 50, "corrected_train_pairs": 51, "pair": corrected_pair,
            "dev_unchanged": True, "training_and_selected_checkpoints_unchanged": True,
            "original_summary_retained": str(root / "summary.json")}}
    write(destination / "summary.json", summary)
    jsonl(destination / "swap_pairs.jsonl", pairs)
    jsonl(destination / "family_paired_deltas.jsonl", family_deltas)
    with (destination / "family_paired_deltas.csv").open("w", newline="") as out:
        fields = [k for k in family_deltas[0] if not k.endswith("sample_ids")]
        writer = csv.DictWriter(out, fields); writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in family_deltas)
    jsonl(destination / "empty_cases.jsonl", [r for r in rows if not r["anchor_pixels"]])
    jsonl(destination / "remaining_dev_misses.jsonl", [r for r in rows if r["split"] == "dev" and
        r["anchor_pixels"] and not r["models"]["M1"]["anchor_hit"]])
    initial = read(root / "initial_baseline_metrics.json")
    initial["train"]["matched_swap_pairs"] = groups["train"]["matched_swap_pairs"]
    write(destination / "initial_baseline_metrics.json", initial)
    assert all(sha(p) == digest for p, digest in hashes.items())
    write(destination / "provenance.json", {"read_only_source_artifacts_sha256": hashes,
        "source_sha256": {str(Path(__file__).resolve()): sha(__file__),
            str(Path(__file__).resolve().parents[1] / "src/pcrau/anchor_shadow_pilot.py"):
            sha(Path(__file__).resolve().parents[1] / "src/pcrau/anchor_shadow_pilot.py")},
        "changed_sources_archived": changed_sources, "original_training_source_snapshots": snapshots,
        "checks": {"original_artifacts_unchanged": True, "dev_checkpoint_selection_unchanged": True,
            "baseline_dataset_and_other_protected_files_unchanged": True,
            "no_model_execution_or_training": True, "no_test_or_calibration_samples_read": True}})
    print(json.dumps({"report": str(destination), "train_pairs": groups["train"]["matched_swap_pairs"],
        "dev_unchanged": True, "upgrade_gates_pass": gates["pass"]}))


if __name__ == "__main__":
    main()
