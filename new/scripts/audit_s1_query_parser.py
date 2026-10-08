#!/usr/bin/env python3
"""Prompt-only S1 parsing, then train/dev annotation audit. No model forward."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrau.dataset import ArchiveLayout
from pcrau.language_augmentation import PREFIXES, validate_prefix
from pcrau.query_parser import PARSER_VERSION, parse_query
from pcrau.text import parse_relations, prompt_anchor_mask, tokenize, words
from pcrau.utils import sha256_file, workspace_path

SCOPE = {"direct", "left_of", "right_of"}
FRUIT = {"apple", "orange", "lemon", "mango"}


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def expected_phrase(ids, registry):
    """Generator's semantic phrase policy, evaluator only."""
    assert ids and all(key in registry for key in ids)
    names = {registry[key]["semantic_class"] for key in ids}
    if len(ids) > 1 and names.issubset(FRUIT):
        return "fruit"
    return next(iter(names)) if len(names) == 1 else "object"


def span_checks(prompt, q, cfg):
    masks = q.token_masks()
    if not q.supported:
        return (q.predicate is None and q.reference_frame is None and q.target is None
                and not q.anchors and not any(masks["target_token_mask"])
                and not any(masks["anchor_slot_mask"])
                and not any(any(m) for m in masks["anchor_token_masks"]))
    ids, real = tokenize(prompt, q.max_tokens, int(cfg["vocab_size"]))
    phrases = [q.target, *q.anchors]
    selected = [masks["target_token_mask"], *masks["anchor_token_masks"][:len(q.anchors)]]
    for phrase, mask in zip(phrases, selected):
        if words(prompt[phrase.char_start:phrase.char_end]) != phrase.text.split():
            return False
        if words(prompt)[phrase.token_start:phrase.token_end] != phrase.text.split():
            return False
        phrase_ids, _ = tokenize(phrase.text, len(phrase.text.split()), int(cfg["vocab_size"]))
        if [v for v, keep in zip(ids, mask) if keep] != phrase_ids:
            return False
        if any(keep and not actual for keep, actual in zip(mask, real)):
            return False
    return (masks["anchor_slot_mask"] == prompt_anchor_mask(prompt, q.max_anchors)
            and parse_relations(prompt, int(cfg["max_relations"])) == [q.predicate])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="new/outputs/pcrau_s1_query_parser_20261005_r3")
    args = parser.parse_args()
    output = workspace_path(args.output)
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    active_path = workspace_path("new/outputs/active_experimental_profile.json")
    active = load(active_path)
    cfg_path = workspace_path(active["config"])
    config = load(cfg_path)
    cfg = config["model"]
    layout = ArchiveLayout(config, "development")
    protected = {}

    def protect(path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Hash mismatch: {path}")
        protected[str(path)] = digest

    for key in ("config", "checkpoint", "calibrator", "profiles"):
        protect(workspace_path(active[key]), active[key + "_sha256"])
    protect(active_path)
    for relative in ("new/src/pcrau/text.py", "new/src/pcrau/model.py", "new/src/pcrau/dataset.py",
                     "new/src/pcrau/losses.py", "new/src/pcrau/language_augmentation.py",
                     "new/scripts/audit_s0_anchor_binding.py"):
        protect(workspace_path(relative))
    protect(layout.manifest)
    entries = load(layout.manifest)["entries"]
    assert len({e["sample_id"] for e in entries}) == len(entries)
    assert all(e["split"] in {"train", "dev"} for e in entries)
    families = {split: {e["family_id"] for e in entries if e["split"] == split} for split in ("train", "dev")}
    assert not families["train"] & families["dev"]
    queries, predictions, presentations = {}, [], []
    for e in entries:
        prompt = e["feature_input"]["prompt"]
        assert hashlib.sha256(prompt.encode()).hexdigest() == e["feature_input"]["prompt_sha256"]
        q = parse_query(prompt, int(cfg["max_tokens"]), int(cfg["max_anchors"]))
        assert span_checks(prompt, q, cfg), e["sample_id"]
        queries[e["sample_id"]] = q
        predictions.append({"sample_id": e["sample_id"], "family_id": e["family_id"], "split": e["split"],
                            "prompt": prompt, "parsed": q.to_dict()})
        prefixes = PREFIXES if e["split"] == "train" else ("",)
        for index, prefix in enumerate(prefixes):
            presentation = validate_prefix(prompt, prefix, cfg) if e["split"] == "train" else prompt
            p = parse_query(presentation, q.max_tokens, q.max_anchors)
            assert span_checks(presentation, p, cfg), (e["sample_id"], prefix)
            assert p.status == q.status and p.predicate == q.predicate
            if q.supported:
                assert [v.text for v in [p.target, *p.anchors]] == [v.text for v in [q.target, *q.anchors]]
                for original, shifted in zip([q.target, *q.anchors], [p.target, *p.anchors]):
                    assert shifted.token_start == original.token_start + len(words(prefix))
                    assert shifted.char_start == original.char_start + len(prefix)
            presentations.append({"sample_id": e["sample_id"], "split": e["split"], "prefix_index": index,
                                  "prefix": prefix, "prompt": presentation, "parsed": p.to_dict()})
    output.mkdir(parents=True)
    write_jsonl(output / "parsed_query_predictions.jsonl", predictions)
    write_jsonl(output / "presentation_predictions.jsonl", presentations)
    # Persist prompt-only outputs before any record, registry or mask reads.
    prompt_digest = sha256_file(output / "parsed_query_predictions.jsonl")
    registry_path = layout.dataset_root / "object_registry.json"
    protect(registry_path)
    registry = {v["id"]: v for v in load(registry_path).values()}
    rows, inventory, negatives = [], Counter(), []
    for e in entries:
        q = queries[e["sample_id"]]
        relation = e["audit_only"]["relation"]
        inventory[(e["split"], relation, q.status)] += 1
        assert q.supported == (relation in SCOPE), e["sample_id"]
        row = {"sample_id": e["sample_id"], "family_id": e["family_id"], "split": e["split"],
               "variant": e["variant"], "prompt": e["feature_input"]["prompt"], "status": q.status,
               "reason": q.reason, "predicate": q.predicate, "reference_frame": q.reference_frame,
               "target_phrase": q.target.text if q.target else None,
               "target_char_span": [q.target.char_start, q.target.char_end] if q.target else None,
               "target_token_span": [q.target.token_start, q.target.token_end] if q.target else None,
               "anchor_phrases": [p.text for p in q.anchors],
               "anchor_char_spans": [[p.char_start, p.char_end] for p in q.anchors],
               "anchor_token_spans": [[p.token_start, p.token_end] for p in q.anchors],
               "annotation_relation": relation, "span_tokenizer_match": True,
               "annotation_match": None, "annotation_checks": None}
        if q.supported:
            record_path = layout.dataset_path(e["record_path"])
            protect(record_path, e["record_sha256"])
            record = load(record_path)
            oracle = record["evaluator_only"]
            spatial = oracle["spatial_label"]
            masks = oracle["masks"]["anchor"]
            expected = expected_phrase(spatial["candidate_target_ids"], registry)
            checks = {
                "prompt": record["inference_payload"]["prompt"] == e["feature_input"]["prompt"],
                "target_phrase": q.target.text == expected,
                "predicate": spatial["relations"] == [q.predicate],
                "anchor_phrase_count": len(q.anchors) == len(spatial["anchor_ids"]) == len(masks) == len(e["supervision"]["anchor_masks"]),
                "frame": spatial["reference_frame"] == ("image" if q.anchors else "object_semantics"),
                "anchor_phrases": all(p.text == expected_phrase([oid], registry) for p, oid in zip(q.anchors, spatial["anchor_ids"])),
                "graph": spatial["relation_graph"] == [{"clause_index": 0,
                    "predicate": q.predicate, "reference_frame": "image" if q.anchors else "object_semantics",
                    "source_ids": spatial["valid_target_ids"], "target_ids": spatial["anchor_ids"]}],
            }
            visible_pixels, mask_paths = [], []
            checks["mask_association"] = True
            for slot, descriptor in enumerate(masks):
                path = layout.dataset_path(descriptor["path"])
                supervision = e["supervision"]["anchor_masks"][slot]
                checks["mask_association"] &= (descriptor["object_id"] == spatial["anchor_ids"][slot]
                    and path == layout.dataset_path(supervision["path"])
                    and descriptor["sha256"] == supervision["sha256"])
                protect(path, descriptor["sha256"])
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                assert mask is not None and mask.shape == (480, 640)
                visible_pixels.append(int((mask > 0).sum()))
                mask_paths.append(str(path))
                if not visible_pixels[-1]:
                    negatives.append({"sample_id": e["sample_id"], "family_id": e["family_id"],
                                      "split": e["split"], "anchor_phrase": q.anchors[slot].text,
                                      "mask_path": str(path), "visible_pixels": 0})
            row.update({"annotation_target_phrase": expected, "candidate_target_ids": spatial["candidate_target_ids"],
                        "valid_target_ids": spatial["valid_target_ids"], "anchor_ids": spatial["anchor_ids"],
                        "anchor_mask_paths": mask_paths, "anchor_visible_pixels": visible_pixels,
                        "annotation_checks": checks, "annotation_match": all(checks.values())})
            assert row["annotation_match"], (e["sample_id"], checks)
        rows.append(row)
    write_jsonl(output / "evaluator_rows.jsonl", rows)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output / "phrase_token_annotation.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    unchanged = all(sha256_file(Path(path)) == digest for path, digest in protected.items())
    assert unchanged
    assert sha256_file(output / "parsed_query_predictions.jsonl") == prompt_digest
    checks = {"all_original_span_checks_pass": True, "all_presentation_span_checks_pass": True,
              "supported_annotation_checks_pass": True, "out_of_scope_fail_closed": True,
              "prefix_identity_preserved": True, "train_dev_family_disjoint": True,
              "prompt_predictions_saved_before_oracle": True, "protected_files_unchanged": unchanged,
              "baseline_forward_untouched": True, "model_forward_executed": False,
              "training_executed": False, "calibration_samples_accessed": False, "test_samples_accessed": False}
    counts = {}
    for split in ("train", "dev"):
        subset = [r for r in rows if r["split"] == split]
        p = [r for r in presentations if r["split"] == split]
        neg = [r for r in negatives if r["split"] == split]
        counts[split] = {"samples": len(subset), "families": len(families[split]),
            "supported": sum(r["status"] == "supported" for r in subset),
            "direct": sum(r["predicate"] == "direct" for r in subset),
            "horizontal": sum(r["predicate"] in {"left_of", "right_of"} for r in subset),
            "unsupported": sum(r["status"] != "supported" for r in subset),
            "presentations": len(p), "supported_presentations": sum(r["parsed"]["status"] == "supported" for r in p),
            "empty_active_anchor_samples": len(neg), "empty_active_anchor_families": len({r["family_id"] for r in neg})}
    summary = {"status": "S1_PARSER_TRAIN_DEV_AUDIT_COMPLETE", "parser_version": PARSER_VERSION,
        "counts": counts, "annotation_matches": sum(r["annotation_match"] is True for r in rows),
        "original_span_checks": len(rows), "presentation_span_checks": len(presentations),
        "inventory": [{"split": split, "annotation_relation": relation, "status": status, "count": count}
                      for (split, relation, status), count in sorted(inventory.items())],
        "empty_active_anchors": negatives, "protected_file_count": len(protected),
        "parsed_query_predictions_sha256": prompt_digest, "validation_checks": checks,
        "limits": ["Controlled English grammar, not general natural-language semantic parsing",
                   "Parser does not improve learned anchor predictions", "No new model or uncertainty claim"]}
    write_json(output / "summary.json", summary)
    write_json(output / "validation_checks.json", checks)
    write_json(output / "provenance.json", {"protected_sha256": protected,
        "new_source_sha256": {str(path): sha256_file(path) for path in (
            Path(__file__).resolve(), workspace_path("new/src/pcrau/query_parser.py"), workspace_path("new/tests/test_query_parser.py"))},
        "manifest": str(layout.manifest), "argv": sys.argv, "parser_version": PARSER_VERSION})
    print(json.dumps({"status": summary["status"], "output": str(output), "counts": counts,
                      "annotation_matches": summary["annotation_matches"], "protected_files": len(protected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
