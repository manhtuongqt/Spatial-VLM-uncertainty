#!/usr/bin/env python3
"""Summarize the intentionally stopped left-YCB v3 capture without model access."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_v3"
BATCH = OUT / "batch_0"
CAPTURE = BATCH / "capture_attempt_01"
FIGURES = ROOT / "ketqua1/09_danh_gia/figures/ngay_06"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    output = OUT / "PARTIAL_CAPTURE_QC.json"
    if output.exists():
        raise FileExistsError(output)
    annotations = yaml.safe_load((BATCH / "annotations.yaml").read_text())['scenes']
    rows = [json.loads(line) for line in (CAPTURE / "input_manifest.jsonl").read_text().splitlines() if line]
    state_counts, relation_counts, box_counts = Counter(), Counter(), Counter()
    target_pixels, box_pixels, examples, rgb_hashes = [], [], [], Counter()
    states = defaultdict(lambda: {"captured": 0, "valid_ie": 0, "target_pixels": []})
    for row in rows:
        sid = row['scene_id']; ann = annotations[sid]
        labels = cv2.imread(str(CAPTURE / sid / 'evaluator/semantic_labels.png'), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise RuntimeError(f"missing semantic labels: {sid}")
        rgb = CAPTURE / sid / 'input/rgb.png'
        rgb_hashes[sha256(rgb)] += 1
        state, relation = ann['state'], ann['relation_variant']
        state_counts[state] += 1; relation_counts[relation] += 1; states[state]['captured'] += 1
        record = {"scene_id": sid, "state": state, "relation": relation,
                  "label_shape": list(labels.shape), "rgb_sha256": sha256(rgb)}
        if state == 'INSUFFICIENT_EVIDENCE':
            target = int(ann['target_label']); box = ann['occluder_ids'][0]; box_label = int(ann['occluder_labels'][0])
            visible = int((labels == target).sum()); covered = int((labels == box_label).sum())
            record.update({"target_id": ann['target_id'], "primary_occluder": box,
                           "target_visible_pixels": visible, "occluder_visible_pixels": covered,
                           "valid_insufficient_evidence_1_to_119_px": 1 <= visible < 120})
            states[state]['target_pixels'].append(visible)
            states[state]['valid_ie'] += int(record['valid_insufficient_evidence_1_to_119_px'])
            target_pixels.append((box, visible)); box_pixels.append((box, covered)); box_counts[box] += 1
            examples.append(record)
    duplicate_groups = sum(1 for count in rgb_hashes.values() if count > 1)
    for value in states.values():
        pixels = value.pop('target_pixels')
        value['target_pixels_summary'] = None if not pixels else {"min": min(pixels), "median": float(np.median(pixels)), "max": max(pixels)}
    result = {
        "status": "PARTIAL_CAPTURE_REJECT_REDESIGN_REQUIRED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "mh_pcrau_v3_anti_shortcut_left_ycb_v3",
        "capture_status": "INTENTIONALLY_STOPPED_AFTER_EARLY_SEMANTIC_QC",
        "model_inference_performed": False,
        "captured_rows": len(rows), "planned_rows": 64,
        "capture_manifest_present": (CAPTURE / 'capture_manifest.json').exists(),
        "state_counts": dict(state_counts), "relation_counts": dict(relation_counts),
        "primary_occluder_counts": dict(box_counts), "state_qc": dict(states),
        "ie_examples": examples, "exact_rgb_duplicate_groups": duplicate_groups,
        "decision": "Do not use this partial batch for S1a selection, OOF/S1b, Calibration, Test, or G3. Redesign projected occlusion per box/fruit before recapture.",
        "source_hashes": {"annotations": sha256(BATCH / 'annotations.yaml'), "input_manifest": sha256(CAPTURE / 'input_manifest.jsonl')},
    }
    output.write_text(json.dumps(result, indent=2) + '\n')

    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = ['cracker box', 'sugar box', 'bleach cleanser']
    keys = ['ycb_cracker_box', 'ycb_sugar_box', 'ycb_bleach_cleanser']
    target_by_box = [[pixels for box, pixels in target_pixels if box == key] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].boxplot(target_by_box, labels=labels, showfliers=True)
    axes[0].axhspan(1, 119, color='#8fd694', alpha=.25, label='IE valid region (1–119 px)')
    axes[0].set_ylabel('Visible target pixels')
    axes[0].set_title('Occlusion QC by primary YCB box')
    axes[0].legend(loc='upper right')
    ordered_states = ['FOUND', 'AMBIGUOUS', 'ABSENT', 'INSUFFICIENT_EVIDENCE']
    axes[1].bar(ordered_states, [state_counts[x] for x in ordered_states], color=['#4c78a8', '#f58518', '#e45756', '#72b7b2'])
    axes[1].axhline(16, color='black', ls='--', lw=1, label='required per state in one batch')
    axes[1].set_ylim(0, 18); axes[1].set_ylabel('Captured scenes'); axes[1].set_title('Partial capture coverage')
    axes[1].tick_params(axis='x', rotation=20); axes[1].legend()
    fig.savefig(FIGURES / 'DAY6_LEFT_YCB_V3_PARTIAL_QC.png', dpi=180)
    plt.close(fig)
    print(json.dumps({"status": result['status'], "captured": len(rows), "ie_valid": states['INSUFFICIENT_EVIDENCE']['valid_ie']}, indent=2))


if __name__ == '__main__':
    main()
