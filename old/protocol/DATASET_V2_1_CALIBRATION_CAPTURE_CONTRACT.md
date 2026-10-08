# Dataset V2.1 Calibration Capture Contract

This capture contract is subordinate to
`protocol/PCRA_U_CALIBRATION_CONTRACT.md`; both documents are locked before
capture.

- Protocol: `roborefer_dataset_v2_1_calibration_capture_200`.
- Split: exactly `calibration`.
- Size: 200 independent families, 400 synchronized clean/occlusion captures,
  1,000 dependent variants.
- Primary-family states: 80 `FOUND`, 50 `INSUFFICIENT_EVIDENCE`, 35
  `AMBIGUOUS`, 35 `ABSENT`.
- Batch gates: `canary_000=20`, `batch_001=90`, `batch_002=90`; each batch
  requires observed QC PASS before the next begins.
- IDs, seeds, layouts, captures, exact instructions and template-family IDs
  are new and disjoint from every historical dataset.
- Seen/IID assets only; `pear`, `plum`, `tuna_fish_can` remain Test-OOD-only.
- Relation truth uses clean synchronized semantic-mask centroids and robust
  median metric depth under geometry V2.1.
- Capture is resume-safe and create-only by `capture_id`; ordered shutdown is
  mandatory.
- No P-CRA-U/RoboRefer training or inference occurs during capture.
- Test-IID/Test-OOD are not created or opened.
- Only real RGB/depth/semantic/mask QC images may be used as visual evidence.

Full QC must return `GO_CALIBRATION_INFERENCE_AND_FIT`; otherwise model
inference and calibrator fitting remain forbidden.

