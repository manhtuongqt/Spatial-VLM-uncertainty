# P-CRA-U task uncertainty / S6

Final results: [Vietnamese thesis report](../../../plan/S6_TASK_UNCERTAINTY_RESULTS_20261007.md).
Runtime: [evaluation_r3/bundle_release.json](evaluation_r3/bundle_release.json).
The release changes metadata only; its evaluated parent manifest is retained.
Baseline/active profile/archive are unchanged. This is a separate variant.

- Three trained task heads, 87,199 parameters, best dev epoch15.
- MC20 on learned fusion and task-head dropouts; fixed backbone/text/P1 queries.
- Closed-set semantic node identity; spatial/left-right disagreement; Gaussian
  depth mixture; reference-support completion under physical occlusion.
- No claim of five causal uncertainty sources, full-VLM epistemic decomposition,
  full amodal reconstruction, robot safety or independent unseen IID testing.
- Calibration1000/200families fit after freeze. Risk60 threshold0.37876498603729764.
- Old IID1000/200families:337 correct +34 errors /371 accepts; risk9.1644%.
- Reference live44:330+33/363; risk9.0909%. IID budget7.5% remains unmet.

Data supplement and photos: [data/supplement.json](data/supplement.json),
[preview](data/occlusion_label_preview.png). Masks/labels only in loss/evaluator.
Completion124train/32dev pairs; only9/6 meaningful hidden pairs. No new capture.

Complete train/dev outputs and raw samples are in **evaluation_r2/train,dev**.
Complete calibration/IID outputs and raw batches are in **evaluation_r3**.
R1 numerical guard failed; r2 calibration export was incomplete; all failed
files are preserved. R3 calibration outputs replay the saved r2 values exactly
after JSON roundtrip; no retraining or checkpoint reselection occurred.

Reports/actual case figures: [evaluation_r3/report_r2](evaluation_r3/report_r2).
First reporter stopped on an unsupported parser case and is preserved; the
second reporter renders scope explicitly. No prediction/confidence was edited.
RGB-D vs feature-path parity for one dev case: [smoke/parity.json](evaluation_r3/smoke/parity.json).

CLI from workspace root:

```bash
.conda-roborefer/bin/python3.10 -B new/scripts/infer_task_variant.py \
  --features PATH.safetensors --prompt 'Locate the apple that is right of the purple cube.' \
  --output NEW.json
```

Alternatively use `--rgb RGB.jpg --depth RELATIVE_DEPTH.png`. The CLI does not
capture or move a robot. Component probabilities/intervals are not automatically
calibrated because downstream error risk has been calibrated.
