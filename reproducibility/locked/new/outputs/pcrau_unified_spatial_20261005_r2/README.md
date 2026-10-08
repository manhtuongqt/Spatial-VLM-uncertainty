# P-CRA-U — unified spatial inference bundle

Completed 2026-10-05: existing RGB-D model input or four frozen visual tensors →
frozen Adapter baseline → noun-conditioned P1 anchor → left/right geometry →
44 live features → logistic calibration → perception decision and trace.

- [Architecture and usage](../../docs/UNIFIED_SPATIAL_INFERENCE.md)
- [S3 report](../../../plan/S3_UNIFIED_SPATIAL_INFERENCE_20261005.md)
- [Bundle manifest](bundle.json), [freeze lock](frozen/freeze_lock.json)
- [Calibrator](calibrator.json), [profile](profile.json), [summary](summary.json)
- [RGB-D smoke comparison](smoke/comparison.json)
- [Real dev case table](report/CASE_TABLE.md), [case image](report/dev_cases.png)

All 1600 train, 400 dev and 1000 calibration rows use evidence from a current
model forward, without reading historical predictions into the runtime vector.
Labels are joined only in separate evaluator files after observable export.
Historical dev predictions are read only for a numerical diagnostic.
Calibration uses five family folds and only the calibration split. Threshold
0.2889643687106893; OOF 16 errors / 351 accepted, Wilson upper 7.2757%.
Calibration fit-all counts are resubstitution, not an independent test result.

One existing train RGB-D pair was re-extracted through the actual backbone;
its target grid, 44 features, risk and decision exactly match the feature-cache
route for the same prompt. This is a one-case technical smoke, not universal
parity or an end-to-end robot evaluation.

The S3 live bundle has not been evaluated on IID. Do not attach the S2 cached
330-correct / 33-error / 363-accepted result to this live bundle. Binding and
presence remain UNVERIFIED. Geometry is evidence-only, with no hard veto or
MAP refinement. The selected active baseline profile is unchanged.

Runtime: Sidecar uses the workstation's torch2.13 user-site interpreter, CUDA
BF16; the legacy backbone runs in an isolated conda torch2.5.1 subprocess with
user site disabled. No package install or environment mutation was performed.
Use the CLI normally; do not disable user site in the outer Sidecar process.
The bundle verifies source/runtime/artifact hashes and refers to the baseline
checkpoint/backbone in this workspace; it is not a self-contained portable pack.

The r1 bundle is retained with its source snapshot. It completed feature-level
inference but failed its raw RGB-D smoke at the torch/torchvision import boundary.
Use r2 with current source.
