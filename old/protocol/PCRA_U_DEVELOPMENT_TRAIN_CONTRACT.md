# P-CRA-U development training v1 — locked contract

## 1. Scientific scope

This protocol is authorized by the V2.1.1 dataset full-QC decision
`GO_DEVELOPMENT_TRAIN`.  Its outputs are **EXPLORATORY_DEV_ONLY**: train is used
for gradients, dev is used only for checkpoint selection and exploratory
diagnostics. Calibration, Test-IID and Test-OOD remain sealed, so Tables 2–6
cannot be reported as final thesis results.

The immutable input is exactly 400 independent families / 2,000 records:

- train: 320 families / 1,600 records;
- dev: 80 families / 400 records;
- five variants of every family remain in the family's original split;
- no new sample-level split is created.

## 2. Baseline and feature boundary

RoboRefer is a frozen feature provider. No RoboRefer parameter, projector,
`generate()` path, URDF, controller or Gazebo file may be modified or placed in
the optimizer.

RGB and relative-depth inputs are read only from each record's oracle-free
`inference_payload`. The two towers produce `R0_raw/D0_raw` before their
projectors with runtime shape `[13,1024,1152]`. The already locked fixed transform
is applied immediately:

1. tiles 0–11 reshape to 12 local `32x32` grids;
2. average-pool each grid by `4x4`;
3. assemble the row-major `3x4` tile layout into `[24,32,1152]`;
4. tile 12 contributes only its layer-normalized mean thumbnail feature.

The derived cache is stored FP16 and promoted to FP32 for sidecar training. It is
deduplicated only when the RGB and depth SHA-256 pair is exactly identical.
Raw 61 MiB/sample tower tensors are not retained because the workspace had only
about 12 GiB free at protocol lock; retaining 2,000 raw pairs would require about
120 GiB. This storage decision does not change the feature stage or trainable
transform: pooling is fixed and parameter-free.

Feature cache metadata contains no mask, object ID, target ID, oracle relation
graph, answerability label or uncertainty source label. Supervision is joined by
the training loader after feature extraction and is never passed as model input.

## 3. Model input and outputs

Model input is limited to frozen pooled RGB/depth features and language derived
from the prompt. Relation ID is parsed lexically from the prompt; evaluator
relation labels are audit-only and never fed to the model.

P-CRA-U development v1 contains:

- relation-conditioned RGB/depth fusion;
- target heatmap `[24,32]`;
- four-class answerability head in order `FOUND`, `AMBIGUOUS`, `ABSENT`,
  `INSUFFICIENT_EVIDENCE`;
- four active source logits: `semantic`, `relation`, `depth`, `occlusion`.

The plan's `spatial` source logit is explicitly deferred because Dataset V2.1.1
contains zero positive `spatial` source labels. It is forbidden to invent or
retrofit those labels after observing the data.

## 4. Supervision and loss

Evaluator masks/labels may be read only as train supervision or dev evaluation.
Heatmap loss is active only for ground-truth `FOUND`; this also avoids treating
the known empty masks from `ABSENT` and some `INSUFFICIENT_EVIDENCE` samples as
positive localization evidence.

`L = 1.0 L_heatmap + 0.5 L_answerability + 0.5 L_source`.

Answerability and source class weights are computed once from all 1,600 train
records. Dev never changes weights, thresholds or optimizer state.

## 5. Optimization and selection

The only primary configuration is the byte-locked JSON config. Sidecar training
uses deterministic FP32 AdamW, family-preserving deterministic sample order,
batch 8, LR `3e-4`, gradient clipping at 5, warmup then cosine decay, at most 20
epochs and early stopping on `dev_total_loss` after the locked minimum epoch.

Every epoch checkpoint is immutable. `best_dev_total_loss.json` may be updated
only from dev; calibration/test selection is rejected by the checkpoint manager.
Checkpoint identity binds config, dataset/split manifest, code, feature-cache
index, contract and execution lock hashes.

All commands must run through `protocol/run_pcra_u_development.sh`. The wrapper
sets `PYTHONNOUSERSITE=1` so the RoboRefer environment cannot accidentally mix
user-site Torch with its locked Torch/Torchvision pair; it performs no package or
baseline modification.

## 6. Gates

### Preflight (no optimizer step)

- full-QC remains PASS/`GO_DEVELOPMENT_TRAIN` and tests remain absent;
- exact family/sample/split/state counts and five variants per family;
- all record, RGB, depth, mask and manifest hashes resolve inside the workspace;
- inference payload is oracle-free and OOD heldout assets remain absent;
- prompt-only relation parser matches all audit labels;
- one official train batch has valid shapes, finite forward/backward loss and
  nonzero gradients for all active sidecar groups;
- frozen features have no gradient and no optimizer step occurs;
- baseline/model/dataset hashes are unchanged.

PASS decision: `GO_DEVELOPMENT_TRAIN_CANARY`.

### Canary (optimizer steps allowed on canary train only)

The canary uses the already locked `canary_000` families: 16 train and 4 dev,
all five variants. Two identical runs and a checkpoint resume replay must have
exact model hashes and canonical metrics. Loss/gradient must remain finite,
every active group must receive a nonzero gradient, features remain frozen,
checkpoint audit passes and final/initial median loss ratio is at most 1.10.

PASS decision: `GO_FULL_DEVELOPMENT_TRAIN`.

### Full development training

Only after both preceding gates pass may all 1,600 train records update the
sidecar. All 400 dev records are evaluation-only. The report must retain
`EXPLORATORY_DEV_ONLY`, keep Tables 2–6 `NOT_RUN`, preserve dataset/baseline
hashes and attest Calibration/Test remained sealed.

## 7. Evidence policy

Checkpoints remain under `results/`; `ketqua` stores their path, SHA-256 and audit
metadata, not large binary tensors. Accepted figures are measured training curves
and real RGB/GT/predicted-heatmap overlays. Planned/decorative infographics are
forbidden.
