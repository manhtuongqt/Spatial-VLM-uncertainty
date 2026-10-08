# Dataset V2 relation geometry amendment 01

## Status

- Amendment ID: `DATASET_V2_RELATION_GEOMETRY_AMENDMENT_01`.
- Successor protocol family: `roborefer_dataset_v2_1`.
- Trigger: `batch_001` of Dataset V2 failed three runtime metric-depth relation checks after 190/190 structurally valid captures.
- Decision at entry: `FIX_RELATION_GEOMETRY_BEFORE_MORE_DEVELOPMENT_CAPTURE`.
- Training, Calibration, Test-IID and Test-OOD remain forbidden.
- This amendment does not edit, waive or reinterpret the historical V2 gate.

The 230 existing development captures are permanently marked
`ENGINEERING_DIAGNOSTIC_ONLY_NOT_OFFICIAL_DATA`. They may be used to diagnose
geometry and fit an engineering-only layout predictor, but their family IDs,
capture IDs and seeds are deny-listed from every official V2.1 split.

## Authoritative observable relation contract

All relations are evaluated on a synchronized clean capture containing RGB,
metric depth and semantic-instance labels. Simulator object centers are
auxiliary pre-capture evidence only.

### Instance evidence

For semantic label `l`:

1. `M_l = semantic_instance_labels == l`.
2. The 2-D centroid is computed from all pixels in `M_l`.
3. For depth relations, erode `M_l` by a 3x3 kernel for two iterations.
4. Keep only finite depth values in `[0.10, 2.00]` metres.
5. Require at least 64 valid interior pixels.
6. `z_l` is the median of the remaining metric-depth values.
7. Median absolute deviation and the 10th/90th percentiles are retained as
   diagnostic spread; they do not replace the median label.

### Locked margins

- Horizontal centroid margin: `12 px` at the locked 640x480 resolution.
- Metric-depth relation margin: `0.020 m`.
- Candidate-layout screening buffer: `0.035 m`; this is stricter than the
  runtime decision margin to absorb predictor error.

The margins are fixed before the repair pilot is captured. They may not be
tuned on repair-pilot outcomes. A future margin change requires another
versioned amendment and a new pilot.

### Relation predicates

Let `x_t,z_t` be target evidence and `x_a,z_a` anchor evidence.

- `right_of`: `x_t > x_a + 12`.
- `left_of`: `x_t + 12 < x_a`.
- `front_of` or `nearer_than`: `z_t + 0.020 < z_a`.
- `behind` or `farther_than`: `z_t > z_a + 0.020`.
- `between_in_depth`: with two anchors, `min(z_a)+0.020 < z_t < max(z_a)-0.020`.
- `nearer_than_both`: `z_t + 0.020 < z_a` for both anchors.
- `farther_than_both`: `z_t > z_a + 0.020` for both anchors.
- `direct`: no relation predicate.

An expected positive relation must satisfy its predicate. An
`unsatisfied_relation` negative must fail it while target and anchors remain
visible. Invalid or insufficient masks/depth are `INSUFFICIENT_EVIDENCE`, not
positive or negative relation proof.

## Camera-frame layout predictor

The generator projects each object pose into `camera_color_optical_frame` using
the locked wrist-camera extrinsic. It predicts visible median depth by
subtracting an asset-specific surface offset learned only from the quarantined
230-capture diagnostic set. Each calibration row records count, median offset,
MAD and residual quantiles.

The predictor is a rejection filter, not a label source:

- candidates with missing calibration are rejected;
- candidates must satisfy the requested predicate with the 0.035 m screening
  buffer;
- workspace, footprint collision and visibility-envelope checks still apply;
- final labels come only from the real synchronized repair/official capture.

The former IID wrist pose is retained only as diagnostic provenance. Static FK
feasibility analysis performed before repair-pilot materialization showed that
its optical axis was too close to vertical to place the tall mustard bottle
behind the shorter orange by the 0.035 m screening margin while keeping both
instances inside the conservative image envelope. The repair pilot therefore
uses the independently locked `camera_v2_1_relation` pose in
`dataset_v2_relation_camera_lock_v2_1.json`. This selection used FK,
calibration from the quarantined diagnostic captures and geometry filters only;
no repair-pilot RGB/depth image had been captured or inspected.

## Deterministic candidate policy

1. Lock a fresh V2.1 master seed and a finite candidate seed pool.
2. Generate candidates without reading model predictions.
3. Rank candidates by SHA-256 of protocol namespace, relation and candidate
   seed.
4. Apply only the predeclared camera-frame predictor, collision and visibility
   filters.
5. Select the first passing candidate for each required quota cell.
6. Record every accepted/rejected candidate and reason.

No layout may be hand-picked after viewing its RGB/depth result. Repair-pilot
capture results may decide only `PASS` or `REVISE_PROTOCOL`; they cannot be
silently relabelled.

## Repair pilot

- Role: `REPAIR_PILOT_ENGINEERING_ONLY_NOT_OFFICIAL_DATA`.
- Size: 30 independent families and 30 clean Gazebo captures.
- Fresh namespace: `v21repair_family_*`.
- Required relations: `front_of`, `behind`, `nearer_than`, `farther_than`,
  `between_in_depth`, `nearer_than_both`, `left_of`, `right_of`.
- Must include mustard bottle/orange in both depth orders and tomato-soup-can
  multi-anchor cases.
- Must cover tall, low, round and box-like assets.
- Every family is excluded from official train/dev/calibration/test.

The repair pilot passes only if all 30 captures are structurally valid and all
locked relation predicates match their expected positive label. No waiver,
post-hoc margin tuning or partial-pass rule is allowed.

The repair pilot reuses the already qualified Gazebo, MoveIt camera-motion and
RGB-D/semantic capture stack. No shared world, robot description, controller or
YCB asset is changed by this amendment.

## Development V2.1 authorization

Only a passing repair pilot may authorize generation and static preflight of a
fresh 400-family development V2.1 manifest. V2.1 must preserve the previously
locked 320/80 split and four-state quotas, but use new IDs, seeds, instructions
and layouts disjoint from the pilot, repair pilot and failed V2 run.

`GO_DEVELOPMENT_TRAIN` remains impossible until all 400 V2.1 families are
captured, materialized to 2,000 samples and pass full QC.
