# Dataset V2.1 Development Capture Contract

## Decision and scope

- Gate: `LOCK_DEVELOPMENT_V2_1_CAPTURE_400`.
- Protocol: `roborefer_dataset_v2_1_development_capture_400`.
- This lock materializes and statically validates the design only.
- It does **not** start Gazebo, capture data, materialize samples, train a model,
  create calibration data, or create/open Test-IID/Test-OOD.
- A static PASS authorizes only `canary_000` as the next action.

## Locked population

| Split | Families | FOUND | INSUFFICIENT_EVIDENCE | AMBIGUOUS | ABSENT |
|---|---:|---:|---:|---:|---:|
| Train | 320 | 128 | 80 | 56 | 56 |
| Dev | 80 | 32 | 20 | 14 | 14 |
| Total | 400 | 160 | 100 | 70 | 70 |

Each family owns five variants and exactly two planned physical captures
(`clean`, `occlusion`): 2,000 future samples and 800 future RGB-D captures.
Family is the independent unit; every variant remains in its family split.

## Fresh-identity boundary

- Family namespace: `v21dev_family_000001` … `v21dev_family_000400`.
- Family IDs, capture IDs, all seed values, exact instructions and active-layout
  fingerprints must be disjoint from Dataset V2, the 30-family pilot, both
  relation-repair pilots, and the shutdown canary.
- Dataset V2 is used only as a locked quota/composition template. No V2 capture
  may be copied into the V2.1 official development root.
- Output root is new and must be absent or empty at static lock:
  `datasets/roborefer_dataset_v2_1_development_400_20260824`.

## Split and asset boundary

- Only `train` and `dev` are allowed; split assignment is by family.
- Only the locked seen pool may be active in a scene.
- Apple/orange/banana clones inherit the seen status of their locked base asset.
- `ycb_pear_01`, `ycb_plum_01`, and `ycb_tuna_fish_can_01` remain Test-OOD-only
  and must stay in storage in every development layout.
- `ycb_bleach_cleanser_01`, `cup`, and `kettle` remain excluded.

## Observable relation geometry V2.1

- `left/right`: centroid of the semantic instance mask in the image.
- `front/behind` and `near/far`: robust median metric depth within the eroded
  instance mask.
- `between_in_depth`: target median depth is between both anchors with the
  locked margin.
- Runtime labels use synchronized clean captures with valid masks/depth only.
- Simulator centers are permitted only for deterministic static screening.
- Static screening uses geometry
  `dataset_v2_relation_geometry_v2_1.1.0`, 0.035 m general depth buffer,
  0.070 m `between_in_depth` buffer, and 24 px projected horizontal buffer.
- The passed relation-repair gate is a prerequisite; no post-capture manual
  layout selection is permitted.

## Batch, resume and shutdown contract

Execution order is fixed:

1. `canary_000`: 20 families / 40 captures (16 train, 4 dev; all four states).
2. `batch_001`: 95 families / 190 captures.
3. `batch_002`: 95 families / 190 captures.
4. `batch_003`: 95 families / 190 captures.
5. `batch_004`: 95 families / 190 captures.

Each later batch requires a PASS checkpoint for the immediately preceding
batch. Resume identity is `capture_id`; complete captures are hash-verified and
skipped, while partial/corrupt captures cause a hard failure and are never
overwritten.

The launcher must use the qualified wrapper-only shutdown sequence:

`capture node → action server/Servo → move_group → Gazebo`.

The shutdown coordinator is
`protocol/dataset_v2_relation_repair_shutdown.py`. The baseline launch files,
URDF, controller configuration and Gazebo world are not modified.

## Gate decisions

Static preflight checks exact counts, family cohesion, fresh identities/seeds/
instructions/layouts, seen-only assets, V2.1 predictor geometry, oracle-free
inference payload, prerequisite gate hashes, output-root cleanliness and
baseline hashes.

- PASS: `GO_DEVELOPMENT_V2_1_CAPTURE_400`, with `canary_000` as the only next
  batch until its raw QC passes.
- FAIL: `FIX_V2_1_DEVELOPMENT_LOCK_FIRST`.

Training remains forbidden until all 400 families produce 800 valid captures,
materialize into 2,000 samples, and full Dataset V2.1 QC passes.
