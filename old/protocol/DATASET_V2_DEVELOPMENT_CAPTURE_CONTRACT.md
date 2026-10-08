# Dataset V2 development capture contract

## Status and scientific boundary

- Protocol: `roborefer_dataset_v2_development_capture_400`
- Gate: `GO_DEVELOPMENT_CAPTURE_400`
- Scope: lock and capture the official development-only `train` and `dev` data.
- Independent unit: one `scene_query_family`; five variants from a family are dependent observations.
- Calibration, Test-IID, Test-OOD and all training are forbidden until the completed 400-family dataset passes materialization and QC.
- The 30-family pilot remains `PILOT_ENGINEERING_QC_ONLY_NOT_OFFICIAL_DATA` and is permanently excluded.

This contract authorizes capture only after static preflight passes. Static planned counts are not observed dataset evidence and must be labelled `PLANNED_NOT_CAPTURED` until real Gazebo capture and QC finish.

## Locked composition

| Split | Independent families | `FOUND` | `INSUFFICIENT_EVIDENCE` | `AMBIGUOUS` | `ABSENT` |
|---|---:|---:|---:|---:|---:|
| `train` | 320 | 128 | 80 | 56 | 56 |
| `dev` | 80 | 32 | 20 | 14 | 14 |

Every family has exactly these dependent variants:

1. `clean`
2. `semantic_counterfactual`
3. `relation_counterfactual`
4. `depth_corruption`
5. `occlusion_view_counterfactual`

The locked totals are 400 independent families, 2,000 dependent samples and two raw observations per family (`clean_capture`, `occlusion_capture`), hence 800 planned Gazebo RGB-D captures.

The assignment list must reproduce commitment `d6790084dab886bbca226c892256e0ee4ca86f9a51c6970b9c2c03b99bcbc2a2` from `dataset_expansion_v2_seed_lock.json`.

## Answerability and submodes

- `FOUND`: exactly one valid visible referent; intervention `EXECUTE`.
- `AMBIGUOUS`: at least two valid referents; include same-class duplicate, attribute tie, relation tie and multi-anchor conflict; intervention `ASK_USER`.
- `ABSENT`: zero valid referents; include target absent, anchor absent and visible-but-unsatisfied relation; intervention `ABSTAIN`.
- `INSUFFICIENT_EVIDENCE`: one latent referent with inadequate observation; include invalid/corrupt depth, occlusion, too-small/out-of-view and cross-modal conflict; intervention `REOBSERVE`.

Primary submode counts across train and dev must match `dataset_expansion_v2_spec.json`. A relation-negative is not allowed to rely on accidental invisibility: candidates and anchors for `unsatisfied_relation` must remain in the calibrated visibility envelope.

## Independence, leakage and asset partition

- All five variants and both captures of a family stay in the same split.
- No family ID, sample ID, capture ID, seed value, exact instruction, language-template family ID or exact active layout may overlap the pilot.
- No family ID, capture ID, layout seed, exact instruction or language-template family ID may cross train/dev.
- Pilot family IDs, seeds, captures, layouts and instructions are deny-listed inputs, never training data.
- Train/dev may use only the seen pool and clone instances declared by the locked clone policy.
- `ycb_pear_01`, `ycb_plum_01` and `ycb_tuna_fish_can_01` remain Test-OOD-only and must not be active targets, anchors, counterfactual objects or distractors in train/dev.
- `cup`, `kettle` and `ycb_bleach_cleanser_01` remain excluded.
- `cube_red_01` remains a runtime fixture attached to the gripper and cannot be queried or active task evidence.
- Inference payloads contain only RGB, model-input depth, prompt/config and coordinate convention. Mask paths, semantic labels, object IDs, evaluator graphs, answerability labels and target relations are forbidden.

## Batch execution lock

Capture order is fixed to five sequential gates:

| Order | Batch | Families | Planned captures |
|---:|---|---:|---:|
| 0 | `canary_000` | 20 | 40 |
| 1 | `batch_001` | 95 | 190 |
| 2 | `batch_002` | 95 | 190 |
| 3 | `batch_003` | 95 | 190 |
| 4 | `batch_004` | 95 | 190 |

The canary must contain both splits, all four primary answerability states and all six family categories. A later batch is forbidden until the previous batch checkpoint passes raw-capture QC.

Capture is resume-safe and create-only:

- identity key is `capture_id`;
- a missing capture is written to a temporary directory and atomically renamed;
- an existing capture is skipped only after every declared artifact hash is verified;
- a partial/corrupt existing capture causes failure and is never overwritten automatically;
- progress is checkpointed atomically after every completed family and at each batch boundary;
- raw manifest completeness is evaluated against 800, never against variants.

## Static preflight

Gazebo remains forbidden until `dataset_v2_development_preflight.py` reports `PASS` and the execution lock decides `GO_DEVELOPMENT_CAPTURE_400`.

Preflight must verify:

- deterministic generator replay and the locked assignment commitment;
- exact split/state/category/submode/family/sample/capture counts;
- five-variant family cohesion and batch composition `20 + 95 × 4`;
- world-model/semantic-label uniqueness, workspace bounds, target-bin clearance and pairwise collision margins;
- target/anchor membership, answerability cardinality and relation geometry;
- required evidence inside the calibrated visibility envelope;
- seen-only train/dev asset use and held-out/excluded/runtime-fixture rejection;
- pilot/development and train/dev disjointness for IDs, seeds, instructions, template families and layout fingerprints;
- oracle-free inference payload template;
- pilot gate PASS and unchanged pilot/WP0-WP3/WP2 hashes;
- no calibration or test assignment/capture payload.

The execution lock hashes the contract, generator, manifests, preflight source/report, capture runtime, design artifacts, pilot gate artifacts, Gazebo world and camera description. Any later source change invalidates capture authorization.

## After real capture

After all five batches, materialization and QC must independently verify exactly 2,000 records, family-level split isolation, ontology/masks, registered RGB-depth-label timestamps, metric relations, deterministic replay and unchanged protected hashes.

Only after that QC may the next decision be `GO_DEVELOPMENT_TRAIN`. Until then:

- training is `NOT_AUTHORIZED`;
- Calibration/Test-IID/Test-OOD are `SEALED_NOT_CREATED`;
- Dataset Table 1 remains planned until observed QC replaces it;
- Tables 2-6 remain `NOT_RUN`;
- no planned infographic or decorative figure may be produced.
