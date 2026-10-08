# Dataset V2 pilot-only execution contract

## Status and scope

- Protocol: `roborefer_dataset_v2_pilot_only`
- Gate: `GO_DRY_RUN_30_PILOT_ONLY`
- Scientific status: engineering/QC evidence only; never eligible for the official dataset.
- Training, calibration, Test-IID and Test-OOD are forbidden in this gate.
- The WP2 generator and all locked WP0-WP3 artifacts are read-only protected inputs.

This gate tests the Dataset V2 generator and Gazebo capture path before any 400-family development capture. A pilot family ID, seed, raw capture, layout, instruction, paraphrase or derived sample must never be reused in an official split.

## Locked experimental unit

The independent unit is one `scene_query_family`. Each family has five dependent variants:

1. `clean`
2. `semantic_counterfactual`
3. `relation_counterfactual`
4. `depth_corruption`
5. `occlusion_view_counterfactual`

All variants stay in `pilot_only`. Counts of variants must not be reported as independent observations.

## Locked pilot composition

The manifest shall contain exactly 30 families and 150 samples. Primary clean-query strata are exactly:

| State | Families |
|---|---:|
| `FOUND` | 8 |
| `INSUFFICIENT_EVIDENCE` | 8 |
| `AMBIGUOUS` | 7 |
| `ABSENT` | 7 |

Each of the six family categories has exactly five families. Each OOD axis (`none`, `asset`, `layout`, `viewpoint`, `language`, `depth_noise`) has exactly five families. These assignments must reproduce commitment `45f0ec073d1b71d3e4a07336262c98e5ef875ac5befd2a585b51bdb16b3cb22c` from `dataset_expansion_v2_seed_lock.json`.

## Ontology rules

- `FOUND`: exactly one valid referent and sufficient visible RGB-D evidence.
- `AMBIGUOUS`: at least two valid referents. Required pilot submodes are same-class duplicate, attribute tie, relation tie and multi-anchor conflict.
- `ABSENT`: zero valid referents. Required pilot submodes are target absent from the active task scene, required anchor absent, and an unsatisfied locked relation.
- `INSUFFICIENT_EVIDENCE`: exactly one latent valid referent but insufficient current evidence. Required pilot submodes are invalid/corrupt depth, occlusion, too-small/out-of-view and cross-modal conflict.

`candidate_target_ids` and oracle scene membership are evaluator-only. A target mask is the union of visible pixels belonging to `valid_target_ids`; consequently an `ABSENT` sample has an empty target mask even when a visible semantic candidate violates the relation.

## Static preflight gate

Gazebo capture is forbidden until `dataset_v2_pilot_preflight.py` reports `PASS` and writes `dataset_v2_pilot_execution_lock.json` with decision `GO_PILOT_GAZEBO_CAPTURE_60`.

Preflight must verify:

- exact locked counts and assignment commitment;
- all model names, instance labels and scene IDs are unique;
- active-layout workspace bounds and pairwise collision margins;
- target/anchor scene membership and ontology cardinality;
- locked relation geometry in base coordinates;
- single-axis OOD isolation and asset partition;
- no excluded asset (`cup`, `kettle`, bleach cleanser);
- no evaluator/oracle key in an inference payload template;
- pilot family IDs, seeds and capture IDs are disjoint from the reserved 400-family development assignment;
- all protected file and tree hashes remain unchanged.

The execution lock hashes the contract, generator, manifest, capture plan, preflight source/report, design artifacts, Gazebo world and camera description. Any hash change invalidates capture authorization.

## Real Gazebo capture

After static PASS, capture exactly two registered observations per family:

- `clean_capture`
- `occlusion_capture`

The expected total is 60 raw captures. Each capture must atomically store:

- original RGB PNG;
- metric depth `float32` NPY in metres;
- semantic-instance label PNG;
- camera intrinsics;
- TF snapshot;
- RGB/depth/label timestamps and maximum synchronization spread;
- requested layout and realized joint camera pose;
- SHA-256 for every raw artifact.

The capture process consumes only the locked plan. It must not create a RoboRefer request and must not publish a target or manipulation command. The `asset` pilot axis may use pear, plum or tuna fish can, but every resulting artifact remains pilot-only.

## Materialization and QC gate

Materialization must create exactly 150 schema-valid records from the 60 raw captures. QC must verify:

- all four ontology states and required negative submodes;
- binary target/interior/anchor/graspable/reachable/valid-depth masks and subset invariants;
- synchronized RGB, metric depth and semantic-label shapes/timestamps;
- projective 2D relations and metric-depth relations against captured evidence;
- semantic label visibility consistent with evaluator scene membership;
- deterministic replay of all derived records and files;
- no oracle leakage into inference payloads;
- pilot/development ID, seed and capture non-overlap;
- unchanged WP0-WP3, WP2 dataset and protected result-tree hashes.

The only allowed visual evidence is deterministic QC output generated from real captures: selected RGB, metric-depth visualization, semantic-instance labels, target/anchor masks and optional overlays. Planned infographics, decorative figures and manually drawn GUI-style evidence are forbidden.

## Decision

- PASS: `GO_DEVELOPMENT_CAPTURE_400`
- FAIL: `FIX_DATASET_V2_GENERATOR_OR_CAPTURE_PIPELINE_FIRST`

At this gate only Dataset Table 1 and a real-capture dataset QC figure may be updated. Tables 2-6 remain `NOT_RUN`.
