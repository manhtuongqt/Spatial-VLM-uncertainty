# P-CRA-U development failure audit and architecture decision contract

## 1. Scope and scientific status

This contract authorizes one read-only evaluation of the selected development
checkpoint. It does not authorize training, calibration fitting, threshold
selection, Test-IID access or Test-OOD access. Every result produced here is
`EXPLORATORY_DEV_ONLY`.

The evaluation cohort is exactly the locked V2.1.1 development split:

- 80 independent scene-query families;
- 400 records, with all five variants of every family kept together;
- no train record is an evaluation observation;
- no calibration or test record may be discovered, loaded or materialized.

The independent statistical unit is the family, never an individual variant.
The selected P-CRA-U checkpoint is immutable:

- path:
  `results/pcra_u_runs/pcra_u_development_train_20260824/checkpoints/development/step_000002000`;
- epoch/global step: `10 / 2000`;
- model SHA-256:
  `14dba40d2de437f082aece9b342931a48c58202ae4242d577d4a4ffd7eecaab3`;
- checkpoint-manifest SHA-256:
  `b59f742727d35cdbb2c9575571d89d2b5853f903dcd5b1358014d7d977ec64a5`.

The checkpoint, frozen RoboRefer baseline, development dataset and feature cache
must have the same hashes before and after the audit. No optimizer, scheduler,
gradient update or checkpoint write is permitted.

## 2. Locked systems and interventions

All systems use the same locked dev records and evaluator labels. Evaluator
masks, object identities and relation labels are metric-only data and must never
enter an inference payload.

### B0 — real RoboRefer RGB

Run the existing RoboRefer RGB baseline through its real inference path. A
cached prediction is acceptable only when its input, configuration, checkpoint
and output hashes resolve to that same real run. A proxy, heuristic or copied
P-CRA-U output may not be reported as B0.

The dataset prompt's final pixel-coordinate sentence is an interface suffix,
not scene language. For B0/B1 only, replace that exact suffix with RoboRefer's
locked normalized-point suffix (`[(x, y)]`, both values in `[0,1]`, no extra
words). Preserve every preceding instruction byte and use the same adapted
prompt for B0 and B1. Store hashes of both source and adapted prompts. This
adapter is required because the frozen RoboRefer output/parser contract is
normalized coordinates; normalized output must not be silently interpreted as
pixel coordinates. Some B0 instructions still mention RGB-D/depth in their
scene wording; retain that wording for a paired modality-only comparison and
report it as a limitation.

### B1 — real RoboRefer RGB-D

Run the existing RoboRefer RGB-D baseline through its real, unmodified
inference path. The same provenance requirements as B0 apply. RoboRefer source,
weights and `generate()` behavior remain unchanged.

### B2 — B1 plus the existing geometry gate

B2 is B1 followed by the already implemented `depth_component_gate_v2`. The
primary B2 analysis is restricted to the 80 clean dev captures, because this is
the locked domain of the observed-geometry gate. Results on counterfactual
variants, if emitted for debugging, are secondary and must not be mixed into the
primary B2 estimate.

### P1 — selected P-CRA-U development checkpoint

P1 is the unmodified checkpoint at `step_000002000`, with its trained RGB,
depth, language and relation-conditioned inputs. P1 is the reference system for
the two paired post-hoc interventions below.

### U1-posthoc — neutral structured relation ID

U1-posthoc uses the exact P1 weights and inputs, except that the structured
relation-ID input is deterministically replaced by `direct`. The original prompt
tokens and their lexical language features are retained. Consequently this is
not a retrained "heatmap without relation" model and not complete removal of
relation information; it is a diagnostic intervention on the explicit relation
conditioning channel only.

### P1-no-depth-posthoc — zeroed depth features

P1-no-depth-posthoc uses the exact P1 weights and inputs, except that `D0` and
`d_thumb` are replaced by exact zero tensors immediately before the P-CRA-U
forward pass. RGB, prompt tokens and structured relation ID remain unchanged.
This is a diagnostic intervention, not a retrained RGB-only causal ablation.

The existing four-source head (`semantic`, `relation`, `depth`, `occlusion`) is
evaluated descriptively. No `spatial` source label or fifth source output may be
invented.

## 3. Required failure audit

The audit must account for every P1 error previously summarized as:

- 9 false-`FOUND` predictions among ground-truth `AMBIGUOUS` or `ABSENT`
  records;
- 24 point-outside-target errors among 168 ground-truth `FOUND` records.

Machine-readable rows must identify family and variant and stratify errors by:

- relation, including explicit reporting for `right_of` and
  `between_in_depth`;
- variant type;
- target and anchor asset;
- target-mask area and interior-mask area;
- robust target/anchor metric depth and locked depth margin when available;
- occlusion/visibility evidence.

All denominators and missing/invalid observations must be reported. Subgroups
with small family counts are diagnostic and cannot support a standalone claim.
Any visual evidence must be derived from real dev RGB/depth/mask/prediction
artifacts; decorative or planned infographics are forbidden.

## 4. Metrics and family-cluster inference

The primary answerability failure rate is

`false-FOUND / (ground-truth AMBIGUOUS + ground-truth ABSENT)`.

The primary grounding metric is family-equal point-in-target on ground-truth
`FOUND`: compute the mean eligible point-in-target indicator inside each family,
then average the family values. Answerability macro-F1 is recomputed from all
records belonging to the sampled family clusters.

Confidence intervals use a family-cluster bootstrap with these immutable
settings:

- resampling unit: one of the 80 dev families, retaining all five variants;
- replicates: `5000`;
- seed: `24082027`;
- seed application: reinitialize that exact seed for every method and paired
  contrast; no hidden seed offsets or derived seeds are permitted;
- interval: two-sided percentile 95%, using the 2.5th and 97.5th percentiles;
- paired comparisons: the identical resampled family indices are used for P1
  and its intervention.

Define paired deltas so that positive values favor the current P1 architecture:

- relation delta: `metric(P1) - metric(U1-posthoc)`;
- depth delta: `metric(P1) - metric(P1-no-depth-posthoc)`.

For each intervention, report paired deltas and 95% confidence intervals for
both family point-in-target and answerability macro-F1. Report point estimates
and intervals even when they do not pass a decision threshold.

## 5. Integrity and sealed-split gate

Before an architecture decision, all of the following must pass:

1. The evaluation manifest resolves exactly 80 dev families and 400 dev
   records, with five records per family and no other split.
2. B0 and B1 outputs have real RoboRefer run provenance; B2 records the exact
   `depth_component_gate_v2` identity and uses clean-only data for its primary
   result.
3. P1, U1-posthoc and P1-no-depth-posthoc use the same immutable selected
   checkpoint and deterministic record order.
4. Calibration fitting is `false`; calibration, Test-IID and Test-OOD access is
   `false`; no threshold is selected from dev for a final test claim.
5. No optimizer step, parameter mutation or new training checkpoint occurs.
6. Dataset, feature-cache, checkpoint and protected RoboRefer/WP3 hashes remain
   unchanged.
7. Predictions and bootstrap outputs are finite, complete and reproducible from
   the locked manifest, seed and code hash.
8. Inference payloads remain oracle-free; supervision appears only in metric
   evaluation.

Failure of any integrity or sealed-split condition forces
`ONE_CONTROLLED_DEVELOPMENT_REVISION`; it can never be waived by a favorable
metric.

## 6. Locked architecture decision rule

The decision is `FREEZE_PCRA_U_ARCHITECTURE_AND_CHECKPOINT` if and only if every
condition below is true:

1. all integrity and sealed-split gates in Section 5 pass;
2. P1 false-`FOUND` point estimate is at most `0.10`;
3. the lower endpoint of P1's family-cluster 95% confidence interval for
   point-in-target is at least `0.70`;
4. relation non-degradation passes: the lower 95% confidence bound is greater
   than `-0.02` for **either** the paired relation delta in point-in-target or
   the paired relation delta in macro-F1;
5. depth non-degradation passes: the lower 95% confidence bound is greater than
   `-0.02` for **either** the paired depth delta in point-in-target or the paired
   depth delta in macro-F1.

The strict comparison is `> -0.02`, not greater-than-or-equal. The two
interventions are judged separately; passing relation cannot compensate for
failing depth, or vice versa.

If any condition fails, the only allowed decision is
`ONE_CONTROLLED_DEVELOPMENT_REVISION`. That decision authorizes one V1.1 change
targeted only at a failure demonstrated by this audit, followed by one locked
train/dev run. It does not authorize repeated dev tuning or access to
Calibration/Test.

## 7. Reporting restrictions and next actions

The report must label B0/B1/B2/U1-posthoc/P1 comparisons, post-hoc interventions,
source-head results and confidence intervals as `EXPLORATORY_DEV_ONLY`.
U1-posthoc and P1-no-depth-posthoc must be described as post-hoc diagnostic
interventions, never as retrained causal ablations.

Tables 2–6 remain `NOT_RUN`; this audit must use separately named
`table_dev_*_exploratory` artifacts and must not overwrite the locked final-table
scaffold.

After a freeze decision, the next allowed step is capture of a separately locked
Calibration split. Fitting the calibrator/threshold follows that capture;
Test-IID and Test-OOD remain sealed until architecture, checkpoint, calibrator
and threshold are all frozen. After a revision decision, the next allowed step
is the single controlled V1.1 development revision, while Calibration and both
test splits remain sealed.
