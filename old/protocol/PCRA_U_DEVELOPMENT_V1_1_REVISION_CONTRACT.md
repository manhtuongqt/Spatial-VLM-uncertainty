# P-CRA-U development V1.1 — locked revision contract

## 1. Scope and scientific status

V1.1 is an add-only development branch authorized to test one predeclared
regularized objective. It does not alter or resume the V1 sidecar, RoboRefer,
the 400-family development dataset, its family-preserving train/dev split, or
the frozen pooled feature cache. Every V1.1 seed starts from a fresh sidecar
initialization.

All V1.1 measurements remain `EXPLORATORY_DEV_ONLY`. The same 80 development
families have already informed the revision, so neither checkpoint selection
nor the acceptance audit is an independent generalization result. Test-IID and
Test-OOD remain sealed. The V1 calibration run and calibrator are historical V1
diagnostics only: V1.1 code must not read them for optimization, checkpoint or
seed selection, acceptance, calibration, or threshold choice. If V1.1 is
accepted, it requires a newly collected independent calibration set before any
test is opened.

## 2. Immutable parent boundary

The exact parent paths and SHA-256 digests are recorded in
`pcra_u_development_v1_1_manifest.json`. Preflight must fail closed if any
parent artifact differs. In particular:

- the train/dev unit remains family, with 320/80 disjoint families and
  1,600/400 records;
- the V1 supervision manifest and V1 full feature-cache index are reused
  byte-for-byte;
- RoboRefer stays frozen and outside every optimizer;
- V1 checkpoints, predictions, reports, architecture lock, and calibrator lock
  remain immutable historical artifacts;
- no calibration/test record, prediction, metric, threshold, or label is a
  V1.1 model-development input.

## 3. Single locked V1.1 intervention

The architecture width, input boundary, tokenizer, relation parser, heads,
class definitions, and top-level loss weights remain V1-compatible. The locked
revision consists only of:

1. dropout `p=0.10` after the hidden GELU in the target, answerability, and
   source heads;
2. AdamW learning rate `1e-4` and weight decay `1e-3`;
3. a fourfold heatmap weight for ground-truth `FOUND` records whose variant is
   exactly `clean`.

For every ground-truth `FOUND` sample (i), let

`ell_i = weighted_BCE_i + soft_Dice_i`

and let `w_i=4` for `variant == clean`, otherwise `w_i=1`. The batch heatmap
objective is normalized, not rescaled:

`L_heatmap = sum_i(w_i * ell_i) / sum_i(w_i)`.

If a batch contains no `FOUND`, `L_heatmap` is the differentiable zero used by
V1. The complete objective stays

`L = 1.0 L_heatmap + 0.5 L_answerability + 0.5 L_source`.

The train split contains 128 clean-FOUND and 545 other-FOUND records. The
locked multiplier therefore assigns effective aggregate weights 512 versus
545, approximately balancing the clean and non-clean FOUND strata without
changing the nominal scale of the heatmap mean. No multiplier sweep is
permitted after observing V1.1 dev results.

Weighted objective loss and the ordinary unweighted dev heatmap diagnostic
must be named separately. A weighted loss may be aggregated only by its sum of
sample weights; multiplying a weighted batch mean by the raw FOUND count is
invalid. Total dev loss is diagnostic and never selects a V1.1 checkpoint.

## 4. Optimization and three-seed replication

The independent seeds are `24082026`, `24082027`, and `24082028`. Each seed
controls fresh parameter initialization, dropout randomness, and deterministic
epoch order. Resume or initialization from any V1/V1.1 checkpoint is forbidden.

All seeds run exactly 20 epochs, batch size 8, 200 warmup steps followed by the
V1 cosine schedule, and gradient clipping at 5.0. No seed may stop early. Every
epoch is evaluated and saved immutably. The three seeds are replications, not a
hyperparameter search; all three results must be reported, including failures.
The canary root is
`results/pcra_u_runs/pcra_u_development_v1_1_canary_20260824`; the campaign
root is `results/pcra_u_runs/pcra_u_development_v1_1_campaign_20260824`; and
each immutable seed run uses the campaign-root name suffixed by its locked
`_seed_<seed>` identifier. Existing paths must never be overwritten.

## 5. Checkpoint metrics and selection

Metrics use all 80 dev families and fixed, uncalibrated model outputs:

- `Gc`: point-in-target on the 32 ground-truth clean-FOUND records, regardless
  of the answer-head prediction;
- `Ga`: point-in-target on all 168 ground-truth FOUND records;
- `A`: four-class answerability macro-F1 over all 400 dev records;
- `F`: predicted `FOUND` rate on the 95 ground-truth
  `AMBIGUOUS union ABSENT` records.

The operational clean-FOUND joint success
`prediction == FOUND and point-in-target` must also be logged, but it is a
secondary diagnostic and does not replace the locked score.

A checkpoint is eligible only when all four conditions hold:

- epoch is at least 6;
- `Ga >= 0.8371428571`;
- `A >= 0.7220238435`;
- `F <= 0.1147368421`.

The latter three are the predeclared V1 non-inferiority bounds: respectively
V1 minus 0.02, V1 minus 0.02, and V1 plus 0.02. Among eligible checkpoints,
maximize

`C = 0.45 Gc + 0.20 Ga + 0.20 A + 0.15 (1 - F)`.

On an exact score tie, prefer lower `F`, then higher `Gc`, then the earlier
epoch. The checkpoint manager must receive `eligible_for_best=false` for an
ineligible epoch. A seed with no eligible checkpoint fails; thresholds must not
be relaxed post hoc. Selection reads dev only and must use the pointer
`best_dev_scientific_score_v1_1.json`, never V1's
`best_dev_total_loss.json`.

## 6. Acceptance policy

V1.1 is accepted only if every condition below passes without post-hoc repair:

1. all three seeds produce an eligible, hash-audited selected checkpoint;
2. every selected seed has `Gc >= 0.50`;
3. at least two selected seeds have `Gc > 0.53125`, the V1 value `17/32`;
4. after conditions 1–3, a locked 10,000-replicate paired family-cluster
   bootstrap (seed `24082029`) resamples the 80 family IDs and keeps all variants
   together. For each resample, V1.1 is the mean metric across the three selected
   seeds and is paired against the immutable V1 epoch-10 predictions;
5. every delta uses the conventional direction `V1.1 - V1`; the lower endpoint
   (2.5th percentile) of the two-sided 95% percentile interval for `delta Gc`
   is greater than 0;
6. the corresponding two-sided 95% lower endpoints for `delta Ga` and
   `delta A` are at least `-0.02`, and the two-sided 95% upper endpoint (97.5th
   percentile) for `delta F` is at most `+0.02`;
7. the across-seed point estimate `delta F` is strictly below 0 (equivalently,
   V1.1 `F` is strictly below V1's `0.0947368421`).

Bootstrap inference is conditional on the three selected development
checkpoints and is still exploratory because the development set was reused.
Report per-seed values, mean, standard deviation, selected epoch/hash, all
eligibility failures, paired deltas, and bootstrap bounds. Reporting only the
best seed is forbidden.

## 7. Preflight and execution lock

`pcra_u_development_v1_1_preflight.py` validates the locked configuration,
manifest counts, parent hashes, implementation presence, cache identity,
historical V1 reference values, and the calibration/test boundary. It may write
`protocol/pcra_u_development_v1_1_preflight_lock.json` only after every check
passes. The lock hashes the exact design and implementation files used by the
run. A failed preflight authorizes no optimizer step and must leave no new lock.
