# P-CRA-U Calibration Contract v1

Status: `LOCK_BEFORE_CALIBRATION_CAPTURE_AND_FIT`  
Date: `2026-08-24`  
Allowed upstream decision: `FREEZE_PCRA_U_ARCHITECTURE_AND_CHECKPOINT`  
Allowed next upstream action: `CAPTURE_CALIBRATION`

## 1. Scope and immutable model

This protocol captures an independent IID Calibration split, runs only the
frozen P-CRA-U checkpoint at epoch 10 / step 2,000, fits lightweight risk
calibrators, and freezes a `FOUND/ABSTAIN` threshold before any locked test is
opened.

- P-CRA-U architecture and model weights are immutable.
- RoboRefer, URDF, controllers, Gazebo world and baseline launch files are
  immutable.
- Calibration may fit calibrator parameters only; it may not select an
  architecture, loss, model checkpoint or feature extractor.
- Test-IID and Test-OOD remain absent and unreadable throughout this protocol.

## 2. Calibration dataset commitment

Calibration contains 200 independent scene-query families, five dependent
variants per family and two synchronized physical captures per family:

| Primary family stratum | Families |
|---|---:|
| FOUND | 80 |
| INSUFFICIENT_EVIDENCE | 50 |
| AMBIGUOUS | 35 |
| ABSENT | 35 |
| **Total** | **200** |

Thus `AMBIGUOUS ∪ ABSENT = 70` independent families, above the previously
locked minimum of 60. The category quota is exactly half of the 400-family
development composition: direct grounding 30, 2D relation 40, front/behind
30, nearer/farther 35, multi-anchor depth 30 and occlusion/depth evidence 35.

- Split is exactly `calibration`; no train/dev/test assignment is created.
- Capture batches are `20 + 90 + 90` families. A batch is forbidden until the
  previous batch's observed RGB-D/semantic QC passes.
- Assets are seen/IID only. `pear`, `plum` and `tuna_fish_can` stay Test-OOD
  only.
- Family ID, capture ID, every seed, layout fingerprint, raw capture, exact
  instruction and language-template family ID must be new and disjoint from
  WP2, pilot, repair, shutdown, V2 and official V2.1.1 development data.
- Clean, counterfactual and view variants of one family stay together.
- Only real RGB/depth/semantic/mask QC images selected by a locked hash rank
  may be published; no planned infographic is evidence.

## 3. Capture and materialization gates

Every raw capture must contain synchronized RGB, metric depth,
semantic-instance label, camera information, TF and timestamps. Observable
relations use semantic-mask centroids for left/right and robust median metric
depth for front/behind, near/far and multi-anchor depth ordering. Simulator
object centers are secondary static screening only.

Full QC requires exactly 200 families, 400 complete captures and 1,000 records;
valid masks; deterministic replay; no duplicate/leakage; seen-asset compliance;
oracle-free inference payloads; and unchanged upstream locks. Failure stops the
protocol before feature extraction.

## 4. Strict inference/evaluator separation

Raw frozen-checkpoint inference reads only RGB/depth-derived feature tensors,
prompt tokens and prompt-parsed relation IDs. It must persist prediction logits
and observable evidence before any evaluator mask or label is opened. The raw
inference attestation must state `oracle_or_annotation_read=false`.

Only the separate fit/scoring phase may join evaluator labels to the immutable
raw predictions. No gradient is permitted through P-CRA-U and no optimizer may
contain P-CRA-U parameters.

## 5. Primary calibration event and decision

For each variant:

```text
selected_grounding_correct =
    (ground-truth state == FOUND) AND (P-CRA-U MAP point is inside target mask)

y_error = 1 - selected_grounding_correct

decision = FOUND    if calibrated P(y_error=1 | evidence) <= tau
           ABSTAIN  otherwise
```

This event measures whether executing the selected grounding would be valid.
It deliberately marks AMBIGUOUS, ABSENT and INSUFFICIENT_EVIDENCE as unsafe to
execute and never mixes controller timeout or action failure into perception
risk.

## 6. Locked evidence and calibrators

The evidence definitions and coefficients are fixed in
`pcra_u_calibration_config.json` before capture.

- `spatial_raw_score`: fixed scalar from answerability FOUND log-odds,
  normalized heatmap entropy, 3×3 MAP mass, peak margin and mode count.
- `multimodal_raw_score`: the spatial score plus fixed source-head,
  RGB-depth agreement and valid-depth terms.
- `spatial_platt`: one-dimensional logistic/Platt fit on spatial raw score.
- `multimodal_platt`: one-dimensional logistic/Platt fit on multimodal raw
  score; this is the predeclared primary calibrator.

Both Platt models are frozen regardless of their Calibration diagnostics. The
multimodal model is not declared better unless locked Test later supports that
claim. A source-conditioned multivariable model is not introduced in v1.

## 7. Fit diagnostics and threshold rule

Report both full-fit diagnostics and deterministic five-fold family-grouped
cross-fit diagnostics. Cross-fit is diagnostic only; final calibrators are fit
on all Calibration families. Required metrics are Brier, NLL, 10-bin ECE,
AURC, risk at 80% coverage and coverage at 5% accepted risk. Variants are not
treated as independent for confidence intervals; uncertainty uses 5,000
family-cluster bootstrap resamples with the locked seed.

The operating point is chosen only for `multimodal_platt`:

1. enumerate deterministic unique predicted-risk thresholds;
2. retain thresholds with empirical accepted-incorrect risk at most 0.05 and
   at least 60 independently represented accepted families;
3. choose maximum coverage; ties choose the lower risk threshold;
4. if no threshold qualifies, freeze `ABSTAIN_ALL` and report zero coverage.

The bootstrap upper bound is reported but is not substituted silently for the
locked empirical selection rule. Reference 80% coverage is diagnostic only.

## 8. Freeze and reporting boundary

Freeze artifacts must bind model/checkpoint hash, dataset/QC hash, evidence
features, calibrator coefficients, threshold, target risk, code/config hashes
and the raw-inference manifest. Only a passing freeze authorizes the next
action `CAPTURE_AND_OPEN_LOCKED_TEST_IID_OOD`.

All Calibration tables and figures are labeled `CALIBRATION_FIT_ONLY`, not
final generalization evidence. Table 4 may be populated as Calibration-fit
diagnostics, while Tables 2, 3, 5 and 6 remain `NOT_RUN` until locked Test.

