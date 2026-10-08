# P-CRA-U development training — preflight report

- Decision: `GO_DEVELOPMENT_TRAIN_CANARY`
- Static checks: `20/20`
- Dataset: `320 train + 80 dev families`
- Samples: `1600 train + 400 dev`
- Prompt-only relation parser: `2000/2000`
- Unique RGB-depth pairs: `1242`
- One-batch loss: `3.139304`; optimizer step: `NO`
- One-batch peak VRAM: `5.721 GiB`
- Calibration/Test: `SEALED`; Tables 2–6: `NOT_RUN`

## Gate checks

- `protocol_and_full_qc_authorize_training`: `PASS`
- `calibration_and_tests_sealed`: `PASS`
- `exact_train_dev_counts`: `PASS`
- `family_split_disjoint`: `PASS`
- `unique_and_aligned_sample_ids`: `PASS`
- `all_locked_artifact_hashes_valid`: `PASS`
- `state_mask_invariants`: `PASS`
- `feature_manifest_oracle_free`: `PASS`
- `prompt_only_relation_parser_matches_2000`: `PASS`
- `only_four_supported_source_labels`: `PASS`
- `spatial_source_not_fabricated`: `PASS`
- `ood_heldout_assets_absent`: `PASS`
- `train_dev_leakage_qc_pass`: `PASS`
- `dataset_baseline_model_protected_hashes_match`: `PASS`
- `checkpoint_manager_unit_tests_pass`: `PASS`
- `one_batch_loss_and_logits_finite`: `PASS`
- `one_batch_all_active_gradients_nonzero`: `PASS`
- `frozen_features_no_gradient`: `PASS`
- `no_optimizer_step_in_preflight`: `PASS`
- `protected_inputs_unchanged_during_preflight`: `PASS`

## Active gradient norms

- `target_head`: `1.939995591e+00`
- `answer_head`: `1.792417893e+00`
- `source_head`: `1.316846802e+00`
- `language_relation`: `6.603841595e-02`
- `rgb_depth_fusion`: `4.862858126e+00`
