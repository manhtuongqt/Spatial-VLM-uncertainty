# Scientific boundary

- Training/dev results remain `EXPLORATORY_DEV_ONLY`.
- The 80-family dev split selects checkpoints and must not be reported as Test.
- The 200-family calibration split fits calibration only after model freeze.
- No Test-IID/Test-OOD path is present in this revision.
- `spatial` source supervision is weak and must be reported separately.
- Relation/anchor capacity beyond the distribution present in the dataset is
  implemented but not empirically validated.
- The conformal region is the smallest highest-density grid set whose
  calibrated probability mass reaches the split-conformal quantile; its
  declared coverage event is intersection with the evaluator target mask.
- Point-in-target is not robot-safe grasp success.
- All outputs are new revision artifacts; the historical V1 checkpoint and
  calibration decision remain unchanged.
