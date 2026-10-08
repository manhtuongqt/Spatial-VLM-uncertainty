# Gazebo Train-UQ v2 full capture attempt 01 decision

- Decision: **REJECT — immutable negative full-data attempt**.
- Geometry QC: 317/320 verified; 3/320 failed the preregistered
  `INSUFFICIENT_EVIDENCE` gate of 1–119 visible target pixels.
- Failed scenes: `train_uq_126` (129 px), `val_uq_302` (0 px), and
  `val_uq_319` (147 px).
- The attempt must not be filtered, patched, combined with another capture,
  materialized, used for inference, or used for estimator fitting/selection.
- B0, the WP4 hypothesis, risk definition, metric suite, selection rule,
  bootstrap protocol, and sealed Dev/Calibration/Test policy are unchanged.
- The allowed repair scope is geometry/capture stability only. A successor
  revision must use fresh scene, family, layout, and pose-signature identities
  and must pass 320/320 as one complete attempt.

The failure is a geometry-boundary failure, not evidence for or against the
scientific uncertainty hypothesis.
