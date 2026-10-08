# Gazebo Train-UQ v2 full-r2 capture attempt 01 decision

- Decision: **REJECT — immutable negative full-data attempt**.
- Geometry QC: 316/320 verified; 4/320 failed.
- Failures: three `INSUFFICIENT_EVIDENCE` visibility boundary failures
  (132, 0, and 126 target pixels) and one `AMBIGUOUS` rendered-tie failure.
- RGB duplicate QC passed (320 unique SHA-256 images; zero locked-rule
  perceptual near-duplicate pairs).
- The attempt must not be filtered, patched, combined, materialized, used for
  B0 inference, or used for estimator fitting/selection.
- B0, the WP4 hypothesis, metrics, selection rule, bootstrap protocol, and
  sealed Dev/Calibration/Test policy remain unchanged.
- A successor must have fresh identities and pass 320/320 in one attempt.

This remains engineering evidence about render/physics stability, not evidence
for or against the scientific uncertainty hypothesis.
