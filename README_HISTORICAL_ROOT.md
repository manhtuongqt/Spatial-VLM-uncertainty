# Spatial-VLM RefSpatial and Gazebo UR3

Current work follows the eight-week plan in
[`plan/ke_hoach_trien_khai_spatial_vlm_refspatial_gazebo_ur3.md`](plan/ke_hoach_trien_khai_spatial_vlm_refspatial_gazebo_ur3.md).

The project develops RGB-D tabletop grounding with RoboRefer, audited spatial
relations, and an ACT / REOBSERVE / ABSTAIN policy evaluated in Gazebo UR3.

## Current status

- RefSpatial source inventory and automated QC are complete.
- The independent WP1 human audit is complete: 300/300 reviews and the target,
  relation, and anchor sample gates pass. See
  [`MANUAL_AUDIT_HUMAN_REPORT.md`](results/spatial_vlm_refspatial_v1/wp1_source_audit/MANUAL_AUDIT_HUMAN_REPORT.md).
- `D_tabletop_clean_v1` is released for B1 target grounding only: 1,500 train
  samples / 1,500 families, balanced 500/500/500 across the three retained
  horizontal-ranking relations. Extraction, split, lineage, media, leakage,
  and human-sample gates pass.
- Clean B1 LoRA training completed for one epoch (375 optimizer steps). It moves
  Hit@.08 from 97.22% to 100% on the small 36-case internal held-out set, but
  does not improve the independent 99-case human-certified stress set
  (B0 96.97%, B1 95.96%). B1 is therefore not promoted over B0.
- WP5's frozen spatial-risk estimator passed its Val-UQ eligibility gate.
  Final Calibration then completed on 128 independent Gazebo families, but its
  one-parameter temperature scaler improved Brier while slightly worsening
  ECE-10. The locked decision is
  `CALIBRATION_NEGATIVE_NO_ROBOT_DEPLOYMENT`.
- B2 remains closed; Test-IID/OOD and robot deployment remain sealed.

The current figures, machine-readable tables, plan-alignment matrix, and
detailed Vietnamese progress report are collected in
[`project_scientific_summary_20260914`](results/spatial_vlm_refspatial_v1/project_scientific_summary_20260914/BAO_CAO_TONG_TIEN_DO.md).
The compiled presentation report is available as
[`BAO_CAO_KHOA_HOC_SPATIAL_VLM.pdf`](results/spatial_vlm_refspatial_v1/project_scientific_summary_20260914/latex_report/BAO_CAO_KHOA_HOC_SPATIAL_VLM.pdf).

## Active workspace

- `datasets/RefSpatial-*`: source RefSpatial folders.
- `protocol/refspatial_*`: source audit, review, release-gate, and evaluation tooling.
- `protocol/gazebo_spatial_v1_*`: Gazebo development split and seed design.
- `results/spatial_vlm_refspatial_v1/`: artifacts of the active plan.
- `results/refspatial_drive_audit_20260907/`: source-audit evidence used by the active plan.
- `RoboRefer/`, `sam2/`, `ur3/`, `.conda-roborefer/`: local dependencies and robot integration.

## Archive

The prior P-CRA-U/Dataset V2 plan, its pilots, reports, metrics, result catalog,
feature-cache manifests, and related protocol files are in [`old/`](old/).
They remain for historical inspection only and are not inputs to the active data
release or model evaluation.
