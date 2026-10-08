# Dataset V2 relation geometry amendment 02

- Scope: repair-pilot round 2 only; official train/dev remain locked.
- Observable runtime labels are unchanged from amendment 01: `0.020 m` depth
  margin and semantic-mask/metric-depth evidence.
- Round 1 captured `30/30` and passed `29/30`; the only rejection was
  `between_in_depth` for mug/mustard/blue with signed margin `-0.016719 m`.
- To absorb the measured tall-object surface-depth residual, static candidate
  screening for `between_in_depth` is raised from `0.035 m` to `0.070 m` per
  side. Other relations keep `0.035 m`.
- Candidate pool size is raised deterministically from `4096` to `65536`.
- Round-1 captures remain diagnostic and are never relabeled or reused as
  accepted round-2 data.
- No baseline, URDF, controller, Gazebo world, calibration/test, or training
  component is changed by this amendment.
