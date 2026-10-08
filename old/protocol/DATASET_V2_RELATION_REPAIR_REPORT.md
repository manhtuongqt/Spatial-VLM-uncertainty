# Dataset V2.1 relation-repair pilot report

- Decision: `PASS_RELATION_REPAIR_PILOT`.
- Real clean captures: `30/30`.
- Captures passing structural and observed-relation QC: `30/30`.
- Geometry implementation: `dataset_v2_relation_geometry_v2_1.1.0`.
- Relation labels use semantic-mask centroid or robust median metric depth; simulator centers are not final labels.
- All families are engineering-only and permanently excluded from official train/dev/calibration/test.
- Training performed: `False`.

## Relation summary

| Relation | Captures | Pass | Fail | Minimum signed margin |
|---|---:|---:|---:|---:|
| behind | 4 | 4 | 0 | 0.0538459825515747 |
| between_in_depth | 6 | 6 | 0 | 0.050942178368568417 |
| farther_than | 4 | 4 | 0 | 0.053875188827514645 |
| front_of | 4 | 4 | 0 | 0.0034764218330383297 |
| left_of | 2 | 2 | 0 | 21.520358121266952 |
| nearer_than | 4 | 4 | 0 | 0.027398924827575683 |
| nearer_than_both | 4 | 4 | 0 | 0.021812196373939514 |
| right_of | 2 | 2 | 0 | 99.5762663515317 |

The CSV tables contain the per-capture target/anchor evidence, camera-pose error, predictor residual and observed signed margin. The QC image directory contains only captured RGB and direct depth/semantic/mask renderings selected by a locked seed; no planned infographic was generated.
