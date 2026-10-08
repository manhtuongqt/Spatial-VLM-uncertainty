# WP1 — Depth Sensitivity Diagnostic

**Quyết định chính thức: `GO_PCRA_F`.**

Lý do: `EXISTING_MODEL_NEARLY_INVARIANT_TO_DEPTH_COUNTERFACTUALS`.

## Kết quả theo điều kiện

| Điều kiện | Parse | Point hit single-target | Interior hit | Median displacement vs correct | Median latency |
|---|---:|---:|---:|---:|---:|
| `rgb_only` | 10/10 | 8/8 | 7/8 | 0.0042 | 1041.1 ms |
| `correct_depth` | 10/10 | 8/8 | 8/8 | 0.0000 | 1840.0 ms |
| `flat_depth` | 10/10 | 8/8 | 7/8 | 0.0035 | 1840.8 ms |
| `shuffled_depth` | 10/10 | 7/8 | 7/8 | 0.0223 | 1845.6 ms |
| `inverted_depth` | 10/10 | 8/8 | 7/8 | 0.0030 | 1842.1 ms |
| `localized_holes_edge` | 10/10 | 8/8 | 8/8 | 0.0028 | 1846.3 ms |

## Paired depth-counterfactual evidence

- Sensitive displacement: 6/40 (15.0%) với ngưỡng normalized Euclidean `0.02`.
- Instance switch: 1/40 (2.5%).
- Correct-hit → miss dưới depth corruption: 1 paired records.

## Gate và provenance

- 60 prediction được query đúng một lần bằng greedy decoding, seed `8132026`.
- Runner không đọc annotation, semantic labels hoặc Gazebo oracle.
- Evaluator chỉ mở oracle sau khi `prediction_lock.json` và toàn bộ inference hash được xác minh.
- Pilot tree digest trước/sau: `df8ec334c82b14447463101cccb5e54836fb27e579a760322256f84d157a2b9c`.
- Kết quả gốc: `results/roborefer_depth_sensitivity_v1_20260818_152700`.

## Diễn giải có giới hạn

WP1 chỉ có 10 scene nên đây là diagnostic, không phải kiểm định ý nghĩa thống kê. Checkpoint hiện chỉ xuất một point, vì vậy không có heatmap-change metric. Quyết định trên tuân theo threshold đã khóa trước inference, không được chọn lại sau khi xem oracle.

## Artifact

- GUI: `results/roborefer_depth_sensitivity_v1_20260818_152700/evaluation/WP1_GUI.html`
- Summary: `results/roborefer_depth_sensitivity_v1_20260818_152700/evaluation/summary.json`
- Paired CSV: `results/roborefer_depth_sensitivity_v1_20260818_152700/evaluation/paired_comparisons.csv`
