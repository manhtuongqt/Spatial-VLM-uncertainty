# C1/P1/P2 pilot v2 — 05/10/2026

**PILOT_COMPLETE; upgrade gates FAIL.** P1 vàC1 cùng48/61anchor,8/15swap,
12/16FOUND-both; P2 44/61,6/15,11/16. Chưa có thêm binding benefit của lossv2.

Đọc [báo cáo tiếng Việt](../../../plan/S1_ANCHOR_PEAK_PILOT_V2_20261005.md),
[paired report](paired_report/SUMMARY.md), [full metrics](summary.json),
[execution addendum](../../../plan/S1_ANCHOR_PEAK_PILOT_V2_EXECUTION_LOCK_20261005.md).

| Arm | Best checkpoint | Best epoch | Completed epochs/steps |
|---|---|---:|---:|
| C1 | [MLP](C1/checkpoints/best/mlp.safetensors) | 11 | 15/240 |
| P1 | [MLP](P1/checkpoints/best/mlp.safetensors) | 8 | 13/208 |
| P2 | [MLP](P2/checkpoints/best/mlp.safetensors) | 1 | 6/96 |

Đây là residual riêng, cần frozen bundle trong`freeze_lock.json`, không phải
full model đã promote. C1 train lại từinit, checksum best trùng historicalM1.

| Artifact | Nội dung |
|---|---|
| `locked_protocol.json`, `pre_optimizer_receipt.json`, `source_snapshot/` | Execution/config/source/init lock trước optimizer |
| `split_manifest.json`, `family_batch_plans.json` | Data/prefix/plans giữ v1/v2preflight |
| `epoch_history.json`, `dev_inference_epoch_*.jsonl` | Dev mỗi epoch, losses/counts, selection/patience |
| `selected_inference_predictions.jsonl`, `selected_spatial_logits.npz` | Best-reloaded inference; oracle masks tách riêng |
| `evaluator_rows.jsonl`, `swap_pairs.jsonl`, `changed_cases.*` | Exact mask-center hits và corrections/regressions |
| `empty_cases.jsonl`, `peak_diagnostics.jsonl` | Từng5train/3devempty, maxlogits/BCE và inside/outside/mass |
| `family_paired_deltas.jsonl`, `paired_report/*.csv` | Paired family/sample results |
| `paired_report/learning_curves.png` | Actual curves; selected epoch stars |
| `summary.json`, `latency.json`, `freeze_lock.json`, `provenance.json` | Metrics/gates/CI/latency/freeze/source/data hashes |
| `final_verification.json` | Kiểm chứng cuối nhiệm vụ |

Không tune/retrain từ report; chưa verifier, MC, calibration samples/fit hoặc
IID/OOD. Các flags preflight lịch sử trong config cũ giữ nguyên; execution
receipt hiện tại ghi orchestration đã triển khai và train được giao.
