# WP3 — Feature-hook runtime và determinism smoke

> **Kết luận:** `GO_PCRA_U_OVERFIT_SMOKE`

> **Thời điểm UTC:** 2026-08-21T05:08:31.202493+00:00

Protocol chạy đúng 15 sample train thuộc 15 family. Không train, không scale dataset, không đọc evaluator artifact trong baseline/hook và không publish target sang robot.

## 1. Gate

| Gate | Kết quả |
|---|---:|
| `feature_gate_lock_verified` | PASS |
| `exactly_15_train_samples_from_15_families` | PASS |
| `all_three_runs_complete` | PASS |
| `feature_cache_schema_valid` | PASS |
| `inference_cache_oracle_free` | PASS |
| `preprojector_shape_contract` | PASS |
| `projected_shape_contract` | PASS |
| `rgb_depth_tile_alignment` | PASS |
| `runtime_dtype_aligned_and_documented` | PASS |
| `hook_a_b_tensor_exact_equality` | PASS |
| `hook_a_b_tensor_file_hash_equality` | PASS |
| `hook_a_b_stable_metadata_equality` | PASS |
| `hook_does_not_change_baseline_raw_answer` | PASS |
| `hook_does_not_change_parsed_prediction` | PASS |
| `wp0_wp1_wp2_inputs_unchanged` | PASS |
| `no_training_no_dataset_scaling` | PASS |

## 2. Runtime feature contract

- Baseline/hook A/hook B: 15 sample mỗi run;
- pre-projector `R0/D0`: `[N_tiles, 1024, 1152]`;
- projector output: `[N_tiles, 121, 1536]`;
- dynamic tiling: [12] local tiles + 1 thumbnail;
- runtime dtype: `['torch.float16']`;
- CuBLAS workspace: `:4096:8`;
- peak VRAM lớn nhất: 5.721 GiB;
- feature latency median: 1061.967 ms/sample;
- generation latency median: 1912.713 ms/sample.

Checkpoint JSON khai báo `bfloat16`, nhưng loader inference đang dùng trong RoboRefer (`llava/model/builder.py`) chuyển model/tower/projector sang `torch.float16`. Smoke ghi runtime truth là FP16 và không tự ý sửa loader hoặc checkpoint sau khi contract đã khóa.

## 3. Determinism và non-interference

- Tensor exact equality: 30/30;
- max absolute difference toàn bộ R0/D0: 0.0;
- max relative difference toàn bộ R0/D0: 0.0;
- safetensors hash giống nhau A/B: 15/15;
- raw answer baseline = hook A = hook B: 15/15;
- parsed point giống nhau: 15/15.

## 4. Coverage của 15 mẫu

Selection audit là evaluator-only và chỉ được đọc ở bước comparison/report, không được loader baseline/hook đọc.

- Answerability states: `{'ABSENT': 1, 'AMBIGUOUS': 2, 'FOUND': 8, 'INSUFFICIENT_EVIDENCE': 4}`;
- Family categories: `{'ambiguous_absent': 3, 'direct_grounding': 2, 'front_behind_camera': 2, 'metric_comparison': 1, 'multi_anchor_depth_order': 2, 'nearer_farther': 2, 'occlusion_depth_evidence': 1, 'relation_2d': 1, 'shape_comparison': 1}`;
- Variants: `{'clean': 8, 'depth_corruption': 3, 'occlusion_view_counterfactual': 2, 'relation_counterfactual': 1, 'semantic_counterfactual': 1}`;
- Multi-anchor samples: 4.

## 5. Safety và bất biến

- Cache chỉ chứa R0/D0, input/checkpoint/code hashes, tile metadata, baseline prediction và runtime measurements;
- không chứa mask, object identity, oracle/relation graph, answerability/source label, expected intervention, correctness, test metric hoặc Gazebo pose;
- WP0 tree, WP1 tree và toàn bộ WP2 dataset tree giữ nguyên digest trước/sau;
- schema/checkpoint/source/manifest đều được khóa SHA-256;
- không có optimizer, backward, dataset generation hay robot command trong protocol.

## 6. Quyết định

Feature gate WP3 đã pass. Bước kế tiếp được phép là **P-CRA-U overfit smoke trên một train subset rất nhỏ** để kiểm tra loader/loss/gradient.

Quyết định này **không** cho phép development training, calibration fitting, mở prototype test để chọn model hoặc scale dataset.
