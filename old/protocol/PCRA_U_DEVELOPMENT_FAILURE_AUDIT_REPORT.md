# P-CRA-U development failure audit and architecture decision

- Decision: `FREEZE_PCRA_U_ARCHITECTURE_AND_CHECKPOINT`
- Status: `EXPLORATORY_DEV_ONLY`; 80 family là đơn vị độc lập, 400 variants không được coi là 400 quan sát độc lập.
- Checkpoint: `step_000002000` (epoch 10), không có optimizer step hay retraining trong audit.
- Calibration/Test-IID/Test-OOD vẫn đóng; Bảng 2–6 vẫn `NOT_RUN`.

## Kết quả trọng tâm

- P1 false-FOUND trên AMBIGUOUS/ABSENT: `9/95 = 0.0947`.
- P1 point-outside-target trên FOUND: `24/168`; point-in-target `0.8571`.
- Answerability macro-F1: `0.7420`; source micro-F1 bốn nhãn: `0.5876`.
- Bootstrap: `5000` lần, resample theo family, seed `24082027`.

## Diễn giải baseline và ablation

- B0/B1 là RoboRefer thật với greedy decoding; vì luôn buộc trả point nên không có head answerability bốn trạng thái.
- B2 dùng gate độ sâu hiện có và chỉ lấy clean dev làm phạm vi so sánh chính; không dùng oracle mask trong inference.
- U1 chỉ vô hiệu explicit relation slot; từ chỉ quan hệ vẫn còn trong prompt tokens.
- `A_NO_DEPTH` zero D0/thumbnail tại inference. Hai phép này là post-hoc intervention trên cùng checkpoint, không phải retrained causal ablation.

## Gate quyết định

- `selected_checkpoint_integrity`: `PASS`
- `protected_dataset_model_baseline_unchanged`: `PASS`
- `calibration_and_tests_sealed`: `PASS`
- `exact_expected_failure_counts_reproduced`: `PASS`
- `p1_false_found_point_estimate_at_most_0_10`: `PASS`
- `p1_family_grounding_ci_lower_at_least_0_70`: `PASS`
- `explicit_relation_intervention_non_degradation`: `PASS`
- `depth_intervention_non_degradation`: `PASS`

## Phạm vi bước sau

Kiến trúc/checkpoint được khóa; bước được phép tiếp theo là capture Calibration. Test vẫn đóng.

Các bảng so sánh trong run đều mang tên `table_dev_*_exploratory.csv`; chúng không phải Bảng 2–6 kết quả cuối.
