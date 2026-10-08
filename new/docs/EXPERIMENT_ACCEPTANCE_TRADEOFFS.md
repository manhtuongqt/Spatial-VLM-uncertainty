# Profile acceptance residual 5%, 7.5%, 10%

Người dùng ngày 03/10/2026 cho phép lỗi nhận thức 5%, 7.5% hoặc 10% để
tăng số ca hợp lệ được chấp nhận. Vòng này mở các profile trên scorer
đã học train/chọn dev/freeze trước calibration, không train bằng các ca
calibration bị từ chối hoặc Test-IID. Không đổi epoch, feature hay l2.

Source run: `new/outputs/pcrau_acceptance_residual_dev_20261003/`.
Scorer: context_bce_rank epoch 24 (zero-based), residual 794→64→1.
V2 epoch 13/step1120 đóng băng. Calibrator34 = 33 evidence + scorer logit.
Đối chứng dùng logistic33 trên cùng calibration fold; không gọi đối chứng
này là calibrator18 lịch sử Test-IID.

Threshold chọn từ risk OOF 5-fold family, Wilson upper 95% <=budget,
ít nhất 60 family nhận được. Báo hard_found/risk_only cho cả ba budget.
Fit-all chỉ phục vụ functional runtime; không dùng số fit-all làm kết quả
khái quát. Không lấy audit cũ đánh giá các profile vừa fit.

Default thử nghiệm budget10% vì mục tiêu chấp nhận thêm ca đúng và người
dùng cho phép mức này; hai mức thấp vẫn lưu riêng. Đây là lựa chọn vận
hành, không phải chứng minh bảo đảm lỗi10% trên miền mới hoặc robot.
Các kết quả là CALIBRATION_FIT_ONLY, cần test độc lập mới. Macro-F1 và
grounding V2 không đổi, các ca non-FOUND bị chặn chưa tự biến thành FOUND.

Runtime sử dụng inference wrapper chỉ nhận input V2 cho phép, giữ mọi
đầu ra gốc và thêm `ranker_error_logit`. Context từ answer head và 26
evidence đều observable; annotation chỉ là nhãn đánh giá. Scorer chạy
float32 kể cả V2 AMP để khớp đường cached training. Prediction gắn hash
V2/scorer; decision CLI xác minh khớp calibrator và profile trước áp dụng.

## Tạo profile

```
PYTHONPATH=new/src .conda-roborefer/bin/python new/scripts/profile_acceptance_tradeoffs.py \
  --run-root new/outputs/pcrau_acceptance_residual_dev_20261003
```

## Prediction mới trên dev và quyết định offline

```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/evaluate.py \
  --config new/outputs/pcrau_target_v2_full_seed_24082026/config.json \
  --checkpoint new/outputs/pcrau_target_v2_full_seed_24082026/checkpoints/best/model.safetensors \
  --acceptance-ranker new/outputs/pcrau_acceptance_residual_dev_20261003/context_bce_rank/diagnostic.pt \
  --split dev --smoke \
  --output-dir new/outputs/pcrau_acceptance_residual_dev_20261003/runtime_dev_smoke

PYTHONPATH=new/src .conda-roborefer/bin/python new/scripts/decide_experimental_profile.py \
  --predictions new/outputs/pcrau_acceptance_residual_dev_20261003/runtime_dev_smoke/predictions.jsonl \
  --calibrator new/outputs/pcrau_acceptance_residual_dev_20261003/risk_tradeoffs/selected_model_calibrator.json \
  --profiles new/outputs/pcrau_acceptance_residual_dev_20261003/risk_tradeoffs/profiles.json \
  --output new/outputs/pcrau_acceptance_residual_dev_20261003/runtime_dev_smoke/decisions.jsonl
```

Không truyền `--risk-target` thì dùng default10% của profile mới.
Thêm `--risk-target 0.05` hoặc `0.075` để dùng hai mức thấp hơn.
Profile cũ không có default vẫn mặc định5% để giữ tương thích.
CLI chỉ xuất action labels; không có giao tiếp MoveIt/controller.

Không sửa checkpoint/calibrator/policy V2 chính, LaTeX hoặc figure.
