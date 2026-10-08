# Hiệu chỉnh residual score chấp nhận trên V2 đóng băng

Thử nghiệm acceptance_recall từ đầu không có ứng viên qua cổng dev 5%.
Vòng tiếp theo giữ protocol train/dev, ba nhánh, seed, 25 epoch và cổng
chọn của EXPERIMENT_ACCEPTANCE_RECALL.md; không đổi tiêu chí khi thấy lỗi.

Điểm thay đổi: error logit = logsumexp(base answer logits non-FOUND)
- base FOUND logit + learned delta. Output layer delta khởi tạo bằng 0,
nên trước học score đúng bằng 1-p(FOUND) V2. Thêm penalty 0.05*mean(delta^2)
vào BCE/ranking để hạn chế việc ghi đè evidence ban đầu. Standardization
chỉ fit train; oracle vẫn chỉ tạo nhãn supervision. Không dùng calibration.

Sau học đây vẫn là score chưa hiệu chuẩn. Cổng chọn giữ nguyên: tăng đúng
chấp nhận, không tăng số lỗi so với đối chứng dev tốt nhất, không nhận
ABSENT, AURC không kém raw V2, và Wilson upper thực nghiệm <=5% trên dev.
Đây là cổng bảo thủ hơn chỉ giữ ngân sách 5%; chưa là bảo đảm độc lập.

Chạy:
```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/experiment_acceptance_recall.py \
  --run-name pcrau_acceptance_residual_dev_20261003 --residual
```

Không thay model/calibrator/threshold V2 chính, không test hoặc robot.
Các vòng chọn trên dev làm tăng nguy cơ overfit dev; cần test mới.

## Calibration sau freeze, chỉ fit/crossfit

Nếu scorer qua cổng dev, khóa hash trước khi mở calibration. So sánh hai
calibrator cố định: logistic33 baseline và logistic34 (thêm raw error logit
của scorer đã freeze). Cùng l2=0.01, 5-fold family và standardization trong
fold, fit đủ 1000 calibration sample/200 family. Chọn ngưỡng OOF <=5%
Wilson upper với >=60 accepted family, cho hard_found và risk_only.
Không dùng kết quả để chọn lại scorer hoặc epoch. Không đổi feature/l2
theo kết quả calibration; báo cả hai. Đây là CALIBRATION_FIT_ONLY vì bộ
calibration đã được quan sát; không gọi audit cũ là độc lập.

```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/calibrate_acceptance_ranker.py \
  --run-name pcrau_acceptance_residual_dev_20261003
```
