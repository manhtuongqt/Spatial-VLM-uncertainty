# Học ca khó, hiệu chuẩn độc lập và đánh đổi risk–coverage

Người dùng đã yêu cầu thực hiện cả ba bước. Vòng này là DEVELOPMENT_ONLY,
không thay best V2 epoch 13/step 1120, hình 19 hoặc số liệu khóa luận.
Không sử dụng Test-IID/Test-OOD hoặc chạy robot.

## 1. Học ca khó chỉ từ train; chọn model bằng dev

- Backbone và mọi tham số V2 đóng băng. Học residual adapter context
  794 → 64 → 4, dropout 0.3, khởi tạo residual bằng 0.
- Dữ liệu train 320 family/1600 sample, dev 80 family/400 sample.
- Risk proxy dùng logistic 18 evidence, l2=0.001, 5-fold theo family **train**.
  Proxy chỉ phục vụ mining, không dùng calibrator V2 đã fit calibration.
  V2 đã học train, nên proxy OOF này không là đánh giá độc lập của backbone.
- Ngưỡng proxy: lớn nhất thỏa Wilson upper 95% <=5% và >=60 family được nhận,
  trên train crossfit. Nếu không có ngưỡng, dùng 0.1 chỉ để chia nhóm mining.
- Dương khó: FOUND thật và MAP đúng, nhưng answerability dự đoán non-FOUND
  hoặc proxy risk trên ngưỡng. Âm khó cho answerability: non-FOUND thật nhưng
  dự đoán FOUND hoặc proxy risk dưới ngưỡng.
- FOUND nhưng MAP sai vẫn giữ nhãn FOUND trong CE; đây là âm của event risk,
  không được sửa thành ABSENT/INSUFFICIENT_EVIDENCE. Các ca này được lưu riêng.
- Hai nhánh đối chứng cùng initialization/seed 24082026/thứ tự family:
  `control` có example weight 1; `hard_mining` có dương khó weight 2,
  âm khó weight 3, mọi ca khác weight 1. Cả hai dùng inverse train-class weights.
- 25 epoch, AdamW lr 3e-4, weight decay 1e-3, clip 5, 4 family/batch.
- Cổng chọn model: macro-F1 và recall FOUND không thấp hơn V2; tổng false
  FOUND không vượt 18 và riêng ABSENT→FOUND không vượt 4 trên dev.
  Recall ABSENT/INSUFFICIENT_EVIDENCE/AMBIGUOUS không thấp hơn V2.
- Chọn macro-F1 cao nhất trong cổng; hòa ưu tiên ít false FOUND, recall FOUND
  cao hơn, epoch sớm hơn. Nếu không có cải thiện qua cổng, freeze V2 với
  residual zero làm baseline fallback, vẫn thực hiện bước calibration.
- Freeze checkpoint/config/hash/selection trước khi đọc calibration labels.

## 2. Calibration sau freeze, tách vai trò theo family

Sau freeze, dùng 200 family/1000 sample calibration riêng. Hoán vị sorted
family bằng NumPy RNG seed 20261003 rồi chia:

- 100 family/500 sample `fit`: học logistic và standardization;
- 60 family/300 sample `threshold`: chọn ngưỡng hành động;
- 40 family/200 sample `audit`: kiểm tra các model/calibrator/policy đã khóa.

Hai calibrator trên model đã freeze: 18 evidence l2=0.001 và 33 evidence
l2=0.01. Phần thêm là 15 prediction-derived features đã kê ở
`src/pcrau/selective_experiment.py`, không có oracle, metric TF hoặc geometry.
Chọn giữa hai calibrator bằng Brier 5-fold family crossfit trên **fit only**;
hòa ưu tiên 18 evidence. Audit không được dùng chọn mô hình, features hoặc l2.

V2 reference cũng fit lại logistic 18 trên đúng 100 family fit, không dùng
calibrator lịch sử vốn đã thấy toàn bộ 200 family. Tất cả so sánh audit do đó
cùng partition, không lấy số Test-IID cũ so với calibration mới.

## 3. Ba budget và hai kiểu policy

So sánh budget lỗi nhận thức 5%, 7.5%, 10%, không đồng nhất với giá trị
ngưỡng risk trên từng sample. Default profile là risk-only budget 5%; các
profile còn lại là lựa chọn đánh đổi riêng, không triển khai robot.

- `hard_found`: cần predicted FOUND và calibrated risk <= threshold;
- `risk_only`: risk <= threshold có thể vượt qua cổng lớp; nếu không nhận,
  giữ routing hỏi lại/quan sát lại/từ chối theo prediction như policy hiện tại.

Chọn threshold trên threshold partition theo **tập được policy nhận thật**,
Wilson upper 95% <= budget, >=20 family được nhận (partition này chỉ 60
family, không dùng minimum 60 của protocol cũ). Nếu không đạt, threshold=None,
không EXECUTE. Lưu threshold và profile lock trước khi mở audit.

Audit báo coverage, lỗi tổng hợp (non-FOUND thật hoặc MAP sai), đúng FOUND
bị từ chối, số bypass cổng lớp đúng/sai, riêng ABSENT nhận nhầm, reliability
và các action counts. Không chọn lại threshold khi thấy audit.

Wilson ở đây là tiêu chí chọn thực nghiệm; các variant phụ thuộc family và
nhiều threshold được khảo sát, nên không phải chứng minh bảo đảm an toàn.
Audit nhỏ không đủ xác nhận chắc chắn budget thấp; báo cả số đếm và cận.
Đây vẫn là calibration audit, không thay test độc lập hoặc task success robot.
Các vòng phát triển đã biết lỗi Test-IID cũ, cần test độc lập mới sau này.

## Chạy

```bash
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/experiment_hard_cases_selective.py \
  --run-name pcrau_hard_cases_selective_20261003
```

Toàn bộ output vào run mới; không ghi đè source/checkpoint/calibrator cũ.
