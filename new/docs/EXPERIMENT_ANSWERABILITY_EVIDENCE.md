# Vòng 2: answerability với evidence dự đoán, chỉ train/dev

Người dùng yêu cầu tiếp tục cải thiện sau vòng head-only không qua cổng.
Đây là thử nghiệm phát triển riêng; best V2 epoch 13/step 1120 vẫn là mô hình
chính. Hình 19, checkpoint/calibration gốc và các chương không thay đổi.

## Thay đổi và giả thuyết

Giữ cố định toàn bộ V2, kể cả answer_head. Bổ sung residual MLP
26 → 32 → 4 với dropout 0.2, cộng vào logits answerability gốc.
Lớp cuối khởi tạo bằng 0: trước khi học, logits giống V2 chính xác.
26 tín hiệu gồm confidence sigmoid target/interior/anchor, độ tập trung
phân bố vị trí, khoảng cách đỉnh target–interior, edge probability, fusion
gate, RGB/depth thumbnail similarity và MAE, source scores và xác suất
answerability V2. Danh sách chính xác ở `src/pcrau/answerability_evidence.py`.

Các tín hiệu có thể cung cấp thêm thông tin để phân biệt ABSENT với thiếu
evidence và hiệu chỉnh một số ranh giới FOUND. Đây là giả thuyết, không phải
cam kết cải thiện. Source spatial vẫn là đầu ra học từ nhãn yếu; relation
vẫn là proxy. Không diễn giải các feature này thành attribution nhân quả.

Chỉ dùng observable outputs và model input hiện có. Không dùng mask, nhãn,
sample/family ID, variant hay annotation trong forward. ID chỉ ghép cache và
giữ family nguyên vẹn khi sampling. Standardization fit trên train, lưu
mean/scale trong checkpoint; inference không fit lại theo batch hoặc dev.

## Protocol cố định trước huấn luyện

- Train 320 family/1600 sample, dev 80 family/400 sample; không giao nhau.
- Hai nhánh: `balanced_evidence` dùng trọng số nghịch tần suất từ train;
  `unweighted_evidence` không dùng trọng số. Không học lại base head.
- Cùng seed 24082026, 25 epoch, 4 family/batch; cùng thứ tự batch.
- AdamW learning rate 3e-4 cố định, weight decay 1e-3, clip norm 5.
- Mỗi nhánh có smoke một batch, sau đó bỏ update smoke và khởi tạo lại.
- Cổng: macro-F1, recall FOUND, recall ABSENT, recall INSUFFICIENT_EVIDENCE
  và recall AMBIGUOUS đều không thấp hơn V2; false FOUND không vượt V2.
- Chọn macro-F1 lớn nhất trong epoch qua cổng, rồi ít false FOUND hơn,
  rồi recall FOUND cao hơn; hòa hoàn toàn giữ epoch sớm.
- Lưu thêm best macro-F1 không qua cổng dưới tên `diagnostic`, không gọi
  checkpoint này là mô hình được chấp nhận. Nếu cả hai nhánh không đạt,
  kết luận không đạt, không tự sửa cổng hoặc kéo dài run.

Kiểm chứng checkpoint bằng dev inference đầy đủ: matrix khớp cache, tất cả
tham số V2 gốc bất biến, grounding/source/edge giống prediction V2 đã lưu.
Chỉ dùng train/dev; không fit calibration hoặc đánh giá Test-IID/Test-OOD.
Kết quả dev có ảnh hưởng chọn mô hình, một seed, chưa chứng minh khái quát.
Vòng này đã chịu thông tin từ việc xem Test-IID cũ; cần test độc lập mới.

## Chạy

```bash
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/experiment_answerability_evidence.py \
  --run-name pcrau_answerability_evidence_dev_20261003
```

Không resume hay ghi đè run cũ. Artifact nằm trong `new/outputs/<run-name>`.
