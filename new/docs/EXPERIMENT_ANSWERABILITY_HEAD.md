# Thử nghiệm head answerability trên biểu diễn V2 cố định

## Phạm vi và giả thuyết

Đây là thử nghiệm phát triển riêng, không thay P-CRA-U/best V2 trong khóa luận.
Khởi tạo từ best V2 epoch 13, step 1120; chỉ học lại `answer_head`.
Toàn bộ biểu diễn phía trước, grounding, anchor, edge và source được đóng băng.
Biểu diễn train/dev được trích ở chế độ eval, không bật dropout phía trước head.
Mask và nhãn chỉ dùng trong loss/evaluator, không làm đầu vào biểu diễn.

Giả thuyết: ranh giới phân loại answerability có thể được cải thiện mà không
thay đổi grounding. Hai nhánh có cùng khởi tạo, thứ tự batch và ngân sách:

- `balanced_control`: cross-entropy với trọng số nghịch tần suất từ train,
  như loss answerability hiện tại;
- `unweighted_candidate`: cross-entropy không dùng trọng số lớp.

Đây là đối chứng cho cách đặt trọng số trong giai đoạn fine-tune head, không
phải ablation toàn bộ quá trình huấn luyện V2. Không tăng trọng số FOUND
dựa trên số liệu test, không sửa nhãn và không thêm location-mass loss.

## Cấu hình cố định trước khi chạy

- Train: 320 family / 1600 sample; dev: 80 family / 400 sample, không giao nhau.
- Seed 24082026, 8 epoch, batch gồm 4 family nguyên vẹn (20 variant).
- AdamW, learning rate 1e-4 cố định, weight decay 1e-4, clip norm 5.
- Giữ dropout head 0.1; biểu diễn V2 ở chế độ eval cố định.
- Có bước smoke một batch trước huấn luyện; bước này được bỏ, không đi vào run.
- Chọn epoch theo macro-F1 dev cao nhất trong các epoch đạt cả ba cổng:
  macro-F1 không thấp hơn V2, recall FOUND không thấp hơn V2, số non-FOUND
  bị đoán FOUND không cao hơn V2. Khi hòa, ưu tiên ít false FOUND hơn,
  rồi recall FOUND cao hơn, rồi epoch sớm hơn.
- Một epoch chỉ được gọi cải thiện nếu macro-F1 cao hơn V2 ngoài sai số số học.
  Nếu không đạt, ghi thất bại; không tự đổi cổng hoặc tìm tiếp siêu tham số.
- Kiểm tra không có tham số ngoài answer_head thay đổi; kiểm chứng checkpoint
  được chọn bằng inference dev đầy đủ và đối chiếu dự đoán với cache.

## Đánh giá và giới hạn

Lưu confusion matrix, accuracy, macro-F1, recall/precision từng lớp, false
FOUND, lỗi theo variant, prediction từng sample và hash nguồn. Không chỉ báo
accuracy hoặc recall FOUND để che lỗi ở lớp khác.

Chỉ dùng train/dev. Không mở calibration/Test-IID/Test-OOD để chọn mô hình.
Hướng thử đã được gợi từ lỗi Test-IID được xem trước đây, vì vậy Test-IID cũ
không còn là bằng chứng đánh giá độc lập cho vòng phát triển mới này.
Mọi cải thiện ở đây là dev, chưa phải cải thiện test hoặc tăng coverage policy.
Sau khi chốt model cần calibration phù hợp; không dùng lại calibrator V2 để
kết luận risk/policy của head mới. Chưa sửa hình 19 hay số liệu LaTeX.

## Chạy lại

```bash
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/experiment_answerability_head.py \
  --run-name pcrau_answerability_head_dev_20261003
```

Run name phải chưa tồn tại. Mỗi nhánh có history và checkpoint riêng trong
`new/outputs/<run-name>/`; report chung là `SUMMARY.md` và `summary.json`.
