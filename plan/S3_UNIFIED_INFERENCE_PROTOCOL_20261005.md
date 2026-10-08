# S3 — Ghép inference trực tiếp cho biến thể đồ án

Ngày 05/10/2026. Ưu tiên người dùng: cơ chế đúng nghĩa, chạy xuyên suốt và
có thể đưa vào đồ án nhanh. Gate cải thiện 49/61, thắng baseline hoặc budget
IID không còn là điều kiện chặn tích hợp; kết quả vẫn phải báo trung thực.

## Contract khóa trước khi chạy

- Primary cố định P1 best epoch 8 của pilot v2; baseline Adapter và target MAP
  giữ nguyên weights. Không neural training, MC Dropout, robot, capture hay OOD.
- Runtime nhận đúng bốn tensor RGB/depth pre-projector và prompt; không nhận
  sample/family/variant, annotation, mask hoặc prediction JSONL cũ. Đường RGB-D
  dùng feature extractor/pooling đã có, làm tròn FP16 như cache trước Sidecar.
- Export trực tiếp 33 features cũ + 11 parser/anchor/geometry features.
  `observable_answerability_evidence` chứa probabilities V2 trước Adapter;
  answerability cuối là sau Adapter. Không tính lại base probabilities từ output
  sau Adapter. Edge mask = prompt relation_mask AND prompt anchor_mask.
- Không chạy RoboRefer language generation: disagreement=0, missing=1.
- Verifier evidence-only trái/phải trong ảnh, margin 12 px. Direct bypass,
  ngoài scope báo unsupported. Binding/presence UNVERIFIED; không sửa target MAP.
- Trace chứa phép tính thật, feature vector, contributions vào logit risk,
  probability và policy. Không gọi normalized compatibility là xác suất quan hệ đúng.
- Fit lại **một** logistic P1_G44 từ evidence trực tiếp trên calibration cũ:
  L2=0,01, 5-fold theo family, cùng quy tắc hash-fold cũ. Policy hard_found,
  risk target 7,5%, tối thiểu 60 accepted family, ngưỡng từ OOF Wilson upper95%.
  Freeze feature producer/source/config/weights trước fit; lưu calibrator/profile
  và manifest riêng. Không chọn model/feature/ngưỡng theo dev hoặc IID.
- Kiểm tra toàn train 1600 và dev 400; masks chỉ join hậu kiểm sau export.
  Calibration 1000/200 family được mở để khớp producer mới. S3 không mở IID.
- Kiểm tra gradient-free/freeze, input boundary, bypass, finite values,
  geometry thực sự vào risk và reload bundle. Sai khác numerical với exports
  lịch sử được đo, không ép exact bằng cached evidence hoặc sửa confidence.
- Smoke một RGB/depth pair train đã có, không capture mới. Báo riêng latency
  backbone và Sidecar; chưa khẳng định calibration ngoài distribution đã đánh giá.

### Bổ sung môi trường trước run r2

Run đầu đã hoàn thành feature→decision/calibration, nhưng smoke RGB-D không
load được backbone do user-site torch2.13 ghép với conda torchvision0.20.
Không sửa môi trường hoặc artifact run đầu. Run r2 khóa bridge mới: extractor
chạy subprocess `python -s` với conda torch2.5.1/torchvision0.20.1; Sidecar giữ
torch2.13 như calibration. Chỉ bốn tensors FP16 qua ranh giới tiến trình.
Đây là sửa orchestration, không thay cơ chế hình học hoặc neural weights.

## Sản phẩm

Module/CLI inference riêng; freeze lock, calibrator/profile và bundle manifest;
tests và train/dev checks; case thật có parse, target/anchor, geometry, risk và
decision; tài liệu kiến trúc/định nghĩa/giới hạn. Baseline active profile và mọi
artifact lịch sử được bảo toàn. LaTeX vẫn ngoài phạm vi.
