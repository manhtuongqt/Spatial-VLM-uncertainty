# S4 — Reevaluation IID cũ bằng bundle live đã khóa

Ngày 07/10/2026. Người dùng đã giao đánh giá 1000 mẫu/200 family trên IID cũ,
giữ nguyên model/calibrator/threshold và viết tổng hợp để đưa vào đồ án.

## Contract trước forward

- Primary cố định bundle `new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json`.
  Manifest SHA256: `0aacaedf3fa27a2686f936893f5e694b7bea88eeb0675da6503238eecd126446`.
  Freeze SHA256: `5478ac842cab60f8a2bff5e7c8a193b7f806f374ee63b72e38a38f68fa1da81a`.
  Threshold hard_found: `0.2889643687106893`, đã chọn calibration family OOF.
- Không sửa source producer, weights, normalization, feature schema, calibrator
  hoặc threshold; không fit, train, model search, MC, robot, capture hay OOD.
- Chạy đúng `UnifiedInference.predict`, batch20 theo thứ tự manifest, CUDA BF16
  và runtime đã khóa. Dữ liệu input là bốn tensors frozen RGB-D + prompt.
  Không sử dụng 33 evidence từ prediction cũ. Backbone không extract lại 1000
  ảnh vì cache đã có; Sidecar/anchor/verifier/evidence/risk đều forward mới.
- Lưu runtime traces/decisions annotation-free trước evaluator join. Sample và
  family IDs do driver gắn sau predict, không vào model. Masks/truth chỉ hậu kiểm.
- Baseline là active Adapter+logistic33 đã chọn, threshold
  `0.2611932834526145`. Dùng artifact baseline gốc để replay policy/metrics;
  không refit hoặc đổi baseline. So numerical delta để báo, không ép exact
  bằng cách thay evidence trực tiếp hoặc nới gate historical S2.
- Báo PIT target-present, FOUND-only MAP, answerability/confusion, anchor hit
  horizontal visible và empty, policy coverage/risk/valid recall, Brier/NLL/ECE10,
  AURC trên toàn1000. Phân biệt mẫu số và trạng thái scope/binding/presence.
- So sánh paired theo sample và bootstrap200family,5000resamples,seed24082026.
  Báo fixed/broken, thêm/mất valid accept, thêm/loại error accept; không chọn
  model/ngưỡng theo IID hoặc chỉ báo delta tổng.
- Diagnostic đặt raw geometry features=0 trong cùng calibrator chỉ kiểm tra
  wiring, không phải ablation train lại/causal attribution. Quyết định đổi ngoài
  horizontal không tự gọi là lỗi được sửa nhờ geometric verification.
- IID đã quan sát trong lịch sử: đây là reevaluation, không test xác nhận mới.
  Budget7,5% hoặc thắng baseline không là blocker hoàn thành công việc; báo
  trung thực kết quả và CI. Số S2 cached chỉ là lịch sử, không thay lượt live.
- Output riêng `new/outputs/pcrau_unified_spatial_iid_20261007/`; lưu execution
  lock trước forward, source snapshots/provenance, bảo toàn bundle và artifacts cũ.

## Sản phẩm

Runtime/evaluator predictions riêng, metrics và paired changes/CI; vài ảnh case
thật gồm ca sửa được và ca nhận sai; bản tổng hợp kiến trúc + kết quả của đúng
bundle + giới hạn trong `plan/`. Không sửa LaTeX/Prism hay pipeline.png.
