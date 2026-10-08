# S2 — Numerical preflight trước calibration/IID

Ngày05/10/2026. Initial run dừng trước calibration ở parity guard; không có
calibrator hoặc IID result mới trong run đó. Giữ failed root/source snapshot.
Dev-only probe trên400mẫu: target distribution vàsource probabilities exact;
Adapter probabilities lệch ởduy nhất
`v211dev_family_000346__occlusion_view_counterfactual`, maxabsolute
**1,4314427971839905e-5**. Chưa xác định nguyên nhân cụ thể; không gọi mọi
historical dev output exact hoặc tự sửa giá trị để khớp.

Amendment đầu chỉ cho devtolerance2e-5. Runr2 dừng trong calibration inference,
trướcfit, ở`v211cal_family_000137__relation_counterfactual`: Adapter delta
1,3872981071472168e-5, target/source0. Không cócalibrator hoặc IID resultr2.
Giữr2snapshot và bảnamendment đầu riêng trongfailed root.

Numerical contract khóa **trước fit/IID**: áp cùng tolerance2e-5 đã chọn trêndev
cho historical Adapter probability parity ởmọi split; argmax giữ nguyên.
Target/source giữexact ởmọi split. Không nới hơn2e-5 sau khi mởIID. Baseline
plain forward với capture phải exact. Source33riskfeatures dùng cached frozen
baseline exports đã hash, không thay bằng probabilities vừa recompute.

Đây là numerical identity clarification chọn trêndev, không đổi feature schema,
model, margin, loss, calibrator, budget, threshold gate hoặc mẫu số. Initial
failure không có calibration fit/test mining. PrimaryP1_G44 vẫn cố định.
