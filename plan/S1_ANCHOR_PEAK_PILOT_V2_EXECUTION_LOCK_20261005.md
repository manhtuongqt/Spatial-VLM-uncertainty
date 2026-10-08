# S1 — Execution lock pilot C1/P1/P2

Ngày05/10/2026. Người dùng đã giao hoàn thiện orchestration và chạy pilot v2.
File này là addendum triển khai, không sửa [protocol đã hash](S1_ANCHOR_PEAK_PROTOCOL_V2_LOCK_20261005.md)
hoặc [config preflight](../new/configs/anchor_peak_pilot_v2_20261005.json).
Các flags “chưa orchestration/optimizer” ở receipt/config cũ ghi trạng thái
preflight lúc đó; execution receipt mới ghi orchestration đã có và pilot được giao.

Giữ nguyên margin1, peak weight0,10, empty BCE weight0,25, seed24082026,
data/prefix/init và optimizer/budget/selection/gates. Không warm-start M1/M2.
Thứ tự mỗi epoch C1→P1→P2, mỗi arm áp dụng cùng family plan; frozen captures
và supervision được cache/chia sẻ trong epoch. MLP/loss FP32, frozen capture/head
AMP bfloat16 trên CUDA như pilot v1. Mỗi arm early stop độc lập sau5epoch không
improve, tối đa15epoch. Selection lexicographic hits→matched pairs→negative
mean empty sigmoid-max, hòa giữ checkpoint sớm. Save best khi improve, reload
và verify exact dev metrics trước báo cáo. Không chọn bằng train hoặc gate.

Trước optimizer: verify806protected preflight hashes/source, bundle/config/protocol,
split/family plans, baseline/init digests, step-zero maps và oracle boundary;
lưu execution/source snapshot receipt. Runner chỉ optimizer trên residual.
Chạy dev mỗi epoch; lưu curves/logits/loss diagnostics, checkpoints và hashes.
Cuối pilot: tất cả400dev baseline outputs/inactive maps exact, freeze hash và
data/historical artifacts bảo toàn, cùng latency setup batch8/warm20/runs100.

So P1−C1 và P1−P2; báo cả ba arm với M0. Bootstrap5000 theo family seed đã khóa,
CI conditional on dev-selected checkpoints, không phải validation độc lập.
Gates P1:49/61anchor,8/15matched,12/16FOUND-both; từng3devempty không tăng
quá1e-6 soM0; P1≥P2 cảhits/pairs và hơn≥2 ít nhất một metric; latency≤20%.
H_peak thêm yêu cầu P1≥C1 cảhits/pairs và có strict improvement ít nhất một.
Không hồi tố gate; train empties tăng vẫn báo nhưng không dùng chọn lại checkpoint.

Chỉ train/dev. Chưa mở verifier, MC, calibration samples/fit, IID/OOD hay robot;
gate PASS chỉ đủ đề xuất bước verifier tiếp theo, không tự triển khai trong pilot.
