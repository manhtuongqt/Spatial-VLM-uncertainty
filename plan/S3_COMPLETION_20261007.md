# Bàn giao S3 — 07/10/2026

**Đã hoàn thành yêu cầu ghép pipeline inference xuyên suốt.** Tên file/run
05/10 được giữ nguyên; ngày này là ngày hoàn thiện tài liệu và kiểm tra cuối.

- [Báo cáo cơ chế và kết quả](S3_UNIFIED_SPATIAL_INFERENCE_20261005.md).
- [Kiến trúc, định nghĩa và cách chạy](../new/docs/UNIFIED_SPATIAL_INFERENCE.md).
- [Bundle dùng hiện tại](../new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json).
- [Kiểm tra cuối](../new/outputs/pcrau_unified_spatial_20261005_r2/final_verification.json).
- [Tám case dev thật](../new/outputs/pcrau_unified_spatial_20261005_r2/report/CASE_TABLE.md)
  và [hình minh họa](../new/outputs/pcrau_unified_spatial_20261005_r2/report/dev_cases.png).

## Kết quả kiểm tra cuối

13 tests đạt; 7 Python files parse được; các liên kết tài liệu đã kiểm tra.
Tái tính từ evidence đã lưu cho 1600 train, 400 dev, 1000 calibration: 44 features,
risk, từng contribution, policy và tổng metric khớp artifacts runtime.
Không gọi model forward hoặc fit trong lượt kiểm tra cuối này.

Hash baseline, residual, config, nguồn inference, protocol, calibrator và profile
khớp lock. 3300 files trong historical protection manifest giữ nguyên hash;
các files của bundle r2 trước lượt kiểm tra cũng giữ nguyên. Snapshot 1193 files
ban đầu nằm ở `/tmp` không còn tồn tại sau chuyển session; vì vậy lượt kiểm
tra mới dùng hash đã ghi trong provenance S2, không dựng lại snapshot giả từ
trạng thái hiện tại. Summary ngày05/10 vẫn ghi lần kiểm tra1193files khi đó.

Geometry thực sự đi vào phép tính: đặt ba raw geometry features về0 trong cùng
calibrator làm đổi 9 quyết định train, 6 dev, 9 calibration. Đây là diagnostic cố định
coefficients và threshold, không phải đối chứng train lại hoặc bằng chứng lợi
ích nhân quả. Cases và probabilities chi tiết nằm trong final_verification.

Smoke ảnh đã có từ train đi qua backbone thật, đạt target grid/44features/risk/
decision giống feature-cache route cho cùng observation. Chỉ một case kỹ thuật,
không khẳng định mọi ảnh đều parity. Môi trường backbone được tách bằng worker;
không cài hoặc thay packages.

## Phạm vi hoàn thành

Đã có **kiểm chứng ràng buộc không gian trên đối tượng dự đoán và hiệu chuẩn
xác suất lỗi nhận thức**, nối vào quyết định với trace. Đủ sản phẩm kỹ thuật
để mô tả cơ chế trong đồ án: code, bundle, contract, calibration, kiểm tra và
case thật. Hoàn thành bước tích hợp không đồng nghĩa hoàn thành toàn khóa luận.

Giới hạn giữ nguyên: trái/phải 2D; binding/presence UNVERIFIED; không hard veto
hoặc sửa target MAP; chưa phân rã aleatoric/epistemic; chưa robot motion/task
success. Anchor dev vẫn 48/61, không sửa thành PASS 49/61. Producer live mới chưa
reevaluate IID, nên số 330 đúng/33 lỗi/363 của S2 cached path không gán cho bundle
này. Baseline active profile và LaTeX/Prism không đổi.

Không cần train thêm hoặc chạy MC để hoàn thành yêu cầu vừa giao. Nếu người
dùng muốn kết quả IID của chính đường live, bước riêng tiếp theo là reevaluate
IID cũ bằng bundle đã khóa; không chọn lại model/features/ngưỡng từ IID.
