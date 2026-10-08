# Nguồn LaTeX cập nhật P-CRA-U Adapter — 2026-10-03

Sáu chương đã đối chiếu với mô hình được chọn trong
`new/outputs/active_experimental_profile.json`: V2 đóng băng + Adapter
answerability `detail_cost` + calibrator `logistic33`. Checkpoint và các
artifact/hình V2 gốc không bị ghi đè.

## Đưa lên Prism

- Cập nhật `Chapter1/chapter1.tex` đến `Chapter6/chapter6.tex` và
  `references.bib`, giữ file master và template của dự án Prism.
- Chuyển kèm thư mục `hinhanh/`. Nguồn dùng đường dẫn
  `hinhanh/<tên ảnh>.png`; 12 hình Adapter có hậu tố `_adapter_vi`.
- Preamble cần `amsmath`, `amssymb`, `graphicx`, `booktabs`, `tabularx`,
  `array` và cơ chế tiếng Việt đang dùng. Không nạp một package hai lần
  với các tùy chọn mâu thuẫn.
- Bản ZIP đi kèm chỉ chứa sáu chương, tài liệu trích dẫn, hướng dẫn này
  và ảnh thực sự được tham chiếu; không thay bìa, phụ lục hoặc master.

## Những phần cập nhật chính

- Chương 1–2: định nghĩa phiên bản hiện tại và phạm vi đánh giá lại.
- Chương 3: residual Adapter, loss riêng, 33 evidence, calibration và
  ngưỡng `0.2611932834526145`; phần V2 lịch sử được phân biệt rõ.
- Chương 4: cấu hình, mining train, chọn best Adapter, calibration ngoài
  fold và full-fit; audit Gazebo vẫn là artifact V2 lịch sử.
- Chương 5: Bảng 5.1 (`tab:results-overall-comparison`) so sánh tổng;
  answerability, calibration, policy, lỗi và case study cập nhật Adapter.
- Chương 6: kết luận và giới hạn tương ứng với kết quả mới.
- 52 bảng dùng booktabs, không kẻ dọc, cột số căn phải, tiêu đề dài ngắt
  dòng, ghi chú có mẫu số. Không điền số giả cho đối chứng chưa đánh giá.

## Số liệu khóa dùng trong bản này

| Chỉ số | V2 lịch sử | P-CRA-U Adapter |
|---|---:|---:|
| Answerability accuracy | 76.80% | 81.10% |
| Answerability macro-F1 | 0.7752 | 0.8097 |
| False FOUND / 580 non-FOUND thật | 70/580 | 55/580 |
| Point-in-target / 835 target-present | 775/835 | 775/835 |
| Ca hợp lệ nhận đúng / 405 | 249/405 | 327/405 |
| Coverage | 25.50% | 36.10% |
| Selective risk | 6/255 = 2.35% | 34/361 = 9.42% |
| Brier risk đã hiệu chuẩn | 0.09615 | 0.07711 |
| AURC risk đã hiệu chuẩn | 0.26045 | 0.25575 |

Ngân sách chọn ngưỡng Adapter trên calibration là **7.5%**, khác ngưỡng
risk từng mẫu **0.261193** và khác lỗi thực nghiệm IID cũ **9.42%**.
IID cũ đã được phân tích; kết quả Adapter không phải xác nhận trên bộ
test mới chưa quan sát. Grounding/source/edge/region giữ nguyên vì các
nhánh này đóng băng; robot motion và pick-and-place chưa được đánh giá.

Nguồn chính: `new/outputs/pcrau_answerability_language_dev_20261003/`
với `detail_cost/checkpoints/best/metadata.json`,
`calibration_full_fit/selected_calibrator.json`, `profiles.json` và
`test_iid_7p5/full_metrics.json`, prediction, runtime decision.
Các thống kê bổ sung được tính từ output đã lưu, không train, fit hoặc
chọn lại threshold bằng test.

## Kiểm tra bố cục

Đã biên dịch sáu chương và bibliography bằng Tectonic với preamble kiểm
tra riêng, A4/12pt; kiểm tra số cột của toàn bộ bảng và xem các trang có
bảng. Bản PDF kiểm tra không có lỗi biên dịch, tham chiếu chưa giải quyết
hoặc tràn ngang. Template Prism có thể cho số trang/ngắt trang khác.
Nguồn bìa, phụ lục, bibliography và các nội dung không liên quan được giữ
nguyên; không tạo thêm một master thay template hiện tại.
