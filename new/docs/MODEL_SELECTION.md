# Phiên bản P-CRA-U được chọn — 2026-10-03

Theo quyết định người dùng, mô hình được chọn mang tên hiển thị **P-CRA-U**:
V2 đóng băng + adapter answerability `detail_cost` + calibrator `logistic33`.
Khi so sánh phiên bản, gọi bản trước adapter là **P-CRA-U V2 lịch sử**.
Tên file, run ID và hash gốc giữ nguyên để truy nguyên kết quả.

Lựa chọn hiện tại được lưu tại
[`../outputs/active_experimental_profile.json`](../outputs/active_experimental_profile.json).
Đây là bản ghi lựa chọn; các runner cần được truyền đúng checkpoint,
config, calibrator và profiles ghi trong file, không tự đổi mặc định chỉ
do tên hiển thị thay đổi.

## Thành phần và vận hành

- Checkpoint: `../outputs/pcrau_answerability_language_dev_20261003/detail_cost/checkpoints/best/model.safetensors`.
- Config: `../outputs/pcrau_answerability_language_dev_20261003/config.json`.
- Calibrator: `../outputs/pcrau_answerability_language_dev_20261003/calibration_full_fit/selected_calibrator.json`.
- Profiles: `../outputs/pcrau_answerability_language_dev_20261003/calibration_full_fit/profiles.json`.
- Policy: dự đoán `FOUND` và risk đã hiệu chuẩn không vượt
  `0.2611932834526145` thì `EXECUTE` ở mức quyết định nhận thức.
- Ngân sách chọn ngưỡng trên calibration: 7,5%; lỗi thực nghiệm trên
  Test-IID cũ: 34/361 = 9,42%. Người dùng chấp nhận mức đánh đổi thực nghiệm
  này để tăng số ca hợp lệ được nhận. Không thay ngân sách đã ghi hoặc fit
  lại ngưỡng bằng test.

## Bằng chứng Test-IID

Trên cùng 200 family / 1000 sample, macro-F1 answerability tăng từ 0,7752
lên 0,8097; nhận đúng tăng từ 249/405 lên 327/405 ca hợp lệ. Grounding giữ
nguyên 775/835 = 92,81%. Lỗi trong tập chấp nhận tăng từ 6/255 = 2,35%
lên 34/361 = 9,42%; hai policy có calibrator và ngưỡng khác nhau.

Chi tiết tại
[`../outputs/pcrau_answerability_language_dev_20261003/test_iid_7p5/SUMMARY.md`](../outputs/pcrau_answerability_language_dev_20261003/test_iid_7p5/SUMMARY.md).
Đây là đánh giá lại trên IID đã từng được phân tích, không phải xác nhận
trên một bộ IID mới chưa quan sát. Các quyết định nhận thức chưa phải
robot task success hay xác suất an toàn của robot.

Bộ chấm risk ở `pcrau_support_split_dev_20261003` không được chọn.
Checkpoint, calibrator, kết quả, hình và nguồn LaTeX của V2 lịch sử được
bảo tồn; quyết định đặt tên này chưa cập nhật các nội dung đó.
