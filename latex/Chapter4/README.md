# Chương 4 — Thiết lập thực nghiệm

## Cập nhật Adapter ngày 2026-10-03

Nguồn chương đã cập nhật theo P-CRA-U = V2 đóng băng + Adapter
answerability + logistic33. Các bảng dùng booktabs/tabularx, căn số phải
và ngắt tiêu đề dài; preamble cần `amsmath, amssymb, graphicx, booktabs,
tabularx, array` cùng cơ chế tiếng Việt của template.

Đã biên dịch thử cả sáu chương bằng Tectonic với preamble kiểm tra riêng,
kiểm tra bố cục các trang có bảng. Workspace vẫn không thay file master
của Prism; mô tả V2 và tình trạng trước cập nhật bên dưới là lịch sử.
Hình dùng đường dẫn `hinhanh/<tên ảnh>.png`; các asset cần thiết nằm trong
`latex/hinhanh/`. Kết quả Adapter là đánh giá lại IID cũ, không phải test mới.


chapter4.tex có 10 mục: bảy nhóm nội dung theo form, bổ sung giao thức
đánh giá/CI, demo Gazebo và khả năng tái lập. Nội dung dựa trên code,
manifest, config snapshot, history và lock artifact hiện có.

## Bảng và hình

- 18 bảng LaTeX dùng booktabs, không kẻ dọc, caption trên bảng,
  ghi nguồn và lưu ý dưới bảng.
- Ba figure dùng ảnh hiện có: capture RGB/depth/annotation canary
  (bốn panel), dashboard bốn ô và montage audit năm frame.
- Ảnh là capture/render Gazebo, không phải ảnh chụp robot ngoài đời.
- Không tạo ảnh AI, không chạy lại train/capture/inference.

Preamble file master cần:

    \usepackage{amsmath,amssymb,graphicx,booktabs,tabularx}

Đường dẫn asset hỗ trợ biên dịch từ latex/ hoặc workspace root.
Nếu dùng Overleaf, cần chuyển kèm asset và chỉnh \pcrauExpRoot.
Template phải có hỗ trợ tiếng Việt. Chưa có master và trình biên dịch
LaTeX trong workspace, nên mới kiểm tra source và sự tồn tại asset;
chưa xác nhận bố trí bảng/float, ngắt trang hoặc overfull trong PDF.

Nhãn chapter đã đổi thành sec:experimental-setup để không trùng nhãn
sec:experiments ở Chương 5.

## Nguồn số liệu

- Train/dev: old/protocol/pcra_u_development_train_manifest.json.
- Calibration: old/protocol/pcra_u_calibration_eval_manifest.json.
- Test-IID: new/test_iid/protocol/test_iid_eval_manifest.json.
- Quota/perturbation: test_iid_manifest.json và generator Test-IID.
- Camera/batch: test_iid_capture_plan.json.
- QC: new/test_iid/dataset/report_assets/checkpoints/batch_qc/.
- Cache: development/calibration/Test-IID index_full.json.
- Runtime, config, history, best: run pcrau_target_v2_full_seed_24082026.
- Freeze: new/test_iid/contracts/best_v2_freeze_lock.json.
- Calibration: artifact V2 đã fit, không lấy threshold V3 thay thế.
- Baseline: new/test_iid/evaluation/roborefer_original/run_manifest.json.
- CI: evaluation/comparison_original_vs_best_v2/full_comparison.json
  và compare_grounding_family_bootstrap.py.
- Metrics: full_metrics.json và code analyze_results.py.
- Demo: README và audit_summary.json của iid_pose_audit_20261001T080100Z_49405.

Đã đếm nhãn trực tiếp từ manifest cho cả bốn tập; không lấy số family
clean nhân năm để suy nhãn answerability/relation/source.
Spatial là nhãn yếu suy từ AMBIGUOUS, source là đa nhãn.

## Các giới hạn được ghi rõ

- Test-OOD không sử dụng.
- Baseline chính chỉ có RoboRefer RGB-D và best V2 trên Test-IID.
- Ma trận baseline/ablation rộng hơn trong form chưa có đầy đủ
  đối chứng khóa cùng protocol; bảng trạng thái không phải bảng kết quả.
- V3 đổi nhiều thành phần đồng thời, không phải ablation cô lập.
- File new/outputs/v3_three_seed_comparison.json hiện rỗng; không dùng
  file này để dựng bảng mean/std. Các thư mục run không tự chứng minh
  một bảng tổng hợp hợp lệ.
- CI đã lưu cho so sánh grounding dùng 200 family, 20000 lượt bootstrap;
  không tự mở rộng thành CI cho mọi metric.
- Region coverage trong báo cáo dùng điều kiện score <= frozen mass;
  phải phân biệt với kiểm tra giao trực tiếp region rời rạc và mask.
- Demo audit chưa nối MoveIt và không phát chuyển động robot.
- Resume lưu optimizer/scheduler, không lưu đầy đủ RNG/early-stopping
  state; không tuyên bố tái lập bitwise một run bị gián đoạn.
