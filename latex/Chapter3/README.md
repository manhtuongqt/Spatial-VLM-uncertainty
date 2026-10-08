# Chương 3 — Phương pháp đề xuất

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


chapter3.tex có 17 mục theo plan/formkhoaluan.md, mô tả implementation
P-CRA-U V2 và artifact best checkpoint epoch 13. Chương phân biệt rõ thành
phần đã triển khai với thiết kế mục tiêu: query slot không phải noun parser;
fusion dùng gate/MLP; cạnh có nhãn proxy; policy chưa có geometry gate.
Các mở rộng V3 không được mô tả như thành phần đã dùng trong baseline V2.

## Hình

Chương có 9 khối figure với caption, label và nguồn:

1. Kiến trúc mục tiêu: plan/pipeline.png.
2. Language Query Graph: hình 04.
3. Relation-conditioned fusion: hình 05.
4. Target heatmap: hình 06.
5. Phân bố nhiều đỉnh: hình 07.
6. Source uncertainty: hình 08.
7. Independent calibrator: hình 09.
8. Confidence region: hình 12.
9. Bốn quyết định: hình 14.

Các hình 04–14 dùng PDF hiện có trong new/hinhanh, không dựng ảnh mới.
Ảnh RGB bên trong là capture/render Gazebo, không phải ảnh chụp ngoài đời.
Prediction Test-IID trong hình chỉ minh họa phương pháp đã đóng băng;
không được dùng điều chỉnh mô hình/ngưỡng.

Mặc định đường dẫn tính từ thư mục latex/; source cũng kiểm tra để
hỗ trợ biên dịch từ workspace root. Nếu chuyển file sang Overleaf,
cần chuyển kèm các asset và điều chỉnh \pcrauFigureRoot cùng đường
dẫn pipeline.png.

## Tích hợp vào tài liệu

Preamble của file master cần:

    \usepackage{amsmath,amssymb,graphicx}

Dùng cơ chế hỗ trợ tiếng Việt của template hiện tại. Các khóa trích dẫn
đã nằm trong latex/references.bib; cấu hình bibliography theo template,
không dùng đồng thời BibTeX truyền thống và biblatex.

Hiện workspace chưa có file master LaTeX và máy chưa có trình biên dịch.
Đã kiểm tra cấu trúc source, nhãn/tham chiếu, khóa trích dẫn và sự tồn tại
đường dẫn hình. Chưa kiểm chứng dàn trang PDF, ngắt trang và tràn công thức.
Nhãn các chương cần duy nhất khi tích hợp toàn luận văn. Chương 4 dùng
sec:experimental-setup, tách khỏi sec:experiments của Chương 5.

## Nguồn đối chiếu implementation

- new/src/pcrau/model.py, text.py, dataset.py, losses.py.
- new/src/pcrau/engine.py, postprocess.py, calibration.py, policy.py.
- new/configs/pcrau_target_v2.json, new/scripts/train.py.
- old/protocol/pcra_u_development_common.py: pooling tower về 24×32.
- old/protocol/wp3_feature_hook_smoke.py: trích xuất tower đóng băng.
- new/demo_gazebo/capture_ros_rgbd.py: chuyển metric sang relative depth.
- new/ketqua/02_huan_luyen_v2/calibration/calibrator.json: artifact đã fit.
- new/hinhanh/README.md: ý nghĩa, nguồn và giới hạn của các hình.
