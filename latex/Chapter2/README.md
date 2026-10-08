# Chương 2: nội dung và trích dẫn

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


chapter2.tex có đủ 15 mục theo formkhoaluan.md. Nội dung đã đối chiếu với
model, text processing, hậu xử lý, calibration và policy V2 trong workspace.
Các khóa trích dẫn nằm trong latex/references.bib, gồm 18 nguồn nghiên cứu
và tài liệu chính thức.

Hiện latex/ chưa có file master hoặc preamble và máy chưa có trình biên dịch
LaTeX. Cấu trúc source được kiểm tra; dàn trang PDF chưa được xác nhận.

Preamble cần:

    \usepackage{amsmath,amssymb}

Nếu file master đặt trong latex/ và dùng BibTeX truyền thống, đặt cuối
tài liệu trước \end{document}:

    \bibliographystyle{ieeetr}
    \bibliography{references}

Nếu template dùng biblatex, giữ cơ chế đó, thêm
\addbibresource{references.bib} ở preamble và \printbibliography cuối tài liệu.
Không dùng đồng thời BibTeX truyền thống và biblatex.

Hình reliability, risk–coverage và confidence region dựa trên prediction
Test-IID được dành cho Chương 5. Chương 2 trình bày nền tảng và định nghĩa,
không sử dụng số liệu thực nghiệm như ví dụ lý thuyết không ghi nguồn.
