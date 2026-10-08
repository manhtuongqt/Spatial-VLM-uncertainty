# Báo cáo tiến độ P-CRA-U

File chính: `bao_cao_tien_do.tex`.

Báo cáo có cấu trúc bài toán, phương pháp, kết quả thực nghiệm và kết luận.
Phương pháp trình bày kiến trúc hiện tại, kiểm chứng hình học, uncertainty nhiệm
vụ, Kendall & Gal (NeurIPS 2017), MC Dropout và calibration. Phần kết quả và
kết luận dùng các hình/số liệu trong PowerPoint được cung cấp.

## Biên dịch

Chạy từ thư mục `baocao1`, dùng XeLaTeX hai lần:

```bash
cd /home/dhcn/ur_ws/src/myproject/baocao1
xelatex -interaction=nonstopmode -halt-on-error bao_cao_tien_do.tex
xelatex -interaction=nonstopmode -halt-on-error bao_cao_tien_do.tex
```

Hoặc `latexmk -xelatex bao_cao_tien_do.tex`.

Trên Overleaf: tải `.tex` và `figures/` lên cùng project, chọn file trên làm
Main document và XeLaTeX làm compiler. Bibliography đã nằm trong file `.tex`,
không cần BibTeX. Các font/package dùng từ TeX Live thông dụng.

## Ảnh

Báo cáo sử dụng 5 ảnh PowerPoint trong `figures/`, trích nguyên từ các slides
4/5/6/7/9. Tất cả figure dùng đường dẫn tương đối `figures/...`. Không cần
truy cập Downloads hoặc workspace gốc khi biên dịch.

`bao_cao_tien_do.pdf` là bản biên dịch kiểm tra của cùng file nguồn.

Ví dụ figure trong file nguồn:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/ppt_slide_05_image_02.png}
  \caption{Grounding theo quan he.}
  \label{fig:grounding_relation}
\end{figure}
```
