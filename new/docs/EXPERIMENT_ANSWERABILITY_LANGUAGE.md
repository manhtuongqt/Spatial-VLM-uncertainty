# Answerability detail với augmentation ngôn ngữ train-only

Head detail tăng macro-F1 nhưng chưa qua cổng ABSENT→FOUND. Vòng này giữ
kiến trúc, hai nhánh loss, seed, 25 epoch và cổng chọn của DETAIL; thêm
biến thể ngôn ngữ train-only, không thay cổng khi thấy kết quả.

Mỗi câu train giữ bản gốc và thêm ba prefix: Please; In this scene; Can you.
Giữ nguyên toàn bộ câu gốc, không thay target noun, attribute hoặc relation.
QC bắt buộc không truncate câu gốc, relation IDs/masks và anchor mask giống
hệt. Annotation giữ nguyên vì câu vẫn hỏi đúng cùng target/relations.
Các biến thể là augmentation, không phải capture, family hoặc bằng chứng
độc lập mới. Chỉ thay token IDs/mask, không thay RGB/depth feature/mask.
Feature pre-projector RGB/depth là image-only nên dùng lại đúng cache ảnh.

1600 train samples/320 family thành6400 trình bày ngôn ngữ; dev400/80 giữ
nguyên. Mỗi batch4 family chứa5 variant×4 diễn đạt/family=80 mẫu. Số update
mỗi epoch vẫn80, mean/scale chỉ fit trên training presentations. Loss class
weights từ1600 train gốc; tỉ lệ lớp không đổi khi tăng đều4 diễn đạt.

Mục tiêu là tăng đa dạng của observable query features, đặc biệt các ca
base answerability đổi sai dưới diễn đạt tương đương. Không dùng class
hoặc variant ID như feature. Mọi augmenter/model boundary đều có test QC.
Calibration chỉ được mở sau freeze nếu có ứng viên qua cổng dev cũ.
Không sử dụng Test-IID/OOD hoặc robot. V2 chính/LaTeX/figures bất biến.

```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 .conda-roborefer/bin/python new/scripts/experiment_answerability_detail.py \
 --language-augmentation --run-name pcrau_answerability_language_dev_20261003
```

## Sau khi qua cổng dev

Khóa checkpoint/config/hash rồi mới mở calibration. Đánh giá answerability
trên1000 calibration sample và đối chiếu prediction V2 cùng sample. Kiểm tra
grounding/source/edge không thay đổi. So sánh logistic18(l2=.001) và33(.01),
5-fold family; chọn calibrator bằng CV Brier rồi fit đủ calibration. Không
đổi epoch/head/feature theo kết quả calibration. Chọn profile ngân sách
5/7.5/10% trên risk OOF, Wilson upper<=budget và >=60 accepted family.

Đối chiếu action counts với nhánh acceptance residual34 cũ trên cùng data;
đây là so sánh pipeline, không ablation cô lập calibrator. Default mới
hard_found/budget10% để tách sửa classifier khỏi bypass cổng lớp. Risk-only
lưu làm đối chứng. Cả hai đều chưa là test độc lập hoặc robot success.
Decision CLI kiểm tra checkpoint hash để không dùng nhầm prediction V2 cũ
với calibrator mới. Main V2 và các artifact chính thức giữ nguyên.

```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 .conda-roborefer/bin/python new/scripts/calibrate_answerability_detail.py \
 --run-root new/outputs/pcrau_answerability_language_dev_20261003
```

Sau kết quả fit-only, lựa chọn vận hành đổi default thành hard_found/7.5%:
330 ca hợp lệ/16 lỗi so với profile10% cũ307/22. Không đổi head, calibrator
hoặc các threshold đã fit; giữ đủ cả ba budget và10% comparator. Đây là
lựa chọn trên calibration đã quan sát, không phải xác nhận trên test.

## Runtime mới

```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 .conda-roborefer/bin/python new/scripts/evaluate.py \
 --config new/outputs/pcrau_answerability_language_dev_20261003/config.json \
 --checkpoint new/outputs/pcrau_answerability_language_dev_20261003/detail_cost/checkpoints/best/model.safetensors \
 --split dev --smoke \
 --output-dir new/outputs/pcrau_answerability_language_dev_20261003/runtime_dev_smoke

PYTHONPATH=new/src .conda-roborefer/bin/python new/scripts/decide_experimental_profile.py \
 --predictions new/outputs/pcrau_answerability_language_dev_20261003/runtime_dev_smoke/predictions.jsonl \
 --calibrator new/outputs/pcrau_answerability_language_dev_20261003/calibration_full_fit/selected_calibrator.json \
 --profiles new/outputs/pcrau_answerability_language_dev_20261003/calibration_full_fit/profiles.json \
 --output new/outputs/pcrau_answerability_language_dev_20261003/runtime_dev_smoke/decisions.jsonl
```

Mặc định hard_found/7.5%. Thêm `--risk-target 0.05` hoặc `--risk-target 0.1` để đổi budget;
thêm --policy risk_only chỉ để dùng comparator đã lưu. CLI không nối robot.
