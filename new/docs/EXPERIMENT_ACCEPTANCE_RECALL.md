# Tăng recall chấp nhận đúng ở ngân sách lỗi 5%

Mục tiêu là tăng số mẫu FOUND thật và MAP đúng được chấp nhận, không ép
p(FOUND) cao hay đổi nhãn non-FOUND. Người dùng yêu cầu tăng 261/411 trên
calibration; những ca calibration này chỉ nêu vấn đề, không làm dữ liệu train.

Vòng DEVELOPMENT_ONLY này học binary error head trên train 320 family/1600
sample và chọn bằng dev 80 family/400 sample. V2, grounding, answerability,
source và edge đều đóng băng. Không dùng calibration/test để train hoặc
chọn nhánh. Không đổi model chính, LaTeX, hình hoặc phát lệnh robot.

Error = true answerability != FOUND OR frozen MAP outside target. Oracle
chỉ tạo nhãn supervision, không nằm trong feature. Feature gồm 26 evidence
quan sát được và tùy nhánh thêm 768 context của answer head V2 đóng băng.
Không đưa family ID, sample ID, variant hoặc annotation vào đầu scorer.

Ba nhánh cố định: evidence_bce (26), context_bce (794), context_bce_rank
(794). MLP 64 hidden, dropout 0.3; AdamW lr 3e-4, weight decay 1e-3, clip 5,
25 epoch, seed 24082026, batch 4 family. Ranking weight 0.1 cho nhánh cuối;
loss = binary CE + weight*mean softplus(score_good-score_bad).
Standardization chỉ fit train. Cùng seed, thứ tự batch cho các nhánh.

Đối chứng raw 1-p(FOUND), logistic18/33 fit train-only. Không lấy calibrator
đã học calibration làm đối chứng dev. Tại epoch 5,10,15,20,25 (1-based),
quét ngưỡng dev cho risk-only policy; chọn ngưỡng lớn nhất có Wilson upper
95% <=5% và >=20 family chấp nhận. Scorer chưa hiệu chuẩn, nên ngưỡng là
ngưỡng score, không phải xác suất risk đã khóa. Không chạy robot.

Chọn ứng viên có đúng chấp nhận nhiều nhất, lỗi thấp nhất, AURC thấp nhất;
chỉ qua cổng nếu đúng chấp nhận cao hơn đối chứng tốt nhất, số lỗi không
cao hơn đối chứng tốt nhất, không nhận ABSENT, và AURC không kém raw V2.
Giữ thêm bảng tại cùng số mẫu chấp nhận với raw để kiểm tra ranking.
Đây là lựa chọn trên dev có nhiều so sánh, không phải bảo đảm 5% hoặc
ước lượng test. Nếu không qua cổng, không coi là cải thiện.

Lưu checkpoint scorer riêng, score/sample ID, cổng chọn và hash nguồn.
Không fit calibration trong vòng này. Nếu có ứng viên, bước sau cần freeze
scorer rồi calibration riêng và test mới; calibration đã quan sát không
còn là audit độc lập. Không tuyên bố 261/411 đã tăng từ kết quả dev.

Chạy:
```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/experiment_acceptance_recall.py \
  --run-name pcrau_acceptance_recall_dev_20261003
```
