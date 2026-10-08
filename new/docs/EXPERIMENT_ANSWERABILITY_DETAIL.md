# Answerability theo từng target/anchor/relation

Mục tiêu người dùng: giảm FOUND thật bị dự đoán ABSENT/INSUFFICIENT_EVIDENCE,
không thay bằng việc tăng risk threshold. V2 và mọi đầu grounding/source/edge
đóng băng. Học residual answerability adapter từ train, chọn trên dev.
Không dùng calibration/Test-IID để train, chọn feature hoặc epoch.

Các vòng context cũ chỉ cung cấp 26 evidence + global768, thường làm tăng
ABSENT→FOUND. Vòng này đưa thêm từng slot/node/moment dự đoán:
global768 + target query128 + relation queries384 + anchor queries384
+ target node128 + anchor nodes384 + target moments5 + anchor moments15
=2196 detail; cộng26 evidence thành2222. Không dùng oracle mask/IDs/variant.
Slot/node/moment là biểu diễn dự đoán, chưa phải graph parser đã kiểm chứng.

Residual MLP2222→64→4, dropout0.3, output zero-init; chuẩn hóa train-only.
Hai nhánh cùng seed24082026, 25 epoch, 4 family/batch, AdamWlr3e-4,
weight decay1e-3, clip5:
- detail_control: inverse train-class weighted CE;
- detail_cost: CE như trên, trueFOUND bị base đoán sai weight2,
  trueABSENT bị base đoánFOUND weight3, cộng0.1*falseFOUND penalty
  (ABSENT weight2, INSUFFICIENT_EVIDENCE weight0.5) và0.01*mean(delta²).

Cổng trước khi chạy: macro-F1 >V2, FOUND recall >V2; tổng falseFOUND<=18,
ABSENT→FOUND<=4; recall cả ba lớp còn lại không thấp hơn V2. Chọn macro-F1
cao nhất qua cổng; tie ưu tiên recallFOUND, ít falseFOUND, epoch sớm.
Nếu không qua, lưu diagnostic và giữ V2. Không nới cổng khi thấy kết quả.
Budgets5/7.5/10 là lỗi policy, không dùng như tỉ lệ lỗi classifier bốn lớp.

Freeze scorer/model trước calibration nếu có ứng viên. Calibration bộ cũ
chỉ fit/crossfit, không là audit mới; mọi claim test cần bộ độc lập mới.
Không sửa checkpoint/calibrator V2, figure19, LaTeX hoặc robot/world.

```
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 .conda-roborefer/bin/python new/scripts/experiment_answerability_detail.py \
 --run-name pcrau_answerability_detail_dev_20261003
```
