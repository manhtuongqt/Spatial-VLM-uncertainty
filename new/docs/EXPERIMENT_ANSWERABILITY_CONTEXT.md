# Vòng 3: evidence cùng biểu diễn truy vấn V2 cố định

Vòng evidence-only sửa thêm đúng một sample dev, chưa cải thiện FOUND hoặc
ABSENT. Vòng này bổ sung biểu diễn global_feature 768 chiều V2 đã đóng băng
vào 26 evidence, qua residual MLP 794 → 64 → 4, dropout 0.3.
Đây là phương pháp mới trong run riêng, không kéo dài hoặc sửa tiêu chí run cũ.

Giữ nguyên toàn bộ V2 và các cổng ở EXPERIMENT_ANSWERABILITY_EVIDENCE.md:
macro-F1 cùng recall cả bốn lớp không thấp hơn V2, false FOUND không tăng.
Hai nhánh balanced/unweighted, cùng seed 24082026, 25 epoch, AdamW lr 3e-4,
weight decay 1e-3, 4 family/batch và clip norm 5. Mean/scale fit train-only.
Không sửa nhãn, không đưa metadata/mask vào forward, không fit theo test.

Mục tiêu: kiểm tra phần thông tin còn thiếu trong các summary evidence có
giúp giảm nhầm lẫn mà vẫn bảo toàn grounding/source hay không. Đây là lần
phát triển thứ ba trên dev; kết quả chọn dev chịu ảnh hưởng của nhiều lần
thử. Mọi nhánh và thất bại đều được báo cáo, không dùng dev thay test độc lập.
Nếu không qua cổng, ghi không đạt; không tự nới cổng hay tăng epoch.

```bash
PYTHONPATH=new/src PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .conda-roborefer/bin/python new/scripts/experiment_answerability_evidence.py \
  --context --run-name pcrau_answerability_context_dev_20261003
```
