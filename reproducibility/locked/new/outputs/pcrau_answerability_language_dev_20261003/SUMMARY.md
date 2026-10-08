# Answerability detail trên train/dev

| Nhánh | Macro-F1 | Recall FOUND | False FOUND | ABSENT→FOUND | Qua cổng |
|---|---:|---:|---:|---:|---|
| V2 | 0.821410 | 83.33% | 18 | 4 | True |
| detail_control | 0.865365 | 87.50% | 8 | 4 | True |
| detail_cost | 0.872512 | 86.90% | 9 | 4 | True |

Chọn: detail_cost.
Grounding dev326/335, source/edge và V2 parameters bất biến đã kiểm tra.
Chọn trên dev nhiều vòng, chưa chứng minh calibration/test hoặc sửa hết lỗi answerability.
