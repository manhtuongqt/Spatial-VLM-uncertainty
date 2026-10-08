# Answerability detail + augmentation: calibration fit-only

Không phải test hoặc audit độc lập.

| Answerability | Macro-F1 | Accuracy | FOUND recall | False FOUND | ABSENT→FOUND |
|---|---:|---:|---:|---:|---:|
| V2 | 0.810445 | 80.40% | 81.19% | 46 | 6 |
| Adapter mới | 0.843358 | 84.90% | 87.14% | 31 | 7 |

Ca hợp lệ bị sai answerability: 73 → 52 trên411 ca.

| Policy | Budget | Nhận đúng / 411 | Recall hợp lệ | Nhận | Lỗi | Risk | Bypass đúng/sai |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_found | 5.0% | 305/411 | 74.21% | 313 | 8 | 2.56% | 0/0 |
| hard_found | 7.5% | 330/411 | 80.29% | 346 | 16 | 4.62% | 0/0 |
| hard_found | 10.0% | 350/411 | 85.16% | 376 | 26 | 6.91% | 0/0 |
| risk_only | 5.0% | 305/411 | 74.21% | 313 | 8 | 2.56% | 0/0 |
| risk_only | 7.5% | 330/411 | 80.29% | 346 | 16 | 4.62% | 0/0 |
| risk_only | 10.0% | 350/411 | 85.16% | 376 | 26 | 6.91% | 3/1 |

Calibrator chọn theo calibration CV Brier: logistic33. Không chọn lại head/epoch.
V2 grounding/source/edge và mọi annotation evaluator bất biến đã đối chiếu.
Default vận hành thử nghiệm hard_found/budget7.5% sau đối chiếu calibration; giữ10% comparator. Không chọn lại head/epoch.
Không thay model/calibrator chính, hình hoặc LaTeX. Các lỗi còn lại không được tuyên bố đã giải quyết hết.

Giới hạn: recall ABSENT calibration 64.39% →59.85%, ABSENT→FOUND6→7; macro-F1 ABSENT vẫn tăng nhờ precision.
Ca hợp lệ: sửa29 ca sai answerability nhưng hồi quy8 ca, tăng ròng21; còn52 ca, gồm46 thiếu evidence và6 ABSENT.
