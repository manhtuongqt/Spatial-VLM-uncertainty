# Test-IID: adapter answerability, profile 7,5%

**Kết quả: answerability cải thiện, nhưng profile chưa đạt ngân sách lỗi 7,5% trên Test-IID.**

Chạy checkpoint/config/calibrator đã khóa; hard_found, threshold 0.2611932834526145. Không fit lại hoặc chọn ngưỡng bằng test. Bộ Test-IID này đã được phân tích trong lịch sử, không phải một test mới chưa từng quan sát.

| Chỉ số | V2 lịch sử | Adapter mới |
|---|---:|---:|
| Accuracy answerability | 76.80% | 81.10% |
| Macro-F1 | 0.775162 | 0.809686 |
| Recall FOUND | 82.14% | 86.90% |
| False FOUND | 70 | 55 |
| Chấp nhận | 255 | 361 |
| Chấp nhận đúng / 405 ca hợp lệ | 249 | 327 |
| Recall ca hợp lệ | 61.48% | 80.74% |
| Lỗi chấp nhận | 6 | 34 |
| Selective risk | 2.35% | 9.42% |
| Coverage | 25.50% | 36.10% |

Hai policy có calibrator/ngân sách/ngưỡng khác nhau; không phải đối chứng cô lập head.

## Lỗi và bỏ sót

- Lỗi chấp nhận theo nhãn thật: {'FOUND': 5, 'INSUFFICIENT_EVIDENCE': 22, 'ABSENT': 7}. FOUND ở đây là điểm MAP sai.
- 78 ca hợp lệ bị từ chối: {'ABSENT': 16, 'INSUFFICIENT_EVIDENCE': 32, 'FOUND': 30}. FOUND ở đây bị cổng risk chặn.
- Ca hợp lệ bị sai answerability: 66 → 48; sửa 30, hồi quy 12.
- Actions: {'EXECUTE': 361, 'REOBSERVE': 347, 'ASK_USER': 134, 'ABSTAIN': 158}.

## Bất định thống kê

Bootstrap 20000 lượt theo 200 family, giữ threshold cố định: selective risk CI95% [6.37%, 12.53%]; delta macro-F1 CI95% [0.015739, 0.053925].
Wilson upper dùng cách tính cấp mẫu là 12.87%; không phải bảo đảm an toàn theo family hoặc robot.

## Risk và grounding

Risk Brier 0.077108; NLL 0.257066; ECE10 0.031080; AURC 0.255748; oracle AURC 0.229231; excess AURC 0.026517.
Grounding giữ nguyên 775/835 = 92,81%; source/edge, toàn bộ spatial predictions và evaluator labels đối chiếu bằng nhau. Không có cải thiện grounding mới từ adapter answerability.

## Kết luận

Adapter cải thiện phân loại và tăng nhận ca hợp lệ, nhưng risk test 9,42% vượt ngân sách 7,5%. Giữ profile để truy vết thử nghiệm; chưa xác nhận dùng như profile đáp ứng ngân sách. Không đổi threshold, train lại, thay V2 chính thức, cập nhật hình/LaTeX hoặc chạy robot trong lần test này. Nếu dùng lỗi test để phát triển vòng sau, cần bộ IID độc lập mới cho đánh giá xác nhận.
