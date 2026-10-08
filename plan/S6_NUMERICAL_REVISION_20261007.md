# S6 — revision kỹ thuật trước calibration

Lượt evaluation r1 dừng ở guard Bernoulli trước khi export toàn train, chưa
calibration hoặc IID. Tích phân relation bằng column mass float32 có thể vượt
1 rất nhỏ do sai số tổng softmax. R1 freeze và source snapshot được giữ tại
`new/outputs/pcrau_task_uncertainty_20261007/failed_evaluation_r1/` và
`neural_freeze.json`; không sửa nó thành run thành công.

R2 chuyển column mass sang float64, normalize đúng tổng1 trước tích phân.
Chỉ clip endpoints trong tolerance1e-12 của sai số máy; ngoài tolerance vẫn
raise. Không chỉnh confidence theo nhãn, không đổi kernel12px, model, loss,
checkpoint hoặc hyperparameters. Không train lại. R2 dùng thư mục mới
`evaluation_r2/`, freeze lại source đã sửa trước calibration. Protocol khoa
học S6 không đổi; các artifact r1 không dùng cho kết quả cuối.
