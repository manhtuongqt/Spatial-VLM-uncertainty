# Bổ sung profile fit toàn bộ calibration sau audit đã khóa

Audit tách family ở EXPERIMENT_HARD_CASES_SELECTIVE.md đã hoàn tất và được giữ
nguyên: phần threshold 300 sample không có ngưỡng thỏa tiêu chí Wilson 5%.
Không sửa ngưỡng/cổng trong audit đó hoặc lấy audit chọn lại mô hình.

Người dùng yêu cầu fit calibration riêng và chọn mức đánh đổi. Phần bổ sung
này dùng toàn bộ 200 family/1000 calibration sample sau freeze, tương tự
quy mô protocol gốc. Model vẫn là checkpoint đã freeze từ bước trước;
feature set logistic33 và l2=0.01 đã được chọn ở fit-only CV, không thay đổi.
Đối chứng logistic18 V2, l2=0.001 trên đúng cùng 1000 sample.

Crossfit 5-fold family, standardization fit từng training fold. Ngưỡng chọn
trên risk OOF, budget 5%, 7.5%, 10%, ít nhất 60 family được nhận; hard_found
và risk_only có threshold riêng theo tập nhận của policy. Lưu thêm logistic
fit toàn bộ dữ liệu để inference; ngưỡng fit OOF áp dụng model fit-all như
protocol gốc. Risk OOF và risk fit-all khác nhau, nên lưu counts của cả hai.

Đây là CALIBRATION_FIT_ONLY, không phải audit độc lập. Phần audit trước đó
đã tham gia fit của profile mới, nên tuyệt đối không đánh giá profile mới
trên audit cũ rồi gọi độc lập. Không chọn profile/hyperparameter bằng test;
không dùng profile mới để thay V2 chính hoặc chạy robot. Cần test độc lập
mới để đo coverage/risk thực tế của profile đã fit.

Không đồng nhất budget chọn ngưỡng với risk threshold hoặc rủi ro robot.
Wilson còn giới hạn do phụ thuộc variant/family và nhiều ngưỡng đã khảo sát.
