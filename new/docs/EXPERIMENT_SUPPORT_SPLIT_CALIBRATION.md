# Evidence cho quyết định chấp nhận, calibration tách family

Vòng nghiên cứu được đặt ra sau khi đã quan sát lỗi của Test-IID lịch sử.
Không dùng các sample/label/feature test đó để train hoặc chọn mô hình.
Giữ adapter answerability ngôn ngữ và tất cả V2 frozen. Main V2 bất biến.

Train 320 family, dev 80 family; dùng cache train-only 4 cách diễn đạt.
Học score lỗi composite non-FOUND OR MAP sai, không đổi nhãn answerability
của FOUND có điểm sai. Không dùng variant/sample/family/annotation làm feature.
Train mining: positive hợp lệ nhưng non-FOUND probability cao hoặc lớp sai;
negative composite-error có dự đoán FOUND hoặc non-FOUND probability thấp.
Hard positive weight2, hard negative weight3; nhân1.5 cho các presentation
relation/occlusion trong training loss, không truyền variant vào inference.

Đối chứng cùng seed24082026, batch4 family×5 variant×4 wording,
25 epoch, AdamWlr3e-4/weight_decay1e-3, hidden64/dropout0.3:
detail_uniform; scalar_hard; detail_hard; support_hard.
BCE sample-weighted +0.2 pairwise ordering +0.01 residual-logit penalty.
Residual cộng logit non-FOUND của answerability đã freeze; zero-init.
Scalar dùng26 evidence+4 probability; detail thêm2196 slot/node/moment;
support thêm42 proxy query-node, target-anchor/pair separation/overlap,
language-active slots và source-edge/occlusion-peak interaction.
Các proxy không phải graph exact match, physical geometry hay metric TF.

Dev chọn checkpoint bằng số ca hợp lệ được nhận tại Wilson upper<=7.5%,
>=20 accepted family; tie dùng conditional AURC trong predicted FOUND,
Brier và epoch sớm. Gate: không giảm số nhận đúng so với raw score tham chiếu,
không tăng ABSENT/occlusion/relation accepted-error counts ở dev operating point,
và conditional AURC phải cải thiện. Không nới gate sau kết quả.
Đây là selection trên dev, không phải chứng minh ngân sách hoặc robot success.

Sau freeze mới đánh giá calibration đã quan sát trong lịch sử. Chia bằng
SHA256(seed|family_id) cố định100 fit /60 threshold /40 audit family.
Chỉ fit logistic scalar trên ranker logit bằng fit100, không chọn evidence
hay regularization theo threshold/audit. Giữ calibrator fit100 cho runtime:
không refit full200 sau khi chọn ngưỡng. Threshold chọn duy nhất budget7.5%
trên60 family, minimum20 accepted family. Audit40 mở sau lock threshold;
báo đầy đủ kết quả và không đổi ngưỡng khi audit fail. Calibration partition
là verification nội bộ đã quan sát, không là một test mới độc lập.

Thêm diagnostic theo source dự đoán và variant evaluator-only. Không đặt
cổng bằng tên variant. Tất cả reviewer audit gồm cả false positives và
false negatives; không xóa subgroup hoặc đổi lỗi theo grounding hit.

Bộ xác nhận IID mới phải có seed/layout/capture khác dữ liệu trước, cùng
protocol phân bố, registry/camera/depth, IDs và image hashes có kiểm tra
disjointness, annotation evaluator-only, feature pre-projector không oracle.
Khóa checkpoint/calibrator/threshold trước capture/inference; không dùng
bộ mới để fit hoặc chọn lại. Không robot task motion hoặc Test-OOD.
