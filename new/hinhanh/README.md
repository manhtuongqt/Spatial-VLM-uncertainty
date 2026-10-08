# Hình khoa học từ Test-IID đã đóng băng

## Bản bổ sung P-CRA-U được chọn ngày 2026-10-03

Người dùng đã chọn **P-CRA-U = V2 + Adapter answerability + logistic33**.
Trong 12 thư mục 10, 11, 14–23 có thêm PNG/PDF tiếng Việt với hậu tố
`_adapter_vi`; các hình cũ bên dưới vẫn là bản V2 lịch sử và được giữ nguyên.
Xem [ADAPTER_FIGURES.md](ADAPTER_FIGURES.md) để lấy đủ đường dẫn, nguồn dữ liệu
và các chỉ số thay đổi hoặc giữ nguyên. Mã dựng riêng:
`build_adapter_figures.py`.

Hình 20 mới giữ bố cục hai panel nhưng dùng loss train Adapter và chỉ số
answerability dev vì log Adapter không lưu loss dev theo epoch. Ngưỡng risk
đã khóa là 0,261193; risk thực nghiệm trên IID cũ là 9,42%, khác ngân sách
calibration 7,5%. Các hình này chưa được thay vào nguồn LaTeX.

## Hình V2 lịch sử

Các hình trong thư mục này được dựng từ RGB của bộ Gazebo Test-IID, mask đánh
giá, prediction của RoboRefer gốc và best V2, cùng quyết định của calibrator.
Không chạy huấn luyện hoặc suy luận lại. Mỗi hình có PNG để xem và PDF có chữ,
trục, điểm vẽ dạng vector để chèn vào LaTeX. RGB là ảnh render/capture từ Gazebo,
không phải ảnh chụp môi trường vật lý hay ảnh sinh bằng AI.

Tên hiển thị trên các hình là **P-CRA-U** (checkpoint lịch sử V2).
Tên file, sample ID, đường dẫn run và khóa dữ liệu vẫn giữ nguyên để truy vết.

## Danh mục và gợi ý caption

| Mục form | Hình xem nhanh | Chú thích đề nghị |
|---|---|---|
| 4 | [Language Query Graph](04_language_query_graph/fig04_language_query_graph_v2.png) | Sơ đồ khớp code V2 với câu lệnh Test-IID `v211iid_family_000007__clean`: token hóa, mã hóa Transformer, nhận diện `nearer_than`, kích hoạt một relation slot và một anchor slot rồi cross-attend vào text. Các slot là biểu diễn học được; hệ thống không xuất nhãn danh từ–node đã parse. |
| 5 | [Relation-conditioned fusion](05_relation_conditioned_fusion/fig05_relation_conditioned_fusion_v2.png) | RGB/depth feature được chiếu về 128 chiều, ghép với relation context để tạo gate và nhánh cross-modal, sau đó qua LayerNorm và hai residual MLP. Gate mean 0,5039 là prediction thật của cùng ca; không phải bản đồ gate theo pixel. |
| 6 | [Target heatmap](06_target_heatmap/fig06_target_heatmap_real_rgb.png) | Ảnh RGB Test-IID, target mask chuẩn và phân bố xác suất target của best V2. Vòng xanh dương là điểm RoboRefer; dấu × cam là MAP của V2. Mẫu `v211iid_family_000127__clean` (FOUND, fruit). Heatmap 24×32 được nội suy chỉ để hiển thị, colorbar là xác suất trên ô lưới gốc. |
| 7 | [Phân bố nhiều đỉnh](07_multimodal_distribution/fig07_multimodal_spatial_distribution.png) | Ca `v211iid_family_000031__clean` có truth `AMBIGUOUS`, số mode được prediction báo là 2. Hai vòng trắng đánh dấu hai cực đại địa phương tách biệt lớn nhất trên lưới 24×32; đường xanh lá là target mask chuẩn. Cực đại không được coi tự động là định danh vật thể. |
| 8 | [Source uncertainty](08_source_uncertainty/fig08_source_uncertainty_cases.png) | Năm ca RGB Test-IID với xác suất của năm nguồn bất định. Chấm cạnh tên nguồn là nhãn dương; đường dọc là ngưỡng 0,5 đã đóng băng. `spatial` là nhãn yếu suy từ `AMBIGUOUS`. Các ca lần lượt là `000127__clean`, `000165__clean`, `000127__depth_corruption`, `000094__occlusion_view_counterfactual`, `000127__semantic_counterfactual`. Ở (b), spatial thực là 0,9659, hiển thị 0,97; thay sample minh họa, không hạ prediction bằng tay. Ở (d), hộp súp che khối lập phương màu cam, ô zoom lặp lại vùng RGB thật; occlusion thực 0,8069, hiển thị 0,81. Metadata của 000094 ghi physical occluder là hộp súp. Ca cuối minh họa false positive semantic: xác suất cao nhưng nhãn nguồn bằng 0. |
| 9 | [Independent calibrator](09_independent_calibrator/fig09_independent_calibrator_v2.png) | Mười tám đặc trưng evidence đi vào bộ chuẩn hóa và logistic calibrator độc lập; risk, confidence region và luật quyết định nằm sau đó. Ngưỡng risk 0,0937, region mass 0,6042 và ca REOBSERVE là artifact/prediction thật. |
| 10 | [Reliability diagram](10_reliability/fig10_reliability_test_iid.png) | Tỷ lệ grounding error quan sát so với calibrated risk trên 1.000 mẫu Test-IID; 10 bin độ rộng bằng nhau, kích cỡ điểm tỷ lệ với số mẫu, histogram dưới cho biết số mẫu mỗi bin. ECE = 0,04647. |
| 11 | [Risk–coverage curve](11_risk_coverage/fig11_risk_coverage_test_iid.png) | Selective error khi nhận dần mẫu theo risk tăng dần. Ngưỡng đã chọn trên calibration cho coverage 25,5% và error 2,35% trên Test-IID; AURC = 0,26045. |
| 12 | [Spatial confidence region](12_confidence_region/fig12_confidence_region_true_rgb.png) | Hai ví dụ vùng highest-density trên lưới: giao target (`000127__clean`) và trượt target (`000002__clean`). Ô tím là những ô được lưu trong quyết định, đường xanh lá là target mask. Coverage tổng thể 86,83% trên 835 mẫu có target so với danh nghĩa 90%. |
| 14 | [Bốn quyết định](14_decision_cases/fig14_four_policy_decisions.png) | Ví dụ thật cho EXECUTE (`000127__clean`), REOBSERVE (`000003__depth_corruption`), ASK_USER (`000031__clean`), ABSTAIN (`000005__clean`). Hình ghi truth/predicted answerability và risk. Chỉ EXECUTE có dấu điểm thực thi; các hành động khác là quyết định từ ảnh tĩnh, không chứng minh recovery hay robot motion. |
| 15 | [Đúng/sai paired](15_failure_cases/fig15_paired_success_and_failure.png) | Ba ca có target: V2 đúng–RoboRefer sai (`000007__clean`), RoboRefer đúng–V2 sai (`000002__clean`), cả hai sai (`000165__relation_counterfactual`). Toàn bộ 835 ca có target có 107 V2-only, 56 RoboRefer-only, 4 neither. |
| Bổ trợ | [Theo nhóm vật thể](16_stratified_object/fig16_grounding_by_object.png) | Biểu đồ cột ghép point-in-target accuracy theo nhóm vật thể, RoboRefer gốc và best V2. Số trên cột là tỷ lệ phần trăm; `n` là số variant có target trong nhóm. Không hiển thị khoảng tin cậy trên hình. |
| Bổ trợ | [Theo quan hệ](17_stratified_relation/fig17_grounding_by_relation.png) | Cùng định nghĩa trên, phân tầng theo quan hệ. |
| Bổ trợ | [Theo variant](18_stratified_variant/fig18_grounding_by_variant.png) | Cùng định nghĩa trên, phân tầng theo 5 loại Test-IID variant. |
| Chương 5, mục 5.3 | [Confusion matrix answerability](19_answerability_confusion/fig19_answerability_confusion_test_iid.png) | Hai panel: số đếm và phần trăm chuẩn hóa theo hàng nhãn thật; hàng là ground-truth, cột là prediction, tổng 1000 sample. Đường chéo của panel phần trăm là recall từng lớp. |
| Chương 5, mục 5.8 | [Đường cong huấn luyện](20_training_curves/fig20_pcrau_training_curves.png) | Train/dev total loss và dev point-in-target trên 335 sample có target; đánh dấu epoch 13/step 1120, chọn theo dev total loss. Epoch từ 0, nên epoch 13 là lượt huấn luyện thứ 14. |
| Chương 5, mục 5.5 | [Thay đổi source paired](21_source_delta/fig21_pcrau_paired_source_delta.png) | Heatmap trung bình xác suất variant trừ clean, ghép theo 200 family cho mỗi variant. Ô viền đen là source tương ứng, số cạnh hàng đếm cặp có source tương ứng tăng. Spatial là nhãn yếu; chưa có CI, không diễn giải nhân quả. |
| Chương 5, mục 5.4 | [Uncertainty–localization error](22_uncertainty_localization/fig22_pcrau_uncertainty_localization.png) | Ba scatter trên 420 FOUND thật: entropy chuẩn hóa, 1 trừ peak margin và calibrated risk so với khoảng cách tới mask. Giữ 405 khoảng cách 0 và 15 khoảng cách dương; không jitter/outlier removal. Pearson dùng pixel gốc, Spearman dùng average ranks cho ties; trục khoảng cách symlog chỉ phục vụ hiển thị. |
| Chương 5, mục 5.7 | [Coverage–area](23_coverage_area/fig23_pcrau_coverage_area.png) | Quét 201 mass, giữ mass đã khóa 0,6041832. Hai panel theo mass và diện tích trung bình; hai đường là score event đã lưu và giao vùng–mask trực tiếp. Điểm khóa: diện tích 0,5194%, score coverage 86,83%, direct intersection 95,81%. Không dùng test để chọn lại mass. |
| Chương 5, mục quan hệ | [Anchor/edge một mẫu](24_anchor_edge/fig24_anchor_edge_test_iid.png) | Ca `v211iid_family_000121__clean`: RGB với mask hậu kiểm, kích hoạt anchor slot 1 và cạnh `behind` hoạt động, score 0,9935. Anchor logits được trích xuất từ checkpoint best V2 và feature cache đã khóa vì prediction JSONL không lưu lưới anchor; score cạnh hoạt động và MAP được đối chiếu với prediction gốc. Nhãn cạnh là proxy; hình không thay metric anchor/graph. |

`000127__clean` trong bảng là dạng viết ngắn của
`v211iid_family_000127__clean`; các ID đầy đủ có trong hình hoặc script.

## Nguồn và giới hạn diễn giải

Hình huấn luyện đọc history.jsonl và best.json của run
new/outputs/pcrau_target_v2_full_seed_24082026; không dùng Test-IID để chọn epoch.

- Manifest và đường dẫn RGB/mask: `new/test_iid/protocol/test_iid_eval_manifest.json`.
- Prediction V2: `new/test_iid/evaluation/best_v2/predictions.jsonl`.
- Risk, action và region: `new/test_iid/evaluation/best_v2/decisions.jsonl`.
- Metrics: `new/test_iid/evaluation/best_v2/full_metrics.json`.
- Điểm RoboRefer, paired outcomes và bootstrap CI: `new/test_iid/evaluation/comparison_original_vs_best_v2/paired_samples.csv` và `full_comparison.json`.
- CI 95% bootstrap theo family vẫn nằm trong `full_comparison.json`, chỉ được lược khỏi ba biểu đồ cột để dễ đọc.
- Định nghĩa risk error: truth khác `FOUND` **hoặc** MAP point ngoài target.
- Hình 15 chỉ minh họa định tính; các tỷ lệ kết luận phải lấy từ toàn bộ 200 family.
- Không có anchor probability grid trong prediction đã lưu nên hình 6 chỉ thể hiện target heatmap.
- Dữ liệu Test-IID dùng threshold source chung 0,5 và không chứa RoboRefer disagreement thực trong prediction V2.
- Hình 4, 5, 9 là sơ đồ phương pháp từ code V2 và giá trị artifact đã lưu; không phải ảnh chụp tensor nội bộ hay bằng chứng parser định danh từng vật thể.
- Hình 22 chỉ xét nhãn thật FOUND để không trộn localization với non-FOUND. Pearson/Spearman lần lượt: entropy 0,014807/0,011660; 1 trừ peak margin 0,014595/−0,011101; calibrated risk 0,264108/0,225600. Đây là hệ số mô tả, không có CI/p-value theo family.
- Hình 23 dùng score đã lưu của evaluator cho đường score, không thay bằng score tính lại. Direct intersection dùng đúng thứ tự NumPy và quy tắc nhận cả ô vượt mass của policy, với target support lấy từ các block mask 20×20. Tại mass khóa, toàn bộ tập ô tái dựng được đối chiếu với decision đã lưu.
- Hình 24 là ngoại lệ có trích xuất lại một tensor chưa lưu từ checkpoint/feature cache đã khóa, không huấn luyện hay chỉnh ngưỡng. Tái tạo riêng bằng `PYTHONPATH=new/src .conda-roborefer/bin/python3.10 new/hinhanh/build_anchor_edge_figure.py`.
- Sáu sample có thứ tự ô đồng xác suất khác giữa torch.argsort của evaluator và NumPy argsort của policy. Ngoài khác biệt ties, score event và giao region–mask còn khác ở ô vượt biên mass. Vì vậy 800/835 direct intersections không thay cho số liệu 725/835 score coverage trong báo cáo gốc.

Tái tạo toàn bộ bằng:

```bash
cd /home/dhcn/ur_ws/src/myproject
.conda-roborefer/bin/python3.10 new/hinhanh/build_figures.py
.conda-roborefer/bin/python3.10 new/hinhanh/build_method_figures.py
```
