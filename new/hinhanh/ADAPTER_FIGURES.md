# Hình tiếng Việt cho P-CRA-U được chọn (V2 + Adapter)

Dựng từ prediction Test-IID đã lưu, không huấn luyện, suy luận lại hoặc fit ngưỡng.
Mỗi thư mục được thêm một PNG và một PDF với hậu tố `_adapter_vi`; hình cũ giữ nguyên.

| Hình | PNG | PDF |
|---|---|---|
| 10 | [fig10_reliability_test_iid_adapter_vi.png](10_reliability/fig10_reliability_test_iid_adapter_vi.png) | [PDF](10_reliability/fig10_reliability_test_iid_adapter_vi.pdf) |
| 11 | [fig11_risk_coverage_test_iid_adapter_vi.png](11_risk_coverage/fig11_risk_coverage_test_iid_adapter_vi.png) | [PDF](11_risk_coverage/fig11_risk_coverage_test_iid_adapter_vi.pdf) |
| 14 | [fig14_four_policy_decisions_adapter_vi.png](14_decision_cases/fig14_four_policy_decisions_adapter_vi.png) | [PDF](14_decision_cases/fig14_four_policy_decisions_adapter_vi.pdf) |
| 15 | [fig15_paired_success_and_failure_adapter_vi.png](15_failure_cases/fig15_paired_success_and_failure_adapter_vi.png) | [PDF](15_failure_cases/fig15_paired_success_and_failure_adapter_vi.pdf) |
| 16 | [fig16_grounding_by_object_adapter_vi.png](16_stratified_object/fig16_grounding_by_object_adapter_vi.png) | [PDF](16_stratified_object/fig16_grounding_by_object_adapter_vi.pdf) |
| 17 | [fig17_grounding_by_relation_adapter_vi.png](17_stratified_relation/fig17_grounding_by_relation_adapter_vi.png) | [PDF](17_stratified_relation/fig17_grounding_by_relation_adapter_vi.pdf) |
| 18 | [fig18_grounding_by_variant_adapter_vi.png](18_stratified_variant/fig18_grounding_by_variant_adapter_vi.png) | [PDF](18_stratified_variant/fig18_grounding_by_variant_adapter_vi.pdf) |
| 19 | [fig19_answerability_confusion_test_iid_adapter_vi.png](19_answerability_confusion/fig19_answerability_confusion_test_iid_adapter_vi.png) | [PDF](19_answerability_confusion/fig19_answerability_confusion_test_iid_adapter_vi.pdf) |
| 20 | [fig20_pcrau_training_curves_adapter_vi.png](20_training_curves/fig20_pcrau_training_curves_adapter_vi.png) | [PDF](20_training_curves/fig20_pcrau_training_curves_adapter_vi.pdf) |
| 21 | [fig21_pcrau_paired_source_delta_adapter_vi.png](21_source_delta/fig21_pcrau_paired_source_delta_adapter_vi.png) | [PDF](21_source_delta/fig21_pcrau_paired_source_delta_adapter_vi.pdf) |
| 22 | [fig22_pcrau_uncertainty_localization_adapter_vi.png](22_uncertainty_localization/fig22_pcrau_uncertainty_localization_adapter_vi.png) | [PDF](22_uncertainty_localization/fig22_pcrau_uncertainty_localization_adapter_vi.pdf) |
| 23 | [fig23_pcrau_coverage_area_adapter_vi.png](23_coverage_area/fig23_pcrau_coverage_area_adapter_vi.png) | [PDF](23_coverage_area/fig23_pcrau_coverage_area_adapter_vi.pdf) |

## Nguồn và phạm vi số liệu

- Checkpoint: `/home/dhcn/ur_ws/src/myproject/new/outputs/pcrau_answerability_language_dev_20261003/detail_cost/checkpoints/best/model.safetensors`.
- SHA-256 checkpoint: `5a326a8040bfbabfadf3f57fff51fd90be3845438c9a3a8c6ee41bdd1e0fa045`.
- Calibrator logistic33: `/home/dhcn/ur_ws/src/myproject/new/outputs/pcrau_answerability_language_dev_20261003/calibration_full_fit/selected_calibrator.json`.
- SHA-256 calibrator: `19e0c35424144cf753109978b9d08ddaa16edcadbe0c1dc1515103fb88b8de7d`.
- Prediction: `/home/dhcn/ur_ws/src/myproject/new/outputs/pcrau_answerability_language_dev_20261003/test_iid_7p5/inference/predictions.jsonl`.
- Quyết định: `/home/dhcn/ur_ws/src/myproject/new/outputs/pcrau_answerability_language_dev_20261003/test_iid_7p5/runtime_decisions.jsonl`.
- Báo cáo: `/home/dhcn/ur_ws/src/myproject/new/outputs/pcrau_answerability_language_dev_20261003/test_iid_7p5/full_metrics.json`.
- Hình 20: `/home/dhcn/ur_ws/src/myproject/new/outputs/pcrau_answerability_language_dev_20261003/detail_cost/history.jsonl`, metadata của checkpoint best và freeze lock.
- RGB, mask hậu kiểm và điểm RoboRefer: manifest Test-IID cùng paired_samples.csv của phép so sánh gốc.

## Đối chiếu nội dung

- Hình 10: ECE risk = 0,03108; 10 khoảng độ rộng bằng nhau, kích cỡ điểm theo số mẫu.
- Hình 11: AURC = 0,255748; policy nhận 361/1000 mẫu, có 34 lỗi (9,42%). Ngưỡng risk = 0,261193.
  Đường cong xếp hạng risk trên toàn bộ mẫu; điểm cam là policy còn yêu cầu dự đoán FOUND.
  Chỉ lọc risk cho 362 mẫu, khác 361 mẫu được policy nhận. Không ép điểm policy lên đường xếp hạng.
- Hình 14: giữ đúng bốn sample cũ; cập nhật answerability, risk và quyết định từ output Adapter.
- Hình 15–18: grounding giống hệt V2 lịch sử trên 1000 sample đã đối chiếu; giữ số và mẫu minh họa.
- Hình 19: ma trận Adapter, accuracy 81,10%, macro-F1 0,809686; nhãn thật có thể trả lời 420 mẫu.
- Hình 20: giữ bố cục hai panel, dùng loss train Adapter và accuracy/macro-F1 answerability dev.
  Log không lưu loss dev theo epoch. Không dựng đường loss dev giả hoặc dùng lịch sử train V2 cho Adapter.
  Epoch chọn 13, step 1120 theo macro-F1 dev dưới các điều kiện kiểm soát recall/false FOUND; không chọn bằng dev total loss.
- Hình 21: source head đóng băng, paired delta không đổi; xác nhận từ prediction Adapter.
- Hình 22: giữ 420 nhãn thật FOUND (405 MAP đúng, 15 sai); panel risk dùng logistic33 mới.
- Hình 23: spatial distribution không đổi; giữ mass đã fit 0,6041832 của nhánh không gian V2.
  Score coverage 86,83%, direct intersection 95,81%, diện tích trung bình 0,5194%.
  Logistic33 không fit lại spatial mass; vùng ô lưới được kiểm tra với quyết định spatial gốc.

Bộ IID này đã được phân tích trong lịch sử. Ngân sách chọn ngưỡng calibration 7,5% chưa đạt trên test;
risk thực nghiệm 9,42% là mức đánh đổi người dùng chấp nhận. Quyết định trên ảnh chưa phải robot motion/task success.
Chưa sửa LaTeX hoặc các hình nằm ngoài 12 thư mục được yêu cầu.

## Tái tạo và kiểm tra

```bash
.conda-roborefer/bin/python3.10 new/hinhanh/build_adapter_figures.py
```

Script mặc định từ chối ghi đè output đã tồn tại; `--refresh-existing` chỉ dựng lại bản `_adapter_vi`.
Đã kiểm tra hash của checkpoint/config/calibrator/profiles,
hash prediction/decision, cùng sample ID và các đầu ra đóng băng; đối chiếu ma trận, ECE và AURC.
Đã kiểm tra 24 PNG/PDF cũ trong 12 thư mục giữ nguyên SHA-256.

| Output mới | SHA-256 |
|---|---|
| `10_reliability/fig10_reliability_test_iid_adapter_vi.pdf` | `676506526d0ce1f893e4e2a9fd24698c91856021c1691ab8e35cd22853a6d4d3` |
| `10_reliability/fig10_reliability_test_iid_adapter_vi.png` | `518afd6c7ea61b06f9f9bff13ae09252f795d35208a259d1b603914133501520` |
| `11_risk_coverage/fig11_risk_coverage_test_iid_adapter_vi.pdf` | `ef4ed516c1a9220b5a008e239682cd1aeacbcdd2178823ba9775dd2e029a1e1d` |
| `11_risk_coverage/fig11_risk_coverage_test_iid_adapter_vi.png` | `f34114f04862cd823bc94d46a691de80921aeb553b34a0622bda5889588d0e1c` |
| `14_decision_cases/fig14_four_policy_decisions_adapter_vi.pdf` | `e8a3db94dce41fff983aba2a1f6d413a970b69c733bfe26f15bcbb11abce186f` |
| `14_decision_cases/fig14_four_policy_decisions_adapter_vi.png` | `a48f7b5caabd14fe3468805b9aa44164fde018b3c66247ddd8ca6da5e6c19707` |
| `15_failure_cases/fig15_paired_success_and_failure_adapter_vi.pdf` | `abe4769b38a5696b4a9be1da22fe8e6c726fe7e024c970356e19706a38e8262c` |
| `15_failure_cases/fig15_paired_success_and_failure_adapter_vi.png` | `e41233e34d7e9dc1cbb1517b6285e2ade6464a67d13f1f895b994952e0d8684b` |
| `16_stratified_object/fig16_grounding_by_object_adapter_vi.pdf` | `52db7eb16f979e2c862391d267971e0735f08fb71589fdf393e1b397d3899e3b` |
| `16_stratified_object/fig16_grounding_by_object_adapter_vi.png` | `a7fcbd34b0afbebb0180acec7d97c97e023bbc923dcd1602d6f325a59929b672` |
| `17_stratified_relation/fig17_grounding_by_relation_adapter_vi.pdf` | `f5ebebf2bedf6ab570516c37774f1a3633e301e75c7b43069e37d483b52f1de6` |
| `17_stratified_relation/fig17_grounding_by_relation_adapter_vi.png` | `14913d44add22a1e1d0924f26f2e3f71b5e13c38ec29acdd130adaf5d6b8c468` |
| `18_stratified_variant/fig18_grounding_by_variant_adapter_vi.pdf` | `fedb2172827c3feee2d2daa6f6560ffed6743a78ec72e40b88842dc2581d7188` |
| `18_stratified_variant/fig18_grounding_by_variant_adapter_vi.png` | `3fd04ee28af9075c959f29399687697f0f8dfd460a234f962738dec51af94779` |
| `19_answerability_confusion/fig19_answerability_confusion_test_iid_adapter_vi.pdf` | `0a64a5ecf4076eb7704fbc1506592ee18b9ada3fb90ecfa18e95a6b5e833989e` |
| `19_answerability_confusion/fig19_answerability_confusion_test_iid_adapter_vi.png` | `44913b75581e18ef02e0a39a6e626310e3865b88cae4a1f41c587475d0fbcfce` |
| `20_training_curves/fig20_pcrau_training_curves_adapter_vi.pdf` | `61e997a84203287d6a641dd73249aa1d5b7cade3308df102df2dafd2cd9f04ed` |
| `20_training_curves/fig20_pcrau_training_curves_adapter_vi.png` | `be91e3c28fc6d6f76a63ccc756b51c83a4e1598025f63e779ec2007b8a5c9eed` |
| `21_source_delta/fig21_pcrau_paired_source_delta_adapter_vi.pdf` | `f3cf99af57ad8f8840c3eacc02ff6c6aaf901a2b429323084c0a24941165f371` |
| `21_source_delta/fig21_pcrau_paired_source_delta_adapter_vi.png` | `b0d62e0cea13a123a35d069c88f9c26382bbec6cddce859e3491f98500de3f42` |
| `22_uncertainty_localization/fig22_pcrau_uncertainty_localization_adapter_vi.pdf` | `0ce8eefbb187e5a7c593b848f8b2278c8e99341f84c8aa5a195f8dc3fd78c5c0` |
| `22_uncertainty_localization/fig22_pcrau_uncertainty_localization_adapter_vi.png` | `d4ef1dd36216aa2c87d60b68a331608c6a8e8eaaf70809ff280823f0cdc3a035` |
| `23_coverage_area/fig23_pcrau_coverage_area_adapter_vi.pdf` | `66fa97ff29c23c8b4573ff2ef5e3da8adff81e201dabfaff0c2e7d1a03f8d2f5` |
| `23_coverage_area/fig23_pcrau_coverage_area_adapter_vi.png` | `1f475a9f1a84a702652a2dfd0e773e7887ff8a2e20fe547123edc8c5685ca878` |
