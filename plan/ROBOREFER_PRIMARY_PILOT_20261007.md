# RoboRefer làm grounding chính — pilot RefCOCO

Ngày 07/10/2026. Biến thể opt-in, chưa thay active profile.

## Thay đổi đã triển khai

RGB/depth → RoboRefer đầy đủ → một normalized point → đầu ra target cuối.
Song song, pre-projector features → frozen Tasks60 Sidecar → diagnostics/anchor.
Parser + điểm target RoboRefer + predicted anchor peak → kiểm tra trái/phải dạng point evidence.
Direct/unsupported không bịa kết quả kiểm chứng. Không tạo heatmap giả từ point.

Entry point live: `new/scripts/infer_roborefer_primary.py`; adapter: `new/src/pcrau/roborefer_primary.py`.
Sidecar target, answerability và policy cũ không thay/veto đầu ra grounding mới.

## Kết quả replay pilot đã quan sát

| Đầu ra target | Point-in-box |
|---|---:|
| Sidecar trước thay đổi | 101/300 (33.67%) |
| RoboRefer primary | 281/300 (93.67%) |

Delta +60.00 pp; 95% CI theo 100 image groups [53.333333333333336, 66.33333333333333] pp.
Sửa 187 câu, làm sai 7 câu so với Sidecar. Không chọn nhánh theo nhãn thật.
Điểm primary khớp 300/300 với parser điểm official và đúng bằng đối chứng RoboRefer RGB-D đã chạy.
**Đây là lấy lại năng lực backbone, không là cải thiện RoboRefer hoặc bằng chứng uncertainty cải thiện grounding.**
Thiết kế được chọn sau khi thấy pilot cũ; replay dùng nguyên predictions đã lưu, không test độc lập mới.

![Case thật](../new/outputs/pcrau_roborefer_primary_pilot300_20261007/cases.png)

Xanh lá: GT bbox hậu kiểm; đỏ: Sidecar; cyan: primary. Có cả case sửa và case bị làm sai.

## Calibration và MC: phần chưa được chuyển sang target mới

Primary risk/threshold là null; output valid báo `REVIEW_UNCALIBRATED`, không tự EXECUTE.
Invalid/multiple/out-of-range point không fallback Sidecar mà báo `REOBSERVE_INVALID_POINT`.
Các risk và MC distributions cũ chỉ nằm trong `sidecar_diagnostic`, gắn rõ event/nhánh chúng đo.
Không gọi entropy map Sidecar là uncertainty của điểm RoboRefer. Điểm deterministic không tự cung cấp phân bố.
Geometry dùng point target thật, không đẩy Dirac/gaussian giả vào calibrator44/60.

Để hoàn thiện quyết định: freeze nhánh primary; tạo evidence train/dev/calibration từ đúng primary point;
fit calibrator mới trên calibration với event non-FOUND OR điểm RoboRefer ngoài target mask, chọn threshold ở calibration;
sau đó reevaluate IID cũ. Không fit/chọn ngưỡng trên pilot RefCOCO.
Các semantic/depth/completion quantities phụ thuộc vị trí cần được lấy lại tại vị trí primary trước khi gọi là uncertainty cho target mới.
Spatial MC hiện là bất định của Sidecar; muốn bất định grounding backbone phải bổ sung cơ chế phân bố có căn cứ.

## Kiểm chứng và bảo toàn

Adapter kiểm tra prompt alignment, coordinate boundary, không oracle và không đổi outputs Sidecar.
Live worker dùng cùng RGB-D640×480, frozen model, greedy max_new_tokens128, features FP16, inventory hash được kiểm tra.
Không train/fit ở bước này; original pilot, bundle, calibrator và active profile được kiểm tra hash không đổi.
Chưa đưa robot chuyển động. Kết quả này là chất lượng candidate grounding; coverage/risk của pipeline mới còn chưa đánh giá.

## Kiểm chứng live sau replay

Đã chạy lại RGB/depth của ảnh3518 với `Locate the top banana.` bằng entry point mới, không dùng cached answer làm input. RoboRefer sinh `[(0.397, 0.228)]`; pixel640×480 [254,109], đổi về ảnh gốc640×427 là [254,97], nằm trên banana phía trên. Sidecar MAP [330,370] vẫn nằm phía dưới.

Generated answer và feature SHA256 khớp với pilot cũ. JSON live: `new/outputs/pcrau_roborefer_primary_pilot300_20261007/live_top_banana.json`; validation: `live_validation.json`. 10 contract tests đạt; hash bundle/active profile và toàn bộ source predictions không đổi. Primary risk vẫn null, không robot motion.
