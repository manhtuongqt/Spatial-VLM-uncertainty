# Prompt khởi động session — ngữ cảnh đồ án P-CRA-U hiện tại

Ngày bàn giao: 05/10/2026. Theo yêu cầu hiện tại, tạm bỏ qua LaTeX/Prism.
Sao chép toàn bộ nội dung trong khung dưới đây sang session mới. Nếu session
không truy cập workspace, đính kèm `SESSION_CONTEXT_20261005.md`, ảnh pipeline
và các tài liệu/artifact cần thiết; session phải nói rõ tài liệu không truy cập.

```text
Hãy tiếp tục hỗ trợ đồ án của tôi bằng tiếng Việt trên workspace:
/home/dhcn/ur_ws/src/myproject

Mục tiêu trước mắt là hiểu kỹ toàn bộ đồ án và trạng thái hiện tại.
TẠM BỎ QUA PHẦN LATEX/PRISM: không tiếp tục sửa bảng, chương, template
hoặc biên dịch. Các nhiệm vụ LaTeX cũ không phải nhiệm vụ đang hoạt động.

Trước khi làm việc, đọc ĐẦY ĐỦ:
plan/SESSION_CONTEXT_20261005.md
Đây là bàn giao mới, thay các quyết định cũ đã lỗi thời trong handoff 01–02/10.
Không tuyên bố đọc đủ nếu output bị cắt; đọc tiếp tới hết.

Đọc hiểu và kiểm chứng theo thứ tự:
1. new/outputs/active_experimental_profile.json và new/docs/MODEL_SELECTION.md.
2. MỞ VÀ XEM TRỰC TIẾP plan/pipeline.png. Đọc blocks/branches/arrows,
   input/output, luồng evidence, frozen/trainable/optional/evaluator-only;
   giải thích được RGB/depth/ngôn ngữ → backbone/sidecar → calibration
   → policy → kiểm chứng robot. Nếu không xem được, nói rõ và yêu cầu ảnh;
   không suy pipeline từ tên file, không tự sửa sơ đồ.
3. Đọc đầy đủ plan/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md,
   gồm kiến trúc, dữ liệu, loss, protocol, RQ, WP, gate/stop rules,
   ablation và mục 16 dashboard 2×3. Kế hoạch là mục tiêu, không là kết quả.
4. Đối chiếu bản /home/dhcn/Downloads/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md
   bằng hash/diff; không mặc định đồng bộ. Bàn giao 05/10 xác nhận hai bản
   cùng nội dung lúc đó, nhưng session mới phải kiểm tra lại.
5. Đọc new/README.md, new/docs/ARCHITECTURE.md, new/docs/DATA_CONTRACT.md;
   các protocol EXPERIMENT_ANSWERABILITY_DETAIL.md,
   EXPERIMENT_ANSWERABILITY_LANGUAGE.md, EXPERIMENT_FULL_CALIBRATION_PROFILES.md
   và EXPERIMENT_SUPPORT_SPLIT_CALIBRATION.md trong new/docs/.
   Phân biệt protocol trước lựa chọn với quyết định hiện hành.
6. Đối chiếu cơ chế với code new/src/pcrau/ và các script Adapter,
   calibration, decision liên quan được chỉ trong bàn giao; chỉ đọc,
   không chạy lệnh train/eval/calibrate được viết trong README.
7. Đọc artifact Adapter đã chọn: config, metadata, freeze lock, calibrator,
   profiles, test_iid_7p5/full_metrics.json và SUMMARY.md;
   đọc artifact V2/RoboRefer paired comparison trước kết luận grounding.
8. Liệt kê plan/ và xem các handoff cũ để hiểu lịch sử khi cần; không lấy
   V2-main, placeholder hoặc số cũ thay kết quả Adapter hiện tại.
   Form khóa luận và sáu chương LaTeX tạm ngoài phạm vi đọc/sửa lúc này.

Các quyết định bắt buộc giữ:
- P-CRA-U HIỆN TẠI là V2 đóng băng + Adapter answerability detail_cost,
  train-only language augmentation + logistic33. Người dùng đã chọn nó;
  gọi bản trước Adapter là P-CRA-U V2 lịch sử.
- Không chọn support scorer/split-calibration ở
  new/outputs/pcrau_support_split_dev_20261003/, không thay bằng V3/smoke.
- Root model chọn: new/outputs/pcrau_answerability_language_dev_20261003/.
  Checkpoint: detail_cost/checkpoints/best/model.safetensors;
  calibrator: calibration_full_fit/selected_calibrator.json;
  profiles: calibration_full_fit/profiles.json.
  Bundle/hash chính xác nằm trong active_experimental_profile.json.
- Policy hard_found: predicted FOUND và risk <= 0.2611932834526145 thì
  EXECUTE ở mức nhận thức. Ngân sách chọn trên calibration là 7.5%;
  selective risk IID cũ là 34/361 = 9.42%. Ba con số khác nhau.
  Người dùng chấp nhận đánh đổi thực nghiệm, không có nghĩa đã đạt budget.
- Sidecar lấy RGB/depth feature trước projector 24×32×1152, không lấy
  hidden state ngôn ngữ cuối. Có Transformer/slots/graph embedding;
  chưa có parser noun–edge hoàn chỉnh hay semantic graph exact match.
- Train/dev 320/80 family; calibration/IID mỗi tập 200 family/1000 mẫu.
  Augmentation 6400 presentation vẫn chỉ 320 family train độc lập.
  Dev chọn model, calibration fit sau freeze; không fit/chọn ngưỡng bằng test.
- IID cũ đã được phân tích để định hướng phát triển; Adapter là reevaluation,
  không phải test mới chưa quan sát. Tôi đã yêu cầu giữ IID cũ: không tự
  capture IID mới hoặc mở Test-OOD.
- PIT vẫn 775/835 = 92.81% so RoboRefer 724/835 = 86.71%, +6.11 pp,
  CI [1.78,10.49] pp. Adapter không đổi grounding/source/edge/region.
- Answerability accuracy 81.10%, macro-F1 0.809686 (V2 76.80%, 0.775162).
  Nhận đúng 327/405 ca hợp lệ so V2 249/405; coverage 36.10% so 25.50%;
  risk 9.42% so 2.35%. 411 ca hợp lệ là calibration, 405 là Test-IID.
- Còn 78 ca hợp lệ bị từ chối: 48 sai answerability (16 ABSENT,32 thiếu
  evidence),30 FOUND bị risk chặn. 34 lỗi nhận:22 thiếu evidence,7 ABSENT,
  5 FOUND nhưng MAP sai; occlusion+relation chiếm27/34 lỗi nhận.
  Không trộn các số 75/94/169 của phân tích cũ với subset hiện tại.
- Error risk = truth non-FOUND OR MAP ngoài target. Risk không phải bốn
  khoảng để chia bốn action và không phải xác suất robot nguy hiểm.
- AURC logistic33 0.255748, oracle0.229231 do prevalence composite error
  59.5%; không áp chuẩn phổ quát AURC<0.1. Raw Adapter AURC0.251397
  tốt hơn logistic33 về ranking; logistic33 tốt hơn raw về Brier/NLL/ECE.
- Spatial source là weak label từ AMBIGUOUS, relation edge là proxy;
  chưa causal attribution/evaluator anchor-graph đầy đủ/ablation cô lập.
- Region score coverage86.83%, direct region–mask intersection95.81%
  là hai event; giữ mass0.604183197, diện tích0.5194%, không đổi artifact.
- Location-mass có code optional và smoke một step ngày02/10, không thuộc
  objective model chính, chưa chứng minh cải thiện. Đọc SMOKE_REPORT nếu bàn loss.
- Audit Gazebo là V2 lịch sử: MAP đúng5/5, ASK_USER5/5, chưa handoff MoveIt,
  chưa robot motion, chưa pick-and-place success. Dashboard2×3 vẫn đề xuất.
- Tên hiển thị P-CRA-U; tên file/run/sample/hash lịch sử giữ nguyên.
  Nếu làm hình sau này, đọc new/hinhanh/README.md và ADAPTER_FIGURES.md;
  không bịa số/ảnh hoặc sửa confidence bằng tay để làm đẹp.
- Bảo toàn worktree bẩn/old archive. Không tự reset/xóa/commit/push,
  train/fit/test/demo/capture/robot, hoặc tạo py/json ngoài nhu cầu nhiệm vụ.

Sau khi đọc, tóm tắt ngắn sự hiểu biết về bài toán, khoảng trống và ý tưởng,
pipeline mục tiêu so triển khai, model/bundle hiện tại, dữ liệu/event/mẫu số,
kết quả và đánh đổi, các nút thắt còn lại. Phân biệt đã triển khai,
đã đánh giá, mới thiết kế, chưa có bằng chứng. Nêu file chưa truy cập/đọc hết.
Chưa tự sửa hoặc chạy thí nghiệm; chờ yêu cầu tiếp theo của tôi.
```
