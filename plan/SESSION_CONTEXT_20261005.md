# Bàn giao toàn bộ ngữ cảnh đồ án P-CRA-U — 05/10/2026

Workspace: `/home/dhcn/ur_ws/src/myproject`.

Tài liệu này bàn giao bài toán, định hướng, triển khai, lịch sử lựa chọn và bằng chứng thực nghiệm. Theo yêu cầu mới nhất, **tạm bỏ qua phần LaTeX**: không tiếp tục sửa chương, bảng, template hoặc Prism chỉ vì nhiệm vụ đó xuất hiện trước đây.

Đây là snapshot có nguồn kiểm chứng, không thay thế artifact. Đã đối chiếu trực tiếp ảnh pipeline, toàn bộ kế hoạch V3, tài liệu lựa chọn, code liên quan và artifact Adapter ngày 03/10. Hash checkpoint/config/calibrator/profiles khớp bản ghi đang hoạt động khi lập bàn giao. Không chạy train, fit calibration, đánh giá lại mô hình hoặc robot để tạo tài liệu này.

## 1. Những quyết định hiện hành có ưu tiên cao nhất

1. **P-CRA-U hiện tại = V2 đóng băng + Adapter answerability `detail_cost` có augmentation ngôn ngữ + calibrator logistic33.** Người dùng chọn ngày 03/10/2026, không chọn bộ score support/split-calibration thử nghiệm sau đó.
2. Gọi bản trước Adapter là **P-CRA-U V2 lịch sử**. Giữ nguyên tên file, run ID, sample ID và hash; không đổi tên hàng loạt để khớp tên hiển thị.
3. Policy hiện tại là `hard_found`, ngân sách chọn profile trên calibration **7,5%**, ngưỡng risk từng mẫu **0.2611932834526145**.
4. Trên IID cũ, nhận 361/1000 mẫu, đúng 327, lỗi 34: coverage **36,10%**, selective risk **9,42%**. Người dùng chấp nhận mức đánh đổi thực nghiệm này; **không có nghĩa profile đạt ngân sách 7,5% hoặc có bảo đảm an toàn 10%**.
5. Grounding/source/edge/spatial distribution giữ nguyên từ V2. Không coi Adapter đã tăng PIT thêm lần nữa.
6. Test-IID cũ đã được phân tích và dùng để định hướng phát triển. Adapter là **đánh giá lại trên IID đã quan sát**, không phải xác nhận bằng test mới chưa từng xem. Người dùng yêu cầu giữ IID cũ; không tự capture bộ mới.
7. Không tự mở Test-OOD, chạy robot, train tiếp, fit/chọn ngưỡng bằng test hoặc promote một thử nghiệm khác. Tài liệu bàn giao không phải lệnh thực hiện roadmap.
8. Bản ghi lựa chọn: `new/outputs/active_experimental_profile.json`; tài liệu diễn giải: `new/docs/MODEL_SELECTION.md`.

Các câu “mô hình chính vẫn là V2”, “không thay hình/LaTeX”, “chưa đánh giá Test-IID” trong protocol/summary cũ phản ánh thời điểm tạo experiment. Đọc theo lịch sử; quyết định đặt tên mới không làm thay đổi artifact cũ.

## 2. Bài toán, ý nghĩa và khoảng trống nghiên cứu

Tên phương pháp: **Probabilistic Cross-modal Relation-Aware Uncertainty (P-CRA-U)**.

Đồ án nghiên cứu spatial vision-language grounding trong cảnh thao tác trên bàn: từ RGB, relative depth và câu lệnh chỉ vật/quan hệ, định vị target và xác định quan sát có đủ điều kiện để chấp nhận hay cần can thiệp. Không xây foundation model mới, full scene graph hay LLM Agent tổng quát.

Trong runner RoboRefer đang dùng, baseline trả một điểm `p_RR=(u,v)`. Điểm đúng trên ảnh chưa tự cho biết:

- có một target duy nhất hay nhiều ứng viên tương đương;
- target có vắng mặt hoặc bị thiếu evidence không;
- evidence ngôn ngữ, quan hệ, depth, che khuất có đáng tin không;
- xác suất lỗi của quyết định chấp nhận là bao nhiêu.

Khoảng trống được xử lý ở **giao thức/hệ thống cụ thể đang triển khai**, không phải khẳng định mọi phiên bản RoboRefer hoặc toàn bộ literature đều không có uncertainty.

Hai đóng góp phương pháp được giữ ở mức phù hợp bằng chứng:

1. Sidecar có điều kiện theo truy vấn, nối target–relation–anchor representation với phân bố vị trí, answerability và evidence nguồn.
2. Hiệu chuẩn riêng sau freeze và quyết định chọn lọc, đánh giá chất lượng xác suất cùng đánh đổi lỗi–coverage.

Dataset theo family, counterfactual, cache/provenance, hình và Gazebo là thành phần hỗ trợ. Ý tưởng đề xuất là cách tổ chức/kết hợp/kiểm chứng trong hệ thống này; chưa có cơ sở khẳng định tính mới tuyệt đối hay đóng góp nhân quả riêng của mỗi head/loss. Các nghiên cứu thành phần đã tồn tại.

## 3. Pipeline mục tiêu phải xem trực tiếp

Ảnh: `plan/pipeline.png`. Khi khởi động session, mở bằng công cụ xem ảnh, không suy nội dung từ tên file.

Ảnh có bảy khối chính:

1. RGB, relative depth và câu lệnh.
2. RGB/depth encoders tạo `R0/D0`; các projector và language tokens tạo multimodal tokens cho đường backbone.
3. RoboRefer-2B-SFT đóng băng sinh điểm baseline `p_RR`.
4. Language query graph mục tiêu: target, relations, anchors; ví dụ banana–left_of–apple và banana–nearer_to_camera_than–mustard bottle.
5. Sidecar P-CRA-U: fusion theo quan hệ; query-centric relation head; spatial prediction head; uncertainty/answerability head. Các nhánh tạo anchor/edge, graph embedding, target distribution, MAP/top-k và source/answerability.
6. Independent calibrator đọc evidence, fit sau freeze; trả risk và vùng không gian tùy gate. Policy dùng risk/answerability/evidence để EXECUTE, REOBSERVE, ASK_USER, ABSTAIN.
7. Kiểm chứng Gazebo/UR3-compatible; đánh giá nhận thức riêng với task success downstream. Các mũi tên nét đứt quan sát lại/hỏi lại/từ chối là luồng can thiệp mục tiêu, không chứng minh vòng recovery đã chạy.

Chú giải ảnh phân biệt frozen, trainable, optional và evaluator-only. Đường metric depth + CameraInfo/TF là evidence hình học khác với relative depth đi vào tower. `p_RR` còn có nhánh disagreement với `p_PCRA`.

**Ranh giới triển khai:** sidecar thực tế lấy `R0/D0` **trước projector**, không lấy hidden state ngôn ngữ cuối. Parser danh từ–cạnh đầy đủ, geometry/TF trong calibrator, disagreement thực, robot handoff và dashboard 2×3 không được suy là đã hoàn thành từ sơ đồ. Không tự sửa pipeline để đồng nhất với hiện trạng.

## 4. Kế hoạch nghiên cứu và lịch sử

Kế hoạch đầy đủ: `plan/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md` (1556 dòng tại snapshot).

Bản IDE: `/home/dhcn/Downloads/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md`.
Hai bản có cùng SHA-256 khi bàn giao:
`a23eb2411218a8d68f97da48bfac8120936522fd091c07b49b9eecef40b50854`.
Session sau phải kiểm tra lại; không mặc định luôn đồng bộ. “Kế hoạch V3” khác experiment checkpoint “V3”.

Kế hoạch là bản thảo định hướng, không phải bằng chứng tất cả gate/WP đã đạt. Lõi gồm grounding, query-centric relation reasoning, spatial uncertainty, source scores, calibration và selective policy; P-CRA-F, LoRA, full graph, action-failure head và robot thật là mở rộng có điều kiện.

- WP0–WP2: baseline/hạ tầng, depth sensitivity và prototype dataset là bằng chứng lịch sử đã khóa. Prototype 50 family/250 mẫu không phải bộ train/dev/calibration/IID hiện tại.
- WP3: formalize scope/graph, novelty review, feature/cache và data audit.
- WP4: sidecar, spatial/answerability/source, development và ablation.
- WP5: freeze, calibration, locked evaluation.
- WP6: selective intervention/replay/dashboard.
- WP7: kiểm chứng robot sau infrastructure/offline gate.
- WP8: mở rộng tùy chọn và đóng artifact.

RQ lõi hỏi về grounding; uncertainty/answerability; phản ứng theo nguồn; calibration; selective utility. RQ6/P-CRA-F là tùy chọn. Kết quả hiện tại trả lời một phần, không xác nhận tất cả giả thuyết.

**Mục 16 dashboard 2×3 vẫn là đề xuất:** RGB+Depth | Language graph | Predicted graph / Spatial heatmap | Source uncertainty | Calibrated action. Dashboard demo hiện có bốn panel.

Các metric/gate/ablation/robot trial trong kế hoạch cần đối chiếu code và artifact. Một số quy mô/tiến độ/điểm xuất phát trong kế hoạch đã cũ. Không đọc phần đầu rồi tuyên bố đã nắm đầy đủ loss, protocol, WP và nghiệm thu.

`SESSION_HANDOFF_PROMPT_20261001.md`, `SESSION_CONTEXT_20261002.md`, `SESSION_START_PROMPT_20261002.md` là tài liệu lịch sử. Bản 02/10 còn đặt V2 là chính, còn thiếu một số hình lúc đó; không dùng thay quyết định Adapter ngày 03/10. `plan/formkhoaluan.md` phục vụ cấu trúc sáu chương; hiện tạm không cần đọc/sửa LaTeX.

## 5. Kiến trúc thực tế

Luồng hiện tại:

```text
RGB + relative depth + instruction
  → tower RoboRefer-2B-SFT đóng băng
  → R0_GRID/D0_GRID 24×32×1152 + thumbnails 1152
  → sidecar PCRAUTargetV2
     ├─ hashed-token Transformer + target/relation/anchor slots
     ├─ RGB/depth projection → gate/cross-modal residual fusion
     ├─ target/interior/anchor logits → spatial distributions/MAP
     ├─ predicted nodes/moments → edge evidence/graph embedding
     ├─ answerability V2 logits + residual Adapter → 4 lớp
     └─ source logits → 5 score đa nhãn
  → postprocess/evidence
  → logistic33 đã khóa → risk
  → hard_found + threshold → quyết định nhận thức
```

Các kích thước/cấu hình chính: hidden 128, text Transformer 2 lớp/4 attention heads, fusion 2 residual MLP, vocab hash 8192, max 72 token, tối đa 3 relation/3 anchor slot. Có capacity ba slot không có nghĩa đã kiểm chứng xử lý mọi câu nhiều mệnh đề; archive development dùng tối đa hai anchor/một relation theo contract.

Language dùng token hash và nhận diện relation/slot activation từ câu lệnh, không dùng hidden state cuối của LLM RoboRefer. Có learned slots và predicted graph embedding; chưa có parser noun–node–edge hoàn chỉnh/graph exact match đã đánh giá.

Phân bố target dùng softmax toàn lưới 768 ô để lấy MAP/entropy/modes. BCE+Dice supervision dùng sigmoid logits; không đồng nhất hai phép chuẩn hóa. Top-k mode không tự là danh tính instance. Heatmap sắc không tự chứng minh điểm đúng.

`MODEL_INPUT_KEYS`: `r0`, `d0`, `r_thumb`, `d_thumb`, `token_ids`, `token_mask`, `relation_ids`, `relation_mask`, `anchor_mask`. `anchor_mask` ở đây là cờ kích hoạt slot từ query, **không phải target/anchor mask ảnh oracle**. Annotation, ID/variant và mask evaluator chỉ dùng supervision/evaluator, không đi vào forward.

Metric depth, intrinsics, đăng ký RGB–depth, timestamp và TF phải được kiểm chứng riêng cho back-projection/handoff. Relative depth đầu vào tower không phải độ sâu mét. Covariance 2D đã lưu chưa phải vùng an toàn 3D đã hiệu chuẩn.

## 6. Dữ liệu và protocol

| Tập | Family | Mẫu gốc | Vai trò |
|---|---:|---:|---|
| Train | 320 | 1600 | Học sidecar/Adapter |
| Dev | 80 | 400 | Chọn checkpoint/ứng viên |
| Calibration | 200 | 1000 | Fit calibrator/chọn profile sau freeze |
| Test-IID cũ | 200 | 1000 | Đánh giá V2 khóa và đánh giá lại Adapter |

Mỗi family có năm variant: `clean`, `semantic_counterfactual`, `relation_counterfactual`, `depth_corruption`, `occlusion_view_counterfactual`. Split/CI/bootstrap phải giữ các variant cùng family đi cùng nhau; 1000 variant không phải 1000 đơn vị độc lập.

Train/dev/cache cũ nằm dưới `old/`, được xem là archive bất biến. Không sửa mask, nhãn, manifest hoặc dữ liệu archive để làm đẹp số.

Test-IID có RGB-D capture Gazebo riêng, canary 20 family trước full 200; không phải đổi tên dev thành test. Manifest inference và evaluator tách riêng. Ground-truth mask/semantic labels không làm đầu vào mô hình hay control.

Support answerability trên Test-IID: **420 FOUND, 105 AMBIGUOUS, 132 ABSENT, 343 INSUFFICIENT_EVIDENCE**.

Có **835** mẫu target mask không rỗng: 420 FOUND + 105 AMBIGUOUS + 310 INSUFFICIENT_EVIDENCE. Vì vậy 165 mẫu không có mask không đồng nghĩa 165 ABSENT. Mask AMBIGUOUS có thể là hợp ứng viên; PIT trên mask hợp chưa chứng minh chọn được một vật duy nhất.

Semantic counterfactual không mặc định là semantic source dương hoặc target absent; Test-IID hiện có toàn bộ semantic variant nhãn FOUND. Không lấy tên variant thay annotation thật.

Phân biệt các mẫu số: **411 ca hợp lệ là calibration**, **405 ca hợp lệ là Test-IID**, với “hợp lệ” = truth FOUND và MAP đúng. Không mang 261/411 hoặc câu chuyện 169 mẫu ở một phân tích/profile cũ sang kết quả IID hiện tại.

Test cũ đã ảnh hưởng hướng nghiên cứu thông qua failure audit. Dù sample test không dùng vào train/fit/chọn epoch/ngưỡng, đây vẫn không còn là phép xác nhận độc lập mới. Người dùng hiện giữ IID cũ; chỉ đề xuất nhu cầu test mới khi đánh giá claim, không tự tạo.

## 7. Loss V2 và thử nghiệm location-mass

Objective V2 lịch sử:

```text
L_V2 = 1.0 L_target(BCE+Dice)
     + 0.4 L_interior(BCE+Dice)
     + 0.6 L_anchor(BCE+Dice)
     + 0.4 L_edge(BCE)
     + 0.7 L_answerability(CE có class weights)
     + 0.5 L_source(BCE đa nhãn)
     + 0.2 L_counterfactual_ranking
```

Ranking margin 0.25; spatial weak-label weight 0.5. Nhiều loss ràng buộc các đầu ra khác nhau, không phải nhiều phép đo cùng một chất lượng. Total loss có thể lớn hơn 1; không coi đó là xác suất lỗi hoặc chứng minh train hỏng.

Kế hoạch đề xuất location-mass `-log(sum_{i thuộc M_target} H_i + eps)` và interior-mass; **V2/Adapter chính không được huấn luyện bằng location-mass**.

Code hiện **đã có** `masked_location_mass_loss` trong `new/src/pcrau/losses.py`, mặc định trọng số 0; một smoke có đối chứng đã chạy ngày 02/10:

- Config riêng: `new/configs/pcrau_target_v2_location_mass_smoke.json`, weight 1.0, epsilon 1e-8.
- Report: `new/outputs/pcrau_target_v2_location_mass_smoke_20261002T075421Z/SMOKE_REPORT.md`.
- Control: `new/outputs/pcrau_target_v2_control_smoke_20261002T075421Z/`.
- Mỗi run một batch train 20 mẫu, một batch dev 20 mẫu, một optimizer step từ model mới khởi tạo; không phải fine-tune chính hay full ablation.
- Smoke location-mass train/dev 3.311885/3.388450, total 9.343480/9.210021; control total 6.031595/5.821929. Cả hai dev grounding 0.05 trên batch nhỏ.
- Report ghi 13/13 test kỹ thuật đạt ở thời điểm đó. Chưa thêm interior-mass; interior vẫn BCE+Dice.

Ý nghĩa: công thức/gradient/tích hợp hữu hạn hoạt động, không phải bằng chứng tăng grounding. Hai objective khác nhau không so total loss để kết luận tốt/xấu. Không promote checkpoint smoke step 1 hay đưa smoke vào số liệu model chính.

## 8. Adapter hiện tại đã học gì

V2 gốc được đóng băng hoàn toàn, kể cả answer head; chỉ học residual Adapter. Logits mới = logits V2 + delta từ Adapter; output cuối khởi tạo bằng 0 nên ban đầu khớp V2.

Input Adapter **2222 chiều = 26 observable evidence + 2196 detail**:

- 26: target/interior/anchor confidence, peak/margin/entropy/top-mass, khoảng cách peak target–interior, edge score, fusion gate, RGB/depth thumbnail similarity/MAE, năm source score và bốn probability answerability V2.
- 2196: global context 768, target query 128, relation queries 384, anchor queries 384, target node 128, anchor nodes 384, target moments 5, anchor moments 15.

MLP `2222→64→4`, GELU/dropout 0.3, train-only mean/scale. Không có mask/nhãn/family/variant ID làm feature. Chuẩn hóa đã lưu trong checkpoint, không fit lại theo batch suy luận.

Train-only augmentation giữ câu gốc và thêm ba prefix `Please `, `In this scene, `, `Can you `. 1600 mẫu thành 6400 **trình bày**, vẫn 320 family độc lập; không thêm capture hay nhãn mới. Dev giữ nguyên 400. Có QC bảo toàn câu gốc, không truncate và không đổi relation/anchor activation.

Nhánh chọn `detail_cost`: weighted CE theo class và example; FOUND thật nhưng base đoán non-FOUND có weight 2, ABSENT thật nhưng base đoán FOUND weight 3. Mining báo 317 hard FOUND và 30 hard ABSENT **presentation**, không phải số family mới. Mining này không bắt buộc MAP đúng; không diễn giải nó thành chỉ dùng ca định vị đúng.

```text
L_adapter = weighted_CE(logits_V2 + delta, truth)
          + 0.1 * weighted_mean[-log(1-p_FOUND)] trên ABSENT/INSUFFICIENT
          + 0.01 * mean(delta²)
```

Penalty ABSENT weight 2, INSUFFICIENT_EVIDENCE weight 0.5. Giữ ví dụ âm khó để không chỉ tăng FOUND bằng mọi giá. Code source chính: `new/scripts/experiment_answerability_detail.py`.

Huấn luyện thực tế: seed 24082026, 25 epoch, AdamW lr 3e-4/weight decay 1e-3, clip 5, bốn family/batch; augmentation làm 80 presentation/batch nhưng vẫn 80 update/epoch. Best epoch 13 (zero-based), step 1120, chọn **dev macro-F1 dưới các cổng recall/false FOUND**, không phải dev total loss V2. Dev Adapter accuracy 87.25%, macro-F1 0.872512; V2 macro-F1 0.821410.

**Chú ý config kế thừa:** `config.json` còn các trường optimization/calibration V2 gốc. Script Adapter override optimizer/epoch và profile file mới chứa budget/threshold. Không lấy `optimization.learning_rate=0.0002`, `epochs=20` hoặc `calibration.risk_target=0.05` trong config kế thừa để báo sai run Adapter thực tế.

Best V2 và Adapter đều có epoch 13/step 1120 nhưng **khác checkpoint/hash và tiêu chí chọn**.

## 9. Định danh mô hình và artifact chính

Root hiện tại:
`new/outputs/pcrau_answerability_language_dev_20261003/`.

| Thành phần | Đường dẫn trong root |
|---|---|
| Config | `config.json` |
| Checkpoint chọn | `detail_cost/checkpoints/best/model.safetensors` |
| Metadata/history | `detail_cost/checkpoints/best/metadata.json`, `detail_cost/history.jsonl` |
| Model freeze | `freeze_lock.json` |
| Calibrator | `calibration_full_fit/selected_calibrator.json` |
| Profiles | `calibration_full_fit/profiles.json` |
| Calibration summary | `calibration_full_fit/SUMMARY.md` |
| IID metrics/summary | `test_iid_7p5/full_metrics.json`, `test_iid_7p5/SUMMARY.md` |
| IID prediction | `test_iid_7p5/inference/predictions.jsonl` |
| IID runtime decision | `test_iid_7p5/runtime_decisions.jsonl` |
| Evaluator-only decision | `test_iid_7p5/evaluation_decisions.jsonl` |
| Evaluation lock | `test_iid_7p5/evaluation_lock.json` |

SHA-256 snapshot:

```text
Adapter checkpoint: 5a326a8040bfbabfadf3f57fff51fd90be3845438c9a3a8c6ee41bdd1e0fa045
Adapter config:     b214670c6c4a679f755a9953e15bae4bbb71ee5eea1980fa8e3d64030d79f85b
Logistic33:         19e0c35424144cf753109978b9d08ddaa16edcadbe0c1dc1515103fb88b8de7d
Profiles:           df39fa55843d657111a2c0bee2fee68aad1f2c9ada4c2fc257527eb420f6c9ff
```

V2 lịch sử:

- Root `new/outputs/pcrau_target_v2_full_seed_24082026/`.
- Checkpoint `checkpoints/best/model.safetensors`, SHA `1505152fca7b725d7db701c1341ad1a57049d6b5ac450a9f5ce0c8d78705ba26`.
- Epoch 13/step 1120, chọn dev total loss 1.6971108556; dev PIT 326/335 = 97.31%.
- Calibrator `evaluation/calibration/calibrator.json` (logistic18), tau `0.09370088241237143`.
- IID `new/test_iid/evaluation/best_v2/`, so sánh RoboRefer `new/test_iid/evaluation/comparison_original_vs_best_v2/`.

`official_v2_replaced=false` trong artifact nghĩa file V2 khóa được bảo tồn, **không phủ nhận quyết định người dùng đặt Adapter là P-CRA-U hiện tại**. Đổi tên hiển thị không tự đổi default runner/demo; phải truyền đúng checkpoint/config/calibrator/profiles nếu có nhiệm vụ chạy sau này.

## 10. Risk, calibration và policy phải hiểu riêng

Biến cố error được học/chấm:

```text
y_error = (truth_answerability != FOUND) OR (MAP nằm ngoài target)
r_ground = logistic(evidence đã chuẩn hóa), ước lượng xác suất event này
```

Trên Test-IID có 595 error: 580 non-FOUND thật + 15 localization failure trong FOUND. Calibrator **đã học composite event**, không phải chỉ học “confidence thấp”; không đổi nhãn FOUND thành non-FOUND chỉ vì MAP sai.

Risk khác:

- xác suất bốn lớp answerability;
- source score theo supervision;
- entropy/phân bố vị trí;
- xác suất grasp/controller thất bại hay an toàn robot vật lý.

Logistic33 = 18 feature lịch sử + 15 evidence bổ sung. Feature gốc gồm entropy/margin/mode, bốn probability answerability mới, năm source score, relation consistency, fusion gate, thumbnail cosine/MAE và disagreement/missing. Feature thêm gồm target/interior confidence, spatial peak/top5/top10 mass, khoảng cách peak, anchor max mean/min, edge min và **bốn probability base V2**. Danh sách chính xác ở calibrator và `selective_experiment.py`.

Không có metric geometry/CameraInfo/TF trong calibrator chọn. Disagreement và missing có hệ số 0 trong đường này; không nói disagreement đã tạo lợi ích.

Sau freeze, so logistic18 (L2 0.001) và logistic33 (L2 0.01), chọn theo calibration 5-fold family CV Brier, rồi fit toàn calibration cho runtime. Standardization fit từng training fold. Family được sắp theo SHA-256 rồi chia round-robin; không dùng 1000 variant như IID độc lập.

Threshold chọn trên risk **OOF**: quét ngưỡng trong tập đủ điều kiện, lấy ngưỡng coverage lớn nhất đạt Wilson upper 95% ≤ budget và tối thiểu 60 accepted family. Sau đó áp ngưỡng này cho calibrator **fit-all** ở runtime. Risk OOF và fit-all có thể khác; đây là nút thắt độ ổn định, không gọi counts resubstitution là audit độc lập.

| Budget hard_found | Tau | Calibration OOF nhận/lỗi | OOF risk | OOF Wilson upper |
|---|---:|---:|---:|---:|
| 5% | 0.1306702154 | 313/8 | 2.56% | 4.96% |
| **7.5% chọn** | **0.2611932835** | **346/16** | **4.62%** | **7.38%** |
| 10% | 0.4474183592 | 376/26 | 6.91% | 9.94% |

Ở budget chọn: OOF nhận đúng 330/411; resubstitution fit-all trên calibration nhận 351, đúng 337, lỗi 14. Các số này **không phải Test-IID**.

Luật runtime hiện tại, từ `experimental_action`:

```text
Nếu predicted FOUND và risk <= 0.2611932834526145: EXECUTE
Nếu không nhận và predicted ABSENT: ABSTAIN
Nếu không nhận và predicted AMBIGUOUS: ASK_USER
Nếu không nhận và predicted INSUFFICIENT_EVIDENCE: REOBSERVE
Nếu predicted FOUND nhưng risk cao:
  semantic score >= max(depth, occlusion): ASK_USER
  ngược lại: REOBSERVE
```

Risk không chia bốn khoảng để tự nhận biết bốn tình huống. `hard_found` vẫn có cổng lớp; tăng tau không cứu ca bị đoán ABSENT/INSUFFICIENT. `risk_only` là comparator, không phải policy người dùng chọn. Đây là action label trên ảnh; chưa qua geometry gate/handoff robot.

Nguồn runtime: `new/scripts/decide_experimental_profile.py`, `new/src/pcrau/selective_experiment.py`. CLI kiểm tra hash calibrator/checkpoint. Có trạng thái experiment lịch sử trong output không tự biến thành bằng chứng robot.

## 11. Kết quả tổng: RoboRefer, V2 và Adapter

Grounding chung V2/Adapter so RoboRefer gốc trên cùng IID:

- PIT trên 835 target-present: **724/835 = 86.71% → 775/835 = 92.81%**, +6.11 pp, CI95% [1.78, 10.49] pp.
- Interior: 86.59% → 91.50%; khoảng cách tới mask trung bình 27.92 → 8.44 pixel. Không phải sai số tới tâm vật.
- FOUND-only: 397/420 → 405/420 (94.52% → 96.43%), delta CI [-0.98, 5.01] pp chứa 0.
- Paired: 107 ca sidecar sửa baseline, 56 ca hồi quy, 4 cả hai sai, 668 cả hai đúng.
- Tăng ròng 51 hit gồm +60 AMBIGUOUS, +8 FOUND, −17 INSUFFICIENT_EVIDENCE. Mask hợp ở AMBIGUOUS giới hạn claim chọn đúng instance.
- Hoa quả cải thiện nhưng nhóm hộp giảm 29.67 pp; không tốt hơn ở mọi subgroup.
- Bootstrap paired gốc 20000 lượt theo 200 family; không lấy số dev/legacy B1 thay baseline IID.

| Chỉ số Test-IID | V2 lịch sử | P-CRA-U Adapter |
|---|---:|---:|
| Answerability accuracy | 76.80% | **81.10%** |
| Answerability macro-F1 | 0.775162 | **0.809686** |
| Recall FOUND | 82.14% | **86.90%** |
| False FOUND / 580 non-FOUND thật | 70/580 = 12.07% | **55/580 = 9.48%** |
| FOUND thật bị đoán non-FOUND / 420 | 75 | **55** |
| PIT / 835 target-present | 775/835 | 775/835 (không đổi) |
| Số EXECUTE / 1000 | 255 | **361** |
| Coverage | 25.50% | **36.10%** |
| Nhận đúng / 405 ca hợp lệ | 249/405 = 61.48% | **327/405 = 80.74%** |
| Lỗi chấp nhận | 6 | **34** |
| Selective risk | 2.35% | **9.42%** |
| Point hit trong tập nhận | 250/255 = 98.04% | 346/361 = 95.84% |
| Composite đúng trong tập nhận | 249/255 = 97.65% | 327/361 = 90.58% |
| Brier risk | 0.096145 | **0.077108** |
| NLL risk | 0.298476 | **0.257066** |
| ECE10 risk | 0.046470 | **0.031080** |
| AURC risk | 0.260451 | **0.255748** |

Hai pipeline có head/calibrator/budget/tau khác nhau; so toàn hệ thống không cô lập lợi ích riêng của Adapter hay đổi ngưỡng.

Confusion matrix: hàng truth, cột predicted, thứ tự FOUND/AMBIGUOUS/ABSENT/INSUFFICIENT_EVIDENCE:

```text
V2 lịch sử                 Adapter
345   0  31  44            365   0  22  33
  0 105   0   0              0 105   0   0
 17   0  85  30             15   0  87  30
 53   0  57 233             40   0  49 254
```

Không đánh tráo false FOUND / non-FOUND với lỗi trong predicted FOUND (55/420 = 13.10%), hay với 34 lỗi trong EXECUTE. Matrix đẹp hay số F1 cao không tự chứng minh policy an toàn.

## 12. Nút thắt “nhát gan” và đánh đổi đã chọn

Người dùng muốn nhận thêm ca hợp lệ, chấp nhận lỗi nhận thức 5–7.5–10% để tăng độ hữu ích; không muốn chỉ ép confidence hay đổi số trong hình. Đã thử học hard cases/train-dev, residual scorers, sửa answerability rồi calibration riêng. Chọn Adapter trước sau khi xem kết quả, giữ tên P-CRA-U.

Trên **405 ca hợp lệ IID**, V2 bỏ 156 = 66 bị sai answerability + 90 predicted FOUND bị risk chặn; Adapter còn bỏ **78 = 48 + 30**:

- 48 sai lớp: 16 đoán ABSENT, 32 đoán INSUFFICIENT_EVIDENCE.
- 30 predicted FOUND nhưng risk cao.
- Adapter sửa 30 ca hợp lệ sai lớp nhưng hồi quy 12, tăng ròng 18.
- Trên toàn bộ classification sửa 71/hồi quy 28, tăng ròng 43.
- Nhận thêm 78 ca đúng, không mất ca đúng V2 đã nhận; tổng thêm 106 quyết định nhận = 78 đúng + 28 lỗi mới.

**Không thay số 75/94/169 trong trao đổi cũ bằng số hiện tại hoặc coi đó cùng subset.** 75 là toàn bộ true FOUND V2 bị sai lớp ở IID, còn subset true FOUND+MAP đúng bị sai lớp là 66. Nếu cần giải thích con số 169 cũ, phải tìm đúng report/profile/dataset của phân tích đó.

34 lỗi nhận hiện tại: **22 INSUFFICIENT_EVIDENCE, 7 ABSENT, 5 FOUND nhưng MAP sai**. Có 19 non-FOUND vẫn hit mask; hit đó không đủ điều kiện chấp nhận, vẫn tính error composite.

Phân tầng variant của lỗi nhận: **16 occlusion, 11 relation, 6 clean, 1 semantic, 0 depth**; occlusion+relation chiếm **27/34**. Chỉ là vị trí nút thắt quan sát được, chưa chứng minh nguyên nhân nhân quả. Nhóm occlusion nhận 16 mẫu và cả 16 lỗi: cần công bố, không giấu dưới metric tổng.

Actions: EXECUTE 361, REOBSERVE 347, ASK_USER 134, ABSTAIN 158. Một số lượng ASK/REOBS cao không tự là “nhát gan không cần thiết”; phải phân tích true valid recall, false accept và chi phí can thiệp. Baseline forced point ở cả mẫu không đủ evidence không phải bằng chứng baseline tự tin đúng.

CI95% bootstrap family (20000 lượt, seed 24082026, threshold cố định):

- Selective risk mới [6.37%, 12.53%], Wilson upper cấp mẫu 12.87%.
- Coverage [33.8%, 38.4%], recall ca hợp lệ [76.67%, 84.89%].
- Paired macro-F1 delta 0.034524, CI [0.015739, 0.053925].

Chưa bảo đảm risk ≤7.5% hoặc ≤10%; chấp nhận đánh đổi là quyết định vận hành của người dùng, không xóa giới hạn thống kê.

Hướng cải thiện được thảo luận: evidence phân biệt “định vị được vùng” với “đủ bằng chứng FOUND”; hard positive và negative trên train/dev; calibration fit/threshold/audit tách family; matched-coverage comparison. **Chưa có lệnh triển khai mới ở session bàn giao này.**

## 13. Hiểu đúng AURC và calibration

AURC được code tính bằng mean selective risk trên mọi prefix khi xếp 1000 mẫu theo risk tăng dần (sort ổn định). Một scalar AURC thuộc một ranking/phân bố/event, **không có “AURC riêng cho mốc 5%, 7.5%, 10%”** nếu ranking giữ nguyên. Các mốc chỉ chọn điểm vận hành.

Với composite error 595/1000, ranking oracle đặt 405 ca đúng trước cũng có AURC **0.229231** theo định nghĩa này. Vì vậy không dùng ngưỡng phổ quát “AURC phải <0.1” để phán xét bộ dữ liệu/event hiện tại.

- V2 logistic18 AURC 0.260451, excess so oracle 0.031220.
- Adapter logistic33 AURC 0.255748, excess 0.026517.
- Raw Adapter `1-p_FOUND` AURC **0.251397**, tốt hơn logistic33 về ranking toàn tập; raw Brier 0.080142/NLL 0.259195/ECE10 0.048469 so logistic33 0.077108/0.257066/0.031080.

Logistic33 cải thiện chất lượng xác suất theo Brier/NLL/ECE trên cùng Adapter, nhưng **không cải thiện mọi metric và không hơn raw về AURC**. Không gộp “calibration tốt hơn” với “xếp hạng tốt hơn” hoặc “robot an toàn”. AURC tổng cũng chưa xác định chất lượng tại vùng nhận 36.1%; phải xem risk–coverage/local ranking/matched coverage.

Macro-F1 bốn lớp và AURC của lỗi composite là hai event/metric khác nhau; F1 khoảng 0.81 không đòi AURC <0.1. Không gọi 0.255748 là loss huấn luyện để giải thích thay định nghĩa.

## 14. Năm biểu diễn uncertainty và các phần không đổi

1. **Phân bố vị trí:** spatial probability trên lưới, mode/entropy/margin/MAP/covariance; phân bố rộng/nhiều đỉnh thể hiện nhiều vị trí cạnh tranh nhưng không đủ chứng minh instance.
2. **Vùng không gian:** highest-density set với mass đã fit; phải báo event và diện tích.
3. **Answerability:** phân bố bốn lớp. FOUND = đủ điều kiện trả lời theo nhãn; AMBIGUOUS = nhiều ứng viên phù hợp; ABSENT = target vắng; INSUFFICIENT_EVIDENCE = quan sát chưa đủ. FOUND predicted chưa bảo đảm điểm đúng.
4. **Nguồn:** semantic/relation/spatial/depth/occlusion score đa nhãn, không loại trừ lẫn nhau và chưa causal attribution.
5. **Risk đã hiệu chuẩn:** ước lượng composite error để chọn lọc; khác source và uncertainty hình học.

Source head đóng băng: macro-F1 5 nguồn 0.67739, micro-F1 0.62140; bỏ spatial weak label macro-F1 0.59792. Depth F1 0.70391/AP 0.71316. Paired delta semantic +0.2202, relation +0.0723, depth +0.4304, occlusion +0.1860. Spatial source = weak label từ AMBIGUOUS, relation edge = proxy. Source semantic/occlusion còn nhiều false positive.

Relation active-edge accuracy gốc 505/650 = 77.69% theo proxy. Có hình anchor/edge prediction thật, **chưa có evaluator anchor localization/semantic graph exact match đầy đủ**, chưa cô lập lợi ích relation head.

Spatial region giữ mass **0.6041831970**, alpha 0.1, nominal 90%; diện tích trung bình **0.5194% lưới 768 ô**:

- Score coverage chính thức: required HD mass ≤ frozen mass, **725/835 = 86.83%**.
- Direct region–mask intersection: vùng rời rạc giao mask, **800/835 = 95.81%**.

Hai event khác nhau: lấy ô vượt mass và thứ tự ties có thể khác. Không thay 86.83 bằng 95.81 trong artifact gốc. Giao ít nhất một phần mask không bảo đảm chứa toàn vật, chọn một instance, graspability hay an toàn 3D. Quét mass trên test chỉ mô tả, không chọn lại mass.

## 15. Thử nghiệm khác và vì sao không thay model chính

- **V3 regularized ba seed 24082026/27/28:** đổi nhiều yếu tố đồng thời, chưa đạt dev grounding gate 97%; không phải ablation cô lập hoặc bằng chứng nhiều seed của V2/Adapter chính.
- **Answerability head-only/evidence/context/detail trước augmentation:** là các bước phát triển; có nhánh diagnostic không qua cổng. Không lấy checkpoint diagnostic làm ứng viên đã được chọn.
- **Acceptance residual/ranker và logistic34:** binary risk-score experiment, giữ answerability V2. Không thay thành Adapter bốn lớp chỉ vì tên chứa residual; các profile 5/7.5/10% có nguồn/cổng riêng.
- **Support/split-calibration mới:** `new/outputs/pcrau_support_split_dev_20261003/`; binary scorer trên V2+Adapter đã freeze, thêm proxy support và tách 100 fit/60 threshold/40 audit family, runtime giữ fit100. IID cũ nhận đúng 295/405, lỗi 18/313 = 5.75%, so Adapter trước 327/405, 34/361 = 9.42%.
- Ở cùng 313 mẫu nhận theo xếp hạng, Adapter trước 298 đúng/15 lỗi, score mới 295 đúng/18 lỗi; giảm risk ở một coverage khác chưa chứng minh ranking vùng quyết định tốt hơn. Evidence42 chưa có lợi ích riêng xác nhận; audit/Wilson chưa xác nhận budget 7.5%. **Người dùng giữ Adapter trước, không chọn support scorer**.

Không merge số support scorer, logistic34, profile fit-only hay V3 vào báo cáo của P-CRA-U hiện tại. Mỗi model/config/calibrator/threshold là một bundle cần truy vết riêng.

## 16. Robot, Gazebo và demo

Hạ tầng/bối cảnh: ROS2 Humble, Gazebo Fortress, UR3, gripper/SusGrip, MoveIt2, RGB-D D435i eye-in-hand; GPU phát triển theo kế hoạch RTX 2000 Ada 16 GiB. Bối cảnh không phải phép đánh giá robot vật lý đã hoàn thành.

Demo mặc định/four-panel/audit hiện dùng **V2 lịch sử**, không tự chuyển sang Adapter do tên hiển thị mới. Fixed overhead/oblique/layout preview khác camera/phân bố đã đánh giá; risk hiển thị ở đó không được coi đã calibrated cho miền demo.

Audit quan trọng:
`new/demo_gazebo/runs/iid_pose_audit_20261001T080100Z_49405/`.

- Năm frame timestamp riêng, MAP (150,210) target/interior hit 5/5, checker RGB-D apple trong cảnh cụ thể pass 5/5.
- Prediction/checker khóa trước mở semantic label hậu kiểm; oracle không đi vào inference/control.
- ASK_USER 5/5, pre-handoff chưa đạt, `moveit_connected=false`, `robot_motion_commanded=false`.
- Checker đỏ/tròn/depth là apple-/scene-specific, chưa object-identity gate tổng quát.
- Năm frame cùng cảnh không phải năm trial robot độc lập; jitter 0 do lưới không phải sai số 3D bằng 0.

Dashboard bốn panel/live lưu frame inference đồng bộ, không đặt điểm frame cũ lên ảnh mới rồi phát target. Một controller RoboRefer cũ chạy grasp không chứng minh Adapter điều khiển robot.

Các episode cube/banana/timeout trong kế hoạch là lịch sử hạ tầng, không phải kết quả P-CRA-U hiện tại. Chưa có thao tác khép kín đo đúng vật/planning/grasp/lift/place, chưa đánh giá hiệu quả chuỗi REOBSERVE/ASK_USER hoặc budget recovery.

## 17. Hình khoa học và cách làm việc

Nguồn: `new/hinhanh/README.md`, `new/hinhanh/ADAPTER_FIGURES.md`. Có 12 bản PNG/PDF tiếng Việt hậu tố `_adapter_vi` trong thư mục 10, 11, 14–23; ảnh V2 cũ giữ nguyên.

- Thay đổi số: reliability/risk–coverage (10/11), risk/decision cases (14 và nội dung liên quan 15), confusion matrix (19), training Adapter (20), risk panel uncertainty–localization (22).
- Không đổi số grounding các hình 15–18, source delta 21 và spatial coverage–area 23 vì nhánh đó đóng băng. Chuyển tiếng Việt không phải cải thiện số liệu.
- Hình 20 dùng train loss và dev accuracy/macro-F1 Adapter; history không có dev loss từng epoch, không bịa đường dev loss.
- Hình 08 source cases đã thay sample thực phù hợp: spatial ví dụ 0.9659 (<1); occlusion có hộp súp che cube cam trên RGB thật. Không sửa probability bằng tay hoặc ghép vật giả làm prediction.
- Hình 24 anchor/edge thật ở `new/hinhanh/24_anchor_edge/`, sample `v211iid_family_000121__clean`. Anchor logits xuất từ V2+cache khóa, đối chiếu MAP/active edge; hình không thay evaluator graph đầy đủ.
- Năm demo uncertainty tiếng Việt ở `new/hinhanh/uncertainty/` thể hiện phân bố/vùng/answerability/nguồn/risk. Khi cần dùng lại phải đọc nguồn/provenance hiện tại, không mặc định demo minh họa là metric Test-IID mới.

Builder `build_figures.py`/`build_method_figures.py` là đường V2 lịch sử; `build_adapter_figures.py` là đường Adapter. Không chạy nhầm builder rồi overwrite ảnh được chọn. Môi trường `.conda-roborefer/bin/python3.10` có thư viện; system Python không mặc định đủ.

**Hiện tạm bỏ qua toàn bộ LaTeX/Prism**, kể cả sửa khung bảng vừa được thảo luận nhưng chưa triển khai ở lượt đó. Không dùng ảnh/PDF biên dịch kiểm tra như bằng chứng experiment. Khi người dùng quay lại LaTeX, đọc nguồn mới nhất và yêu cầu mới, không tự tiếp tục task cũ trong lúc chỉ đang hiểu đồ án.

## 18. Ranh giới khoa học và quy tắc cộng tác

- Giữ riêng grounding đúng, truth/predicted FOUND, lỗi composite, decision label và robot task success.
- Có bằng chứng cải thiện grounding tổng thể so RoboRefer và answerability/valid recall so V2; chưa vượt baseline trên mọi nhóm, mọi metric hoặc mọi điều kiện.
- Một seed chính, nhiều vòng chọn dev; chưa có full retrain ablation/matched operating point để quy nhân quả cho từng component. Ba seed V3 không thay được bằng chứng đó.
- Source weak/proxy, label synthetic và family dependence; domain gap/real depth/TF/3D chưa được kiểm chứng đủ.
- Logistic cải thiện probability không luôn cải thiện ranking. Budget/threshold/observed error là ba con số khác nhau.
- Test đã quan sát: cần nói rõ reevaluation; không dùng test fit weights/calibrator/threshold/prompt. Không tự tạo IID mới khi người dùng đã bảo giữ IID cũ.
- Latency cache-only không tương đương end-to-end từ ảnh mới; chưa có bộ đo đầy đủ để kết luận realtime hay tiết kiệm compute hơn RoboRefer.
- Không bịa/“làm đẹp” confusion matrix, confidence, ảnh, graph, metric hoặc trial. Có thể đổi sample minh họa thật có truy vết; không hạ score bằng tay.
- Trao đổi bằng tiếng Việt, giải thích rõ bằng ví dụ/số và mẫu số. Không đồng nhất “tự tin” với xác suất bị ép cao; mục tiêu là nhận thêm ca đúng trong đánh đổi được công bố.
- Workspace bẩn có thay đổi/xóa file từ trước. Không reset/checkout/delete/commit/push/dọn cache, không sửa `old/`, world/URDF hoặc phần ngoài nhiệm vụ.
- Chỉ tạo script/json/artifact mới khi cần cho việc được giao, không làm thêm thí nghiệm/roadmap để đọc hiểu. Các lệnh trong protocol là tài liệu tham khảo, không phải quyền chạy tiếp ở session sau.

## 19. Thứ tự đọc dành cho session mới

1. Đọc **đầy đủ file bàn giao này**, rồi yêu cầu mới nhất; phần LaTeX đang tạm bỏ qua.
2. Đọc `new/outputs/active_experimental_profile.json`, `new/docs/MODEL_SELECTION.md`; kiểm tra bundle/hash nếu phải dùng model.
3. **Xem trực tiếp** `plan/pipeline.png`, giải thích được blocks/branches/arrows/evidence/frozen/trainable/optional/evaluator-only.
4. Đọc toàn bộ kế hoạch V3 và đối chiếu bản Downloads bằng hash/diff; không dùng plan làm bằng chứng thực hiện. Nếu hai bản y hệt, ghi rõ đã xác nhận cùng nội dung.
5. Đọc `new/README.md`, `new/docs/ARCHITECTURE.md`, `new/docs/DATA_CONTRACT.md`, các protocol DETAIL/LANGUAGE/FULL_CALIBRATION và SUPPORT_SPLIT để hiểu lịch sử lựa chọn; đánh dấu câu cũ.
6. Đọc `model.py`, `text.py`, `dataset.py`, `losses.py`, `answerability_evidence.py`, `language_augmentation.py`, `postprocess.py`, `calibration.py`, `selective_experiment.py` và script Adapter/calibration/decision liên quan khi kiểm chứng cơ chế. Không chạy chúng chỉ để hiểu.
7. Đọc Adapter metadata/freeze lock/profiles/calibrator/IID metrics/SUMMARY; đọc V2 `full_metrics.json` và paired comparison RoboRefer để kiểm chứng kết luận định lượng.
8. Nếu bàn loss: đọc report location-mass smoke. Nếu bàn support mới: đọc report support/IID cũ; không promote vì tên file mới hơn.
9. Nếu bàn hình/robot: đọc README/provenance và đúng audit run; không suy state từ tên folder/hình.
10. Các handoff cũ dùng để hiểu lịch sử, không lấy câu V2-main/placeholder cũ phủ nhận artifact mới. Không cần đọc/sửa sáu chương hoặc form LaTeX lúc này.

Sau khi đọc, chỉ tóm tắt ngắn: bài toán/khoảng trống; pipeline mục tiêu vs thực tế; bundle hiện tại; dữ liệu/event/mẫu số; kết quả và đánh đổi; nút thắt/giới hạn. Nêu rõ file chưa truy cập/đọc hết. **Chờ nhiệm vụ tiếp theo**, không tự train, fit, test, capture, sửa hình hoặc robot.
