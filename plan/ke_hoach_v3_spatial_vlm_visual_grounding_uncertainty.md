# KẾ HOẠCH NGHIÊN CỨU V3 — SPATIAL VISION-LANGUAGE GROUNDING, RELATION REASONING VÀ MULTIMODAL UNCERTAINTY CALIBRATION

> **Trạng thái:** Bản kế hoạch thảo luận V3, chưa thay thế kế hoạch V2 và chưa phải protocol preregistration cuối.
>
> **Ngày lập:** 20/08/2026
>
> **Model nền:** RoboRefer-2B-SFT
>
> **Phương pháp học ưu tiên:** P-CRA-U — Probabilistic Cross-modal Relation-Aware Uncertainty
>
> **Nền tảng kiểm chứng:** ROS 2 Humble, Gazebo Fortress, MoveIt 2, UR3, gripper, camera RGB-D D435i eye-in-hand
>
> **Phần cứng phát triển:** RTX 2000 Ada 16 GiB

Kế hoạch V3 giữ nguyên toàn bộ bằng chứng WP0–WP2 đã khóa. Kế hoạch V2 vẫn được bảo toàn tại [`ke_hoach_moi_roborefer_llm_vlm_agent_ur3.md`](ke_hoach_moi_roborefer_llm_vlm_agent_ur3.md) để truy vết lịch sử thay đổi phạm vi.

`P-CRA-U` tiếp tục là tên làm việc nội bộ. Không tuyên bố tên hoặc cơ chế này mới so với toàn bộ literature trước khi hoàn thành novelty review có hệ thống.

---

## 0. Quyết định định hướng

### 0.1. Trục nghiên cứu chính

Kế hoạch V3 tập trung vào một câu hỏi trung tâm:

> Khi một VLM định vị target từ RGB, depth và câu lệnh quan hệ nhưng chỉ trả một point, làm thế nào để biểu diễn, phân tách và hiệu chuẩn độ bất định không gian nhằm biết point nào đáng tin, lỗi đến từ nguồn nào và khi nào hệ thống phải quan sát lại, hỏi người dùng hoặc từ chối hành động?

Năm keyword được tổ chức theo vai trò, không coi là năm đóng góp độc lập:

| Vai trò | Thành phần |
|---|---|
| Bối cảnh | Spatial vision-language understanding |
| Tác vụ trung tâm | Visual grounding / referring-expression grounding |
| Biểu diễn và suy luận | Query-centric scene graph / relation reasoning |
| Vấn đề phương pháp | Spatial uncertainty quantification |
| Cơ chế ra quyết định | Multimodal uncertainty calibration và selective prediction |

### 0.2. Thay đổi so với kế hoạch V2

- Không lấy “LLM/VLM Agent tổng quát” làm đóng góp trung tâm.
- Không lấy full closed-loop manipulation và recovery làm điều kiện để chứng minh phương pháp perception–uncertainty.
- Không xây full scene graph của toàn bộ thế giới trong phiên bản lõi.
- Không huấn luyện một Spatial VLM mới từ đầu.
- Giữ RoboRefer làm VLM nền và ưu tiên đóng băng checkpoint.
- Dùng query-centric relation graph chỉ chứa target, relation và các anchor liên quan đến câu lệnh.
- Lấy P-CRA-U, spatial distribution, answerability, source uncertainty và calibration làm lõi học máy.
- Dùng Gazebo/UR3 làm downstream validation cho selective execution, không dùng robot success thay thế metric uncertainty.
- Giữ P-CRA-F, LoRA, action-failure head và full scene graph ở trạng thái tùy chọn.

### 0.3. Câu chuyện luận văn dự kiến

```text
RoboRefer point-only grounding
            ↓
Query-centric target–relation–anchor representation
            ↓
Relation-aware spatial distribution + answerability
            ↓
Uncertainty theo semantic/relation/spatial/depth/occlusion
            ↓
Multimodal uncertainty calibration
            ↓
Calibrated grounding risk + spatial confidence region
            ↓
EXECUTE / REOBSERVE / ASK_USER / ABSTAIN
            ↓
Selective validation trên Gazebo/UR3
```

---

## 1. Thuật ngữ và phạm vi

### 1.1. Spatial VLM được dùng theo nghĩa nào

Trong kế hoạch này, “Spatial VLM” mô tả bài toán VLM hiểu vị trí và quan hệ không gian. Không mặc định rằng đồ án tạo ra một foundation model mới.

Cách gọi chính xác ưu tiên trong luận văn:

> **Spatial vision-language grounding** — định vị đối tượng bằng thị giác–ngôn ngữ có xét quan hệ không gian.

### 1.2. Visual grounding

Input:

```text
RGB + relative depth + instruction
```

Baseline output của RoboRefer trong protocol hiện tại:

```text
p_RR = (u,v)
```

Output mục tiêu của phương pháp:

```text
H_target(u,v)       spatial distribution
H_anchor_k(u,v)     anchor distributions
top-k modes         candidate spatial modes
answerability       FOUND / AMBIGUOUS / ABSENT / INSUFFICIENT_EVIDENCE
source scores       semantic / relation / spatial / depth / occlusion
r_ground            calibrated grounding error risk
C_alpha             spatial confidence region, nếu gate mở
```

### 1.3. Multimodal calibration

Trong kế hoạch này, “multimodal calibration” nghĩa là **multimodal uncertainty calibration**: hiệu chuẩn rủi ro dựa trên bằng chứng RGB, depth, language và relation.

Nó không đồng nghĩa với camera calibration. Intrinsics, RGB–depth registration, hand–eye và TF vẫn là yêu cầu kỹ thuật bắt buộc, nhưng không phải đóng góp phương pháp chính.

### 1.4. Query-centric spatial relation graph

Graph lõi chỉ chứa các thực thể được câu lệnh nhắc tới:

```text
target
relation_1 ... relation_L
anchor_1 ... anchor_K
reference_frame
```

Ví dụ:

```text
banana ─left_of_camera→ apple
banana ─nearer_to_camera_than→ mustard_bottle
```

Không bắt buộc phát hiện và gán nhãn mọi vật trong toàn bộ cảnh.

### 1.5. Phạm vi robot

Robot chỉ dùng để kiểm tra downstream utility:

> Calibrated spatial risk có giảm wrong-object execution và ngăn robot hành động khi target/quan hệ không đáng tin hay không?

Core robot policy chỉ gồm:

```text
EXECUTE
REOBSERVE
ASK_USER
ABSTAIN
```

`REPROMPT`, `REPLAN`, `REGRASP` và corrective place là phần mở rộng, không phải minimum viable contribution.

---

## 2. Bằng chứng hiện có và điểm xuất phát

### 2.1. WP0 — Baseline và hạ tầng

WP0 đã hoàn thành:

- build ROS và interface pass;
- 72 test pass tại thời điểm khóa WP0;
- API RoboRefer RGB và RGB-D pass;
- pilot v0 giữ nguyên digest;
- Gate v2 và structured reason codes hoạt động;
- source/config/checkpoint có inventory và SHA-256.

Sau WP1, registry hiện có bảy reason code do bổ sung `NO_VALID_DEPTH`. Báo cáo WP0 ghi sáu code là trạng thái lịch sử tại thời điểm WP0, không phải lỗi artifact.

Tham chiếu: [`../protocol/WP0_STATUS.md`](../protocol/WP0_STATUS.md).

### 2.2. WP1 — Depth sensitivity

WP1 đã chạy 60/60 prediction trên 10 scene × 6 điều kiện:

- RGB-only;
- correct depth;
- flat depth;
- shuffled depth;
- inverted depth;
- localized holes/edge corruption.

Kết quả chính:

- 6/40 corrupted pairs vượt ngưỡng displacement;
- 1/40 instance switch;
- shuffled depth gây ảnh hưởng mạnh nhất;
- target-absent vẫn bị forced point;
- quyết định preregister là `GO_PCRA_F`, nhưng không đồng nghĩa phải huấn luyện P-CRA-F ngay.

Diễn giải dùng cho V3:

> RoboRefer không hoàn toàn bỏ qua depth, nhưng point-only output không cho biết khi nào depth hoặc relation evidence đang không đáng tin. Đây là động lực cho P-CRA-U và source-aware calibration.

Tham chiếu: [`../protocol/WP1_DEPTH_SENSITIVITY.md`](../protocol/WP1_DEPTH_SENSITIVITY.md).

### 2.3. WP2 — Dataset prototype

WP2 đã khóa:

- 50 family;
- 250 sample;
- 30 family depth-dependent;
- 100 raw Gazebo capture;
- split train/dev/calibration/test theo `family_id`;
- 250/250 record qua schema, hash, mask và oracle-isolation;
- replay 2.334 file dẫn xuất không mismatch;
- 85 relation record qua metric-depth/centroid QC;
- 86 software test pass.

Phân bố trạng thái:

| Trạng thái | Sample |
|---|---:|
| `FOUND` | 146 |
| `INSUFFICIENT_EVIDENCE` | 92 |
| `AMBIGUOUS` | 6 |
| `ABSENT` | 6 |

Giới hạn quan trọng:

- dataset mới đủ cho smoke/prototype;
- `AMBIGUOUS` và `ABSENT` quá ít cho calibration mạnh;
- calibration/test prototype chỉ có 30 sample mỗi split;
- graspable/reachable mask mới là proxy;
- relation graph hiện là evaluator-only oracle, chưa phải predicted graph khi inference.

Tham chiếu: [`../protocol/WP2_DATASET_V1.md`](../protocol/WP2_DATASET_V1.md).

### 2.4. Demo robot hiện tại

Hai episode cube/banana đều:

- perception thành công;
- đến được pre-grasp;
- timeout tại `DESCEND_TO_GRASP`;
- end-to-end success 0/2.

Đây là bằng chứng tích hợp chẩn đoán, không phải locked robot evaluation và không được dùng để tuyên bố closed-loop success.

---

## 3. Khoảng trống nghiên cứu

### 3.1. Point không phải uncertainty

Một point không biểu diễn được:

- nhiều target mode;
- vùng target hợp lệ;
- độ gần biên;
- target absent;
- ambiguity giữa các instance;
- anchor bị mất;
- relation conflict;
- nguồn lỗi modality;
- calibrated probability of error.

### 3.2. Relation reasoning đang implicit

RoboRefer có thể xử lý câu lệnh quan hệ ngầm trong feature, nhưng API hiện không xuất:

- target–relation–anchor structure;
- anchor localization;
- edge confidence;
- reference-frame confidence;
- relation-consistency explanation.

Không thể kết luận “RoboRefer đã xây scene graph” chỉ vì nó đôi khi trả đúng point cho câu lệnh quan hệ.

### 3.3. Source uncertainty đang bị trộn

Một failure có thể đến từ:

```text
semantic identity
relation/reference frame
spatial multimodality
depth corruption
occlusion
sensor/TF calibration
geometry
robot action
```

V3 chỉ học năm nguồn perception đầu tiên. Calibration/sensor, geometry và action được log như evidence bên ngoài, không trộn vào semantic label.

### 3.4. Raw score chưa phải probability

Entropy, peak margin, attention, gate support hoặc classifier logit chỉ là raw evidence. Chúng không được gọi là xác suất đúng trước khi fit và đánh giá calibrator trên split độc lập.

### 3.5. Scene graph oracle không deploy được

Relation graph của WP2 có thể làm supervision và evaluator upper bound. Inference phải tự dự đoán target/anchor distributions và relation evidence chỉ từ payload hợp lệ.

---

## 4. Mục tiêu, câu hỏi nghiên cứu và giả thuyết

### 4.1. Mục tiêu tổng quát

Xây dựng và đánh giá một phương pháp relation-aware multimodal uncertainty cho spatial visual grounding, trong đó RoboRefer được giữ làm model nền, P-CRA-U tạo spatial distributions và uncertainty theo nguồn, calibrator biến evidence thành grounding risk, và policy sử dụng risk cho selective execution.

### 4.2. Mục tiêu cụ thể

1. Biểu diễn câu lệnh dưới dạng query-centric target–relation–anchor graph.
2. Dự đoán heatmap target và anchor từ feature RGB–depth–language, không dùng oracle khi inference.
3. Nhận biết `FOUND`, `AMBIGUOUS`, `ABSENT`, `INSUFFICIENT_EVIDENCE`.
4. Dự đoán uncertainty theo semantic, relation, spatial, depth và occlusion.
5. Hiệu chuẩn grounding error risk trên split riêng.
6. Đánh giá risk–coverage, confidence-region coverage và source attribution.
7. Kiểm tra calibrated risk bằng replay và selective Gazebo/UR3 trials.

### 4.3. Câu hỏi nghiên cứu

- **RQ1 — Relation grounding:** Query-centric relation representation có cải thiện target grounding trên câu lệnh multi-anchor/depth-dependent so với RoboRefer point-only và relation heuristics không?
- **RQ2 — Uncertainty:** Spatial distribution, answerability và source scores có dự báo grounding failure tốt hơn geometry-only hoặc point-confidence heuristics không?
- **RQ3 — Multimodal attribution:** Mô hình có tăng đúng source uncertainty khi RGB/depth/relation/occlusion tương ứng bị perturb, thay vì chỉ tăng một uncertainty chung không?
- **RQ4 — Calibration:** Source-conditioned calibration có cải thiện Brier/NLL/risk–coverage so với raw score và scalar calibration chuẩn không?
- **RQ5 — Selective utility:** Calibrated risk có giảm unsafe execution/false accept tại coverage tương đương, và `REOBSERVE` có phục hồi một phần failure không?
- **RQ6 tùy chọn — Fusion:** P-CRA-F có cải thiện grounding thực sự trên nhóm failure do fusion sau khi P-CRA-U và calibration đã ổn định không?

### 4.4. Giả thuyết

- **H1:** Target/anchor heatmaps điều kiện hóa theo relation cải thiện mass-in-target và relation consistency trên task nhiều anchor so với head không relation.
- **H2:** Answerability + spatial multimodality phát hiện `AMBIGUOUS/ABSENT` tốt hơn hard depth gate.
- **H3:** Source-aware training làm score depth tăng có hệ thống dưới depth corruption, score relation tăng dưới relation counterfactual và score occlusion tăng dưới occlusion view.
- **H4:** Calibrator fit độc lập giảm accepted-incorrect risk hoặc tăng coverage tại cùng risk mục tiêu so với raw confidence.
- **H5:** Selective policy dựa trên calibrated risk giảm wrong-object execution so với forced-point always-execute.
- **H6:** Full relation graph hoặc P-CRA-F không cần thiết nếu query-centric P-CRA-U đã đạt gate; negative result được chấp nhận.

---

## 5. Đóng góp dự kiến và giới hạn phát biểu

### 5.1. Đóng góp phương pháp tối đa

Chỉ giữ tối đa hai đóng góp phương pháp chính:

1. **Relation-aware spatial uncertainty sidecar**
   - target/anchor spatial distributions;
   - query-centric relation evidence;
   - answerability;
   - uncertainty theo nguồn.

2. **Multimodal uncertainty calibration cho selective grounding**
   - calibrated grounding risk;
   - risk–coverage;
   - spatial confidence region nếu gate mở;
   - quyết định `EXECUTE/REOBSERVE/ASK_USER/ABSTAIN`.

### 5.2. Thành phần hỗ trợ

- WP2 dataset và counterfactual generator;
- geometry Gate v2;
- dashboard trực quan;
- Gazebo/UR3 selective execution;
- provenance/leakage protocol.

Các thành phần hỗ trợ không tự động trở thành đóng góp thuật toán mới.

### 5.3. Không tuyên bố nếu chưa có bằng chứng

- không gọi hệ thống là một Spatial VLM mới nếu chỉ train sidecar;
- không gọi query graph là full scene graph;
- không gọi raw score là probability;
- không gọi point-in-interior là robot-safe grasp;
- không gọi Gazebo result là physical UR3 result;
- không gọi two-trial demo là end-to-end evaluation;
- không tuyên bố source score xác định nguyên nhân causal trong dữ liệu thật;
- không tuyên bố novelty trước literature review.

---

## 6. Kiến trúc mục tiêu

```text
OBSERVABLE INPUTS
  RGB + relative depth + metric depth + instruction + CameraInfo + TF
            │                                  │
            │                                  └→ language graph parser
            │                                      target/relation/anchors
            ↓
      RoboRefer frozen
      ├→ original point p_RR
      ├→ RGB feature R0
      └→ depth feature D0
            │
            ↓
      P-CRA-U sidecar
      ├→ target heatmap H_t
      ├→ anchor heatmaps H_a1...H_aK
      ├→ relation evidence
      ├→ answerability logits
      └→ source uncertainty logits
            │
            ├→ metric geometry evidence
            ├→ p_RR/P-CRA disagreement
            └→ modality availability/quality
                         │
                         ↓
              independent calibrator
                         │
                         ↓
              calibrated r_ground
              optional confidence region
                         │
                         ↓
       EXECUTE / REOBSERVE / ASK_USER / ABSTAIN
                         │
                         ↓
           offline replay → Gazebo → UR3
```

### 6.1. Rủi ro chính

```text
r_ground = P(target instance hoặc selected location sai | observable evidence)
```

V3 không bắt buộc học `r_action`. Planning/collision/grasp failure được đánh giá riêng ở downstream robot layer.

### 6.2. Ranh giới oracle

Inference được phép đọc:

- RGB/depth thực gửi model;
- instruction;
- CameraInfo/TF nếu geometry stage cần;
- model features/output;
- quality diagnostics tính từ observable sensor data.

Inference không được đọc:

- Gazebo target ID/pose;
- evaluator semantic/instance labels;
- target/anchor mask oracle;
- relation graph oracle;
- expected intervention;
- split/test labels.

---

## 7. Query-centric relation reasoning

### 7.1. Language graph

Parser tạo cấu trúc:

```text
q_target
q_relation_1...q_relation_L
q_anchor_1...q_anchor_K
anchor_valid_mask
reference_frame
```

Prototype giới hạn:

```text
Kmax = 3 anchors
Lmax = 3 relations
```

Hỗ trợ tối thiểu:

- direct target;
- left/right;
- nearer/farther to camera;
- front/behind theo camera frame;
- between hai anchor;
- hai clause quan hệ.

### 7.2. Predicted visual graph

Không phụ thuộc detector toàn cảnh ở phiên bản lõi. Mỗi graph node là một query-conditioned spatial distribution:

```text
node_target = H_target
node_anchor_k = H_anchor_k
```

Thuộc tính node dự đoán hoặc tính từ observable data:

- MAP/top-k mode;
- centroid/covariance;
- relative/metric depth statistic;
- visibility/depth validity;
- entropy/peak margin.

Edge evidence được tính từ:

- relative geometry giữa distributions;
- mask/centroid/depth statistic khi có predicted mask;
- learned relation compatibility;
- reference frame của query.

### 7.3. Ba chế độ graph để đánh giá

| Chế độ | Mục đích |
|---|---|
| Không graph | RoboRefer/P-CRA baseline |
| Predicted query graph | Phương pháp deployable |
| Oracle query graph | Upper bound evaluator-only |

Oracle graph tuyệt đối không được báo như inference performance.

### 7.4. Full scene graph là tùy chọn

Chỉ mở full graph nếu failure audit cho thấy thiếu object candidates toàn cảnh là nút thắt chính. Nếu mở, phải đánh giá riêng node detection, edge F1 và data association; không gộp lỗi graph vào uncertainty head.

---

## 8. P-CRA-U — Relation-aware spatial uncertainty sidecar

### 8.1. Feature contract

Với tile `448×448`, SigLIP patch size 14, hidden size 1152:

```text
R0 = E_rgb(I) ∈ R^(32×32×1152)
D0 = E_depth(Z_rel) ∈ R^(32×32×1152)
```

Dynamic tiling bắt buộc lưu:

- tile index;
- tile bounding box;
- resize/crop/pad transform;
- patch-to-original mapping;
- overlap merge rule;
- RGB/depth alignment assertion.

P-CRA-U đọc feature trước projector để giữ spatial resolution. Đường media token của RoboRefer không bị thay đổi.

### 8.2. Contextual query extractor

Instruction được contextualize trước khi learned target/relation/anchor slots cross-attend. Không dùng lexical embedding rời rạc cho multi-clause relation.

Output:

```text
Q = {q_target, q_relation_l, q_anchor_k}
```

### 8.3. Relation-conditioned sidecar fusion

Thiết kế làm việc:

```text
g_i = sigmoid(W_g [R0_i ; D0_i ; Q_relation])
F_i = LN(R0_i + g_i ⊙ W_d D0_i + A_rel(R0_i,D0_i,Q))
```

`F` chỉ đi vào auxiliary heads. Đây là điểm phân biệt P-CRA-U với P-CRA-F.

### 8.4. Spatial heads

Các head bắt buộc:

- target heatmap;
- anchor heatmap theo slot;
- target interior heatmap hoặc target-valid-location head;
- relation edge logits/evidence;
- answerability logits;
- source uncertainty logits.

Output bắt buộc lưu:

- MAP point;
- top-k connected modes;
- mode mass;
- entropy;
- peak margin;
- mode count;
- local covariance;
- disagreement với `p_RR`.

Không dùng global mean làm selected point khi heatmap đa mode.

### 8.5. Answerability

Bốn lớp:

```text
FOUND
AMBIGUOUS
ABSENT
INSUFFICIENT_EVIDENCE
```

Không chỉ dùng binary answerable nếu dataset đã đủ mỗi lớp. Macro F1 và per-class recall là metric chính do class imbalance.

### 8.6. Source uncertainty

Các source head:

```text
semantic
relation
spatial
depth
occlusion
```

Đây là multi-label task. Counterfactual source làm supervision trong simulator. Với dữ liệu thật, source label có thể weak/partial và phải ghi rõ.

### 8.7. Metric 3D uncertainty

Metric depth không đi qua cùng mục đích với relative depth. Sau khi chọn spatial mode:

```text
(u_k,v_k) ~ selected mode
z_k ~ local metric-depth evidence
P_camera,k = backproject(u_k,v_k,z_k,K)
P_base,k = T_base_camera · P_camera,k
```

Tính `mu_XYZ` và `Sigma_XYZ` bằng sample propagation hoặc Jacobian. Đây là secondary output; không được làm chậm core 2D/answerability gate.

---

## 9. Multimodal uncertainty calibration

### 9.1. Calibration target

Primary event:

```text
y_error = 1 nếu selected target instance hoặc selected point không hợp lệ
```

Các event phụ phải tách riêng:

- answerability error;
- relation error;
- point-outside-target;
- point-outside-interior;
- 3D localization error vượt tolerance.

Không tạo một nhãn “overall failure” trộn perception với controller timeout.

### 9.2. Evidence vector

Calibrator chỉ đọc observable evidence:

```text
heatmap entropy
peak margin
mode count
p_RR/P-CRA disagreement
answerability logits
source logits
relation consistency
RGB/depth availability and quality
geometry support diagnostics
```

### 9.3. Baseline calibrators

- raw max score;
- temperature scaling;
- Platt/logistic calibration;
- isotonic regression nếu đủ dữ liệu;
- geometry-only risk;
- scalar calibrator không source conditioning.

### 9.4. Proposed calibrator

Phương án ưu tiên là logistic/MLP nhẹ có source conditioning, không xây calibrator lớn trước khi dữ liệu đủ:

```text
r_ground = Calibrate(e_spatial, e_relation, e_depth,
                     e_answerability, e_geometry, task_type)
```

Đóng góp “multimodal” chỉ được giữ nếu source-conditioned calibrator tốt hơn scalar baseline trên locked evaluation hoặc cung cấp subgroup calibration rõ ràng.

### 9.5. Spatial confidence region

Nếu validation gate mở, xây set-valued prediction bằng conformal calibration. Vì label là valid target region, phải preregister nonconformity score và coverage event.

Luôn báo đồng thời:

- empirical coverage;
- coverage gap;
- normalized region area;
- coverage–area trade-off;
- subgroup coverage.

Vùng phủ cả ảnh không được coi là kết quả tốt dù coverage cao.

### 9.6. Freeze rule

Sau khi fit calibrator:

- khóa model weights;
- khóa evidence feature set;
- khóa calibrator;
- khóa threshold selective policy;
- khóa code/config hashes;
- sau đó mới mở locked test.

---

## 10. Dataset cho Plan 2

### 10.1. Đơn vị độc lập

Đơn vị split là `scene-query family`. Mọi view, paraphrase, clean/corrupted/counterfactual của cùng family phải ở cùng split.

Không bootstrap từng variant như sample độc lập nếu chúng cùng family.

### 10.2. Dùng dataset WP2 như thế nào

Dataset 50 family/250 sample được dùng cho:

- loader/schema test;
- feature hook smoke;
- deterministic cache test;
- loss implementation;
- overfit-small-set diagnostic;
- preliminary failure audit.

Không dùng nó để tuyên bố calibration/generalization mạnh.

### 10.3. Data-gap audit trước khi scale

Audit tối thiểu:

- tỷ lệ relation loại nào;
- số anchor mỗi query;
- target size/position/depth range;
- mode ambiguity;
- visible fraction;
- source-severity balance;
- object/category/layout reuse;
- artifact leakage của corruption;
- graph-label consistency;
- RoboRefer failure distribution.

### 10.4. Lộ trình scale

Không khóa ngay con số full dataset trước failure audit và power analysis.

| Tầng | Quy mô định hướng | Mục đích |
|---|---:|---|
| Prototype | 50 family / 250 sample, đã có | Smoke pipeline |
| Development | 300–500 family | Model/loss/failure audit |
| Candidate full | 2.000–4.000 family, tùy runtime/QC | Train + calibration + IID/OOD test |

Candidate split định hướng:

```text
Train        55–65%
Dev          10–15%
Calibration  10–15%
Test-IID     10–15%
Test-OOD     held-out assets/layout/view/noise generator
```

Tỷ lệ cuối được khóa trước scale chính thức.

### 10.5. Cân bằng trạng thái

Phải tăng có chủ đích:

- ambiguous cùng class/màu/hình dạng;
- target absent;
- anchor absent;
- relation conflict;
- nhiều anchor;
- occlusion severity;
- depth missing/noise/bias/shuffle;
- OOD asset/layout/camera view.

Không cần cân bằng tuyệt đối, nhưng calibration/test phải có đủ family độc lập cho từng claim.

### 10.6. Minimum negative evidence

Nếu claim false-accept rate dưới khoảng 5%, tối thiểu cần khoảng 60 negative locked families và 0 false accept để upper 95% bound xấp xỉ 5% theo rule-of-three. Đây là minimum tổng thể; subgroup claims cần nhiều hơn.

### 10.7. Dữ liệu thật

Camera thật chỉ mở sau simulator gate:

- dùng family grouping;
- tách calibration thật khỏi test thật;
- không dùng threshold Gazebo để tuyên bố calibrated coverage trên D435i thật;
- báo domain gap và recalibration cost.

---

## 11. Huấn luyện

### 11.1. Nguyên tắc

- đóng băng RoboRefer ở thử nghiệm chính đầu tiên;
- cache `R0/D0` cùng tile mapping;
- train head nhỏ phù hợp GPU 16 GiB;
- không mở Qwen2/full tower fine-tuning;
- không train action-failure head bằng proxy reachability;
- không dùng calibration/test để chọn kiến trúc.

### 11.2. Loss

Location mass:

```text
L_loc = -log(sum_(u,v in M_target) H_target(u,v) + eps)
L_int = -log(sum_(u,v in M_interior) H_target(u,v) + eps)
```

Target/anchor attention:

```text
L_mask = Dice/BCE(H_target, M_target)
       + sum_k valid_k * Dice/BCE(H_anchor_k, M_anchor_k)
```

Relation:

```text
L_rel = CE/BCE(predicted relation edges, relation labels)
```

Answerability và source:

```text
L_ans = CE(answerability logits, state label)
L_src = BCE(source logits, multi-hot sources)
```

Counterfactual ranking:

```text
L_rank = max(0, margin - (u_perturbed - u_clean))
```

Tổng loss:

```text
L = lambda_loc L_loc
  + lambda_int L_int
  + lambda_mask L_mask
  + lambda_rel L_rel
  + lambda_ans L_ans
  + lambda_src L_src
  + lambda_rank L_rank
```

Các lambda được chọn trên dev và freeze trước calibration.

### 11.3. Training stages

**T0 — Feature contract**

- hook `R0/D0`;
- kiểm shape/dtype/tile mapping;
- chạy cùng sample hai lần;
- xác minh hook không đổi `p_RR`;
- đo latency/VRAM/cache size.

**T1 — Overfit/smoke**

- dùng subset train của WP2;
- kiểm loss giảm;
- kiểm target/anchor heatmap không collapse;
- kiểm answerability head chạy đủ bốn lớp;
- không báo generalization.

**T2 — Development training**

- train trên development expansion;
- chọn architecture/hyperparameters bằng dev;
- chạy ablation nhỏ;
- failure audit theo family/source.

**T3 — Candidate model freeze**

- train lại trên train với cấu hình đã chọn;
- khóa weights và feature contract;
- sinh raw prediction cho calibration.

**T4 — Calibration**

- fit calibrator trên calibration split;
- freeze thresholds;
- không sửa model sau khi xem locked test.

**T5 — Locked evaluation**

- Test-IID;
- Test-OOD;
- paired counterfactual;
- resource report;
- robot subset chỉ sau offline gate.

### 11.4. P-CRA-F tùy chọn

Chỉ mở nếu:

- failure audit có tỷ lệ đủ lớn do fusion/depth-relation;
- P-CRA-U đã pass;
- data đủ;
- locked test chưa mở;
- còn thời gian và compute budget.

Nếu không cải thiện paired metric hoặc làm calibration xấu hơn, bỏ khỏi phương pháp cuối và báo negative result.

---

## 12. Baseline và ablation

### 12.1. Baselines

| Mã | Cấu hình | Câu hỏi |
|---|---|---|
| B0 | RoboRefer RGB-only point | Baseline tối thiểu |
| B1 | RoboRefer RGB-D point | Depth gốc giúp gì? |
| B1-CF | B1 với depth counterfactual | Sensitivity theo depth |
| B2 | B1 + geometry Gate v2 | Geometry-only selective baseline |
| G0 | Rule-based target–relation–anchor geometry | Learned relation có cần không? |
| U0 | Heuristic uncertainty từ entropy/disagreement/geometry | Learned UQ có hơn heuristic? |
| U1 | P-CRA-U không relation slots | Relation conditioning đóng góp gì? |
| U2 | P-CRA-U + predicted query graph, chưa calibration | Raw method signal |
| C0 | U2 + scalar calibration | Standard calibration baseline |
| P1 | U2 + source-conditioned calibration | Phương pháp chính |
| OG | U2 + oracle graph | Upper bound evaluator-only |
| P2 | P1 + P-CRA-F | Tùy chọn |

### 12.2. Ablation tối thiểu

- bỏ depth feature;
- bỏ relation slots;
- bỏ anchor heatmaps;
- bỏ answerability;
- bỏ source supervision;
- bỏ counterfactual ranking;
- bỏ geometry evidence khỏi calibrator;
- scalar so với source-conditioned calibration;
- predicted graph so với oracle graph;
- raw score so với calibrated risk.

Không chạy mọi tổ hợp Cartesian. Chọn ablation trực tiếp trả lời RQ/H.

---

## 13. Evaluation metrics

### 13.1. Grounding

- parse rate;
- point-in-target;
- point-in-interior;
- normalized 2D distance;
- target instance accuracy;
- mass-in-target/interior;
- top-k target recall;
- heatmap entropy/mode count;
- relation consistency;
- metric 3D error nếu có.

### 13.2. Relation graph

- target/anchor extraction accuracy;
- relation macro F1;
- reference-frame accuracy;
- node heatmap mass;
- predicted edge precision/recall/F1;
- multi-anchor exact match;
- oracle-graph upper-bound gap.

### 13.3. Answerability

- confusion matrix;
- macro F1;
- balanced accuracy;
- per-class precision/recall;
- false `FOUND` trên `AMBIGUOUS/ABSENT`;
- unnecessary abstention trên `FOUND`.

### 13.4. Uncertainty

- AUROC/AUPRC cho error detection;
- Brier score;
- NLL;
- correlation giữa uncertainty và localization error;
- failure ranking;
- metrics riêng theo source/subgroup.

### 13.5. Source attribution

- multi-label macro/micro F1;
- per-source AUROC/AUPRC;
- top-1/top-2 source accuracy;
- source confusion;
- paired score delta clean → corruption;
- specificity: source không liên quan không tăng vô cớ.

### 13.6. Calibration

- ECE và adaptive ECE;
- Brier/NLL trước–sau calibration;
- reliability diagram;
- risk–coverage curve;
- AURC;
- risk tại coverage cố định;
- coverage tại risk mục tiêu;
- accepted-incorrect risk;
- subgroup calibration gap.

### 13.7. Spatial confidence region

- empirical coverage;
- coverage gap;
- normalized region area;
- coverage–area trade-off;
- IID/OOD/subgroup coverage.

### 13.8. Selective intervention

- decision accuracy;
- unsafe execute rate;
- unnecessary intervention rate;
- reobserve recovery rate;
- added latency/frame/query cost;
- coverage;
- budget exhaustion.

### 13.9. Robot downstream

- correct-target handoff;
- wrong-object motion/execution;
- safe abstention;
- IK/planning success;
- grasp/lift/place success;
- end-to-end success;
- failure stage;
- infrastructure timeout/crash.

Robot metrics không được gộp vào perception calibration metrics.

### 13.10. Resource metrics

- trainable parameters;
- feature-cache size;
- training time;
- inference latency;
- peak VRAM;
- graph/calibrator overhead;
- API throughput.

---

## 14. Statistical protocol

- split và bootstrap theo `family_id`;
- paired comparison trên cùng scene/query/view;
- report micro và macro/per-family;
- confidence interval cho rates;
- effect size cùng p-value khi dùng kiểm định;
- McNemar/paired permutation cho paired binary outcome khi phù hợp;
- bootstrap/permutation cho continuous metric nếu assumptions không đạt;
- correction cho multiple primary comparisons nếu cần;
- không tuyên bố significance từ pilot 10 scene;
- primary metrics và threshold phải preregister trước locked test.

Primary metric set đề xuất:

```text
1. target instance accuracy / mass-in-target
2. answerability macro F1
3. error-detection AUPRC
4. Brier hoặc NLL sau calibration
5. AURC / accepted-incorrect risk
6. unsafe execute rate ở coverage đã khóa
```

Secondary metrics không được thay primary metric sau khi xem test.

---

## 15. Selective policy và robot validation

### 15.1. Core policy

| Điều kiện | Quyết định |
|---|---|
| `FOUND`, risk thấp, geometry pass | `EXECUTE` |
| depth/occlusion/geometry evidence yếu | `REOBSERVE` |
| nhiều target mode hợp lệ | `ASK_USER` |
| target absent hoặc risk vẫn cao sau budget | `ABSTAIN` |

### 15.2. Offline trước robot

Policy phải pass replay trên locked dataset trước khi publish target cho MoveIt.

Minimum offline safety gate:

- accepted-positive correctness đạt threshold preregister;
- negative locked set đủ cho claim;
- false accept nằm trong bound mục tiêu;
- calibration report pass;
- không oracle leakage;
- threshold đã freeze.

### 15.3. Infrastructure gate

Trước robot evaluation:

- sửa timeout `DESCEND_TO_GRASP`;
- tối thiểu năm full baseline infrastructure run liên tiếp không crash/timeout;
- controller/MoveIt/gripper verification pass;
- negative controls không ra lệnh motion;
- emergency-stop/safety boundary giữ nguyên.

### 15.4. Robot comparison

So sánh tối thiểu:

```text
forced-point always-execute
geometry hard gate
calibrated selective policy
```

Dùng paired scene seeds. Tách model failure và infrastructure failure.

---

## 16. Demo trực quan

Dashboard 2×3 đề xuất:

```text
┌────────────────────┬────────────────────┬────────────────────┐
│ RGB + Depth        │ Language graph     │ Predicted graph    │
│ clean/corruption   │ target/relation    │ node/edge evidence │
├────────────────────┼────────────────────┼────────────────────┤
│ Spatial heatmap    │ Source uncertainty │ Calibrated action  │
│ point/top-k/region │ score breakdown    │ risk + decision    │
└────────────────────┴────────────────────┴────────────────────┘
```

Demo scenarios:

1. clear target → một mode → `EXECUTE`;
2. two equivalent targets → multi-mode → `ASK_USER`;
3. absent target → `ABSTAIN`;
4. shuffled/missing depth → depth uncertainty tăng → `REOBSERVE`;
5. occluded anchor → relation/occlusion uncertainty tăng;
6. new view phục hồi risk → selective execution;
7. robot finale chỉ chạy khi infrastructure gate pass.

GUI evaluator phải đánh dấu rõ oracle overlays. Predicted graph/heatmap không được trộn với ground-truth mask.

---

## 17. Work packages cập nhật

### WP0 — Baseline lock — HOÀN THÀNH

- RoboRefer/API/build/test;
- pilot immutable;
- Gate v2/reason codes;
- provenance.

### WP1 — Depth sensitivity — HOÀN THÀNH

- 60 paired inference;
- depth counterfactual;
- `GO_PCRA_F` sơ bộ;
- P-CRA-F vẫn deferred.

### WP2 — Dataset prototype — HOÀN THÀNH

- 50 family/250 sample;
- relation/depth/counterfactual schema;
- replay/leakage/QC;
- report assets.

### WP3 — Scope formalization, graph contract và data audit

**Công việc**

- novelty/literature review;
- khóa task definition và primary metrics;
- formalize language/predicted/oracle query graph;
- audit WP2 class/source/relation balance;
- feature hook `R0/D0` và tile mapping;
- smoke cache deterministic;
- preregister development data expansion.

**Gate**

- graph/tensor contract pass;
- hook không đổi baseline output;
- no-oracle inference test pass;
- data expansion plan có target power/QC.

### WP4 — P-CRA-U development

**Công việc**

- scale development dataset;
- target/anchor heatmaps;
- relation evidence;
- answerability;
- source uncertainty;
- baseline/ablation trên dev.

**Gate**

- heatmap tốt hơn uniform/center/trivial baseline;
- answerability tốt hơn majority/geometry baseline;
- source score có paired counterfactual signal;
- latency/VRAM trong budget;
- không collapse/copy point RoboRefer mù quáng.

### WP5 — Calibration và locked offline evaluation

**Công việc**

- freeze candidate model;
- fit scalar và source-conditioned calibrators;
- risk–coverage/reliability;
- conformal region nếu gate mở;
- Test-IID/OOD locked.

**Gate**

- calibration tốt hơn raw score;
- phương pháp đầy đủ tốt hơn scalar baseline hoặc giới hạn claim;
- threshold freeze;
- đủ negative evidence;
- report confidence intervals.

### WP6 — Intervention replay và dashboard

**Công việc**

- `EXECUTE/REOBSERVE/ASK_USER/ABSTAIN`;
- retry budget;
- counterfactual/reobserve replay;
- dashboard khoa học;
- demo offline deterministic.

**Gate**

- unsafe execute giảm tại coverage được báo rõ;
- intervention không dựa oracle;
- reobserve recovery có paired evidence;
- demo replay không phụ thuộc robot.

### WP7 — Gazebo/UR3 downstream validation

**Công việc**

- sửa infrastructure timeout;
- baseline stability runs;
- paired selective trials;
- negative controls;
- optional D435i/UR3 thật sau safety review.

**Gate**

- five-run infrastructure stability;
- failure-stage logging;
- no-motion negative controls;
- không gộp infrastructure/model failure.

### WP8 — Optional P-CRA-F, freeze và luận văn

- P-CRA-F chỉ nếu gate mở;
- final ablation;
- method freeze;
- thesis figures/tables/demo video;
- artifact manifest;
- limitations và negative results.

---

## 18. Lịch triển khai định hướng 16 tuần

| Tuần | Công việc | Mốc |
|---:|---|---|
| 1 | Novelty review, khóa scope/RQ/primary metrics | Scope freeze |
| 2 | Query graph contract và WP2 data-gap audit | Graph/data audit |
| 3 | `R0/D0` hooks, tile mapping, deterministic cache | Feature gate |
| 4 | Scale development families và QC | Development data |
| 5 | Target/anchor heatmap baseline | Spatial signal |
| 6 | Relation slots/edge evidence | Relation ablation |
| 7 | Answerability head | Answerability gate |
| 8 | Source uncertainty và counterfactual ranking | Source gate |
| 9 | Failure audit, model selection, freeze candidate | Model freeze |
| 10 | Scale calibration/Test-IID/OOD và khóa manifest | Data lock |
| 11 | Fit scalar/source-conditioned calibration | Calibration report |
| 12 | Risk–coverage/conformal nếu đủ điều kiện | Offline method gate |
| 13 | Selective policy và reobserve replay | Intervention gate |
| 14 | Dashboard/demo offline | Reproducible demo |
| 15 | Gazebo/UR3 paired subset nếu infrastructure pass | Downstream evidence |
| 16 | Final ablation, viết luận văn, đóng artifact | Thesis package |

Nếu chậm tiến độ, thứ tự cắt scope:

1. bỏ P-CRA-F;
2. bỏ full scene graph;
3. bỏ learned action head;
4. giới hạn robot thật thành proof-of-concept;
5. giữ P-CRA-U + calibration + locked offline evaluation.

---

## 19. Go/no-go và stop rules

### 19.1. Graph go/no-go

Giữ relation graph trong method nếu predicted graph/slots:

- cải thiện relation-dependent grounding hoặc uncertainty;
- không chỉ cải thiện oracle-graph upper bound;
- overhead chấp nhận được;
- không cần semantic oracle.

Nếu không đạt, graph được giữ như evaluator/analysis representation, không gọi là đóng góp phương pháp.

### 19.2. P-CRA-U go/no-go

- deterministic/aligned feature cache;
- heatmap mass có tín hiệu trên dev;
- answerability/error detection tốt hơn baseline;
- source head phản ứng đúng counterfactual ở mức pre-register;
- latency/VRAM pass.

Nếu không đạt, dừng scale calibration và audit feature/data/label trước.

### 19.3. Calibration go/no-go

- Brier/NLL hoặc AURC cải thiện so với raw score;
- reliability không xấu hơn trên subgroup chính;
- source-conditioned calibrator có ích hơn scalar baseline hoặc bị loại;
- calibration set đủ lớn cho claim.

Nếu calibration không ổn định, tăng calibration families hoặc giới hạn claim; không tune trên test.

### 19.4. Robot go/no-go

- offline selective gate pass;
- five consecutive infrastructure runs;
- no-motion negatives;
- safety boundary pass.

Nếu không đạt, robot chỉ chạy shadow/offline replay và limitation được ghi rõ.

---

## 20. Điểm mạnh của Plan 2

### 20.1. Câu hỏi nghiên cứu rõ

Plan tập trung vào một chuỗi logic:

```text
grounding → relation → uncertainty → calibration → selective decision
```

Metric và ablation có thể gắn trực tiếp với từng RQ.

### 20.2. Phù hợp với artifact đã làm

- WP1 cung cấp depth counterfactual;
- WP2 có target/anchor/relation/source labels;
- split/leakage/provenance đã có;
- dashboard và UR3 có thể tái sử dụng làm downstream demo.

Không phải bỏ WP0–WP2 để đổi hướng.

### 20.3. Dễ đánh giá định lượng và quy nguyên nhân

Offline evaluation tách được:

- grounding error;
- relation error;
- uncertainty quality;
- calibration quality;
- intervention utility.

Controller timeout không làm mất giá trị của perception/calibration result.

### 20.4. Khả thi với GPU 16 GiB

RoboRefer frozen + feature cache + sidecar nhỏ giảm VRAM và compute so với full fine-tuning hoặc P-CRA-F sớm.

### 20.5. Negative result vẫn có giá trị

- graph không cải thiện;
- source-conditioned calibration không hơn scalar;
- depth contribution thấp;
- P-CRA-F không đáng chi phí.

Nếu protocol sạch, các kết quả này vẫn trả lời RQ và giúp giới hạn claim.

### 20.6. Demo khoa học trực quan

Heatmap, graph, source uncertainty, reliability và decision có thể hiển thị trực tiếp. Robot là đoạn kết chứ không phải điểm thất bại duy nhất của demo.

---

## 21. Điểm yếu và rủi ro của Plan 2

### 21.1. Novelty chưa được xác nhận

Relation-aware grounding, uncertainty estimation, selective prediction và multimodal calibration đều đã có các literature liên quan. Việc ghép keyword không tự tạo novelty.

**Giảm thiểu:** hoàn thành novelty matrix trước WP4; xác định baseline gần nhất; phát biểu đóng góp ở cơ chế và protocol cụ thể.

### 21.2. Scene graph inference chưa tồn tại

WP2 relation graph là oracle. Predicted graph có thể thất bại do target/anchor localization trước khi relation reasoning bắt đầu.

**Giảm thiểu:** dùng query-conditioned heatmap nodes, so predicted/oracle/no-graph, không xây full graph sớm.

### 21.3. Dataset nhỏ và mất cân bằng

6 ambiguous và 6 absent không đủ cho calibration/false-accept claim. Variants cùng family có tương quan.

**Giảm thiểu:** scale theo family, tăng negative states, calibration/Test-IID/OOD riêng, bootstrap theo family.

### 21.4. Synthetic-to-real gap

Gazebo depth/segmentation sạch hơn RealSense; source labels trong simulator có thể không phản ánh lỗi thật.

**Giảm thiểu:** domain randomization, real calibration split, limited claims, báo recalibration cost.

### 21.5. Source uncertainty có thể không identifiable

Một observation có thể đồng thời bị semantic ambiguity, occlusion và depth failure. Source label từ perturbation không đảm bảo causal attribution trong thực tế.

**Giảm thiểu:** multi-label targets, paired controlled counterfactual, không tuyên bố causal diagnosis ngoài domain kiểm soát.

### 21.6. Calibration contribution có thể yếu

Nếu chỉ áp temperature scaling lên một score, phần “multimodal calibration” có thể bị xem là áp dụng kỹ thuật chuẩn.

**Giảm thiểu:** đánh giá source-conditioned evidence, modality disagreement, subgroup calibration và downstream selective utility; nếu không hơn scalar baseline, hạ calibration thành thành phần hỗ trợ.

### 21.7. RoboRefer point-only hạn chế supervision

Baseline không có native heatmap hoặc confidence, nên comparison với distribution head không hoàn toàn đối xứng.

**Giảm thiểu:** dùng point/geometry heuristics, uniform/center baselines, mass metrics và error detection; không gán confidence giả cho RoboRefer.

### 21.8. Metric 3D uncertainty phức tạp

Spatial, depth, intrinsics, TF và time synchronization không độc lập. Covariance đơn giản có thể tạo false precision.

**Giảm thiểu:** giữ 3D uncertainty là secondary, dùng controlled simulator validation và nêu assumptions.

### 21.9. Nguy cơ scope creep quay trở lại

Full graph, GNN, detector, P-CRA-F, LoRA, LLM Agent và robot recovery có thể làm Plan 2 rộng như V2.

**Giảm thiểu:** tối đa hai đóng góp phương pháp; áp dụng stop rules và thứ tự cắt scope ở mục 18.

### 21.10. Robot vẫn cần cho downstream claim

Nếu tuyên bố hệ thống giúp manipulation an toàn hơn, ít nhất cần selective robot evidence. Hạ tầng hiện còn timeout.

**Giảm thiểu:** sửa infrastructure thành workstream riêng; offline claim không phụ thuộc robot; chỉ phát biểu downstream utility đúng mức bằng chứng.

---

## 22. Đánh giá tổng thể Plan 2

| Tiêu chí | Đánh giá | Điều kiện để giữ điểm |
|---|---:|---|
| Câu hỏi nghiên cứu | 8.5/10 | Khóa RQ và không thêm full Agent |
| Tính nhất quán | 8.5/10 | Graph/UQ/calibration cùng phục vụ grounding |
| Phù hợp WP0–WP2 | 9/10 | Bảo toàn provenance và mở rộng đúng data gap |
| Khả thi GPU/thời gian | 8/10 | Frozen RoboRefer, sidecar nhỏ, P-CRA-F tùy chọn |
| Đánh giá định lượng | 9/10 tiềm năng | Cần scale calibration/test và family-level statistics |
| Demo khoa học | 9/10 tiềm năng | Cần predicted graph/heatmap/calibrator thật |
| Demo robot | 7/10 tiềm năng | Phụ thuộc sửa timeout và infrastructure gate |
| Novelty | Chưa chấm chắc chắn | Bắt buộc literature review |
| Bằng chứng phương pháp hiện tại | Thấp | WP3/P-CRA-U chưa được train |

Kết luận công tâm:

> Plan 2 phù hợp làm lõi luận văn hơn kế hoạch end-to-end rộng, nhưng chỉ mạnh nếu giữ graph ở mức query-centric, scale dữ liệu đủ cho calibration và chứng minh source-conditioned uncertainty bằng ablation. Nếu cố xây full scene graph, Spatial VLM mới, P-CRA-F, LLM Agent và robot recovery đồng thời, lợi thế tập trung của Plan 2 sẽ mất.

---

## 23. Tên đề tài dự kiến

### Tên tiếng Việt ưu tiên

> **Nghiên cứu phương pháp định lượng và hiệu chuẩn độ bất định đa phương thức có nhận biết quan hệ cho Visual Grounding không gian trong thao tác robot**

### Tên tiếng Việt ngắn hơn

> **Nghiên cứu hiệu chuẩn độ bất định không gian đa phương thức cho định vị thị giác–ngôn ngữ trong thao tác robot**

### Tên tiếng Anh

> **Relation-Aware Multimodal Uncertainty Quantification and Calibration for Spatial Vision-Language Grounding in Robotic Manipulation**

Chưa đổi tên đăng ký ngay. Khóa tên cuối sau khi WP4 chứng minh P-CRA-U có signal trên dev và novelty review xác định được đóng góp hợp lệ.

---

## 24. Artifact dự kiến

```text
plan/
  ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md

protocol/
  WP3_GRAPH_FEATURE_CONTRACT.md
  query_graph_v1.schema.json
  feature_cache_v1.schema.json
  run_wp3_checks.sh
  WP4_PCRA_U.md
  WP5_MULTIMODAL_CALIBRATION.md
  calibration_protocol_v1.json

feature_cache/
  roborefer_pcra_u_<timestamp>/

checkpoints/
  pcra_u_<timestamp>/
  calibrator_<timestamp>/

results/
  pcra_u_<timestamp>/
    prediction_lock.json
    evaluation/
    report_assets/
    dashboard/
```

Mỗi stage phải lưu:

- code/config/model hashes;
- data manifest;
- train/dev/calibration/test provenance;
- raw predictions;
- calibration parameters;
- primary/secondary metrics;
- failed infrastructure runs;
- figures/tables/GUI assets.

---

## 25. Câu chốt

> Đồ án không cố biến RoboRefer thành một Agent biết làm mọi thứ. Đồ án tập trung biến một point spatial grounding thiếu thông tin tin cậy thành một dự đoán có cấu trúc target–relation–anchor, có spatial distribution, có uncertainty theo nguồn và có calibrated risk đủ rõ để hệ thống biết khi nào nên hành động, quan sát lại, hỏi người dùng hoặc từ chối.
