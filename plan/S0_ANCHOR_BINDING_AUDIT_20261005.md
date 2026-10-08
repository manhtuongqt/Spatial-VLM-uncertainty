# S0 — Audit câu lệnh và anchor predictions trên train/dev

Ngày: **05/10/2026**. Trạng thái: **đã thực hiện frozen inference và hậu kiểm S0**. Nguồn kế hoạch: [GCA/FUSE upgrade plan](GCA_FUSE_UPGRADE_PLAN_20261005.md).

## 1. Kết luận và quyết định sau S0

**Annotation đủ để kiểm tra target–relation–anchor trong phạm vi trái/phải; nhưng predicted anchor hiện tại chưa đủ đáng tin để dùng làm ràng buộc cứng sửa target MAP.**

- Đối chiếu prompt → noun phrases → registry/graph/mask nhất quán **400/400 mẫu** trong subset.
- Khi anchor mask không rỗng, anchor MAP hit **209/251 = 83,27% train**, **44/61 = 72,13% dev**.
- Trong **16 ca truth FOUND dev**, target đúng ở **14/16**, anchor đúng **12/16**, cả hai cùng đúng **11/16 = 68,75%**.
- Có **8 mẫu / 6 family** anchor mask rỗng nhưng mô hình vẫn xuất peak. Một mẫu train có sigmoid-max **0,995801** dù anchor mask rỗng.
- Trên dev, **11/17** anchor misses có sigmoid-max ≥0,9; điểm lớn chưa xác nhận đúng danh từ.

**Quyết định:** ưu tiên làm rõ/bổ sung phrase-conditioned anchor binding và evidence về anchor presence; chưa promote geometric MAP refinement. Có thể nghiên cứu verifier **evidence-only** sau này, giữ MAP baseline và lưu trạng thái chưa xác minh. Audit này chưa huấn luyện thành phần mới hoặc áp verifier.

Đây là quyết định thận trọng từ development evidence, không phải chứng minh rằng phrase conditioning chắc chắn sẽ sửa hết lỗi. Chưa có gate định lượng khóa trước audit để kết luận thống kê pass/fail một ngưỡng chung. Dev chỉ có 16 family; cần công bố sự hạn chế này.

## 2. Phạm vi dữ liệu và bundle

Đọc inventory prompt của manifest development **2000 mẫu / 400 family**. Frozen inference và hậu kiểm mask/graph chỉ trên **80 family có câu trái/phải**, lấy đủ năm variant để tránh chọn riêng case đẹp.

| Phạm vi | Train | Dev | Tổng |
|---|---:|---:|---:|
| Family thuộc subset | 64 | 16 | 80 |
| Mẫu của các family này | 320 | 80 | 400 |
| Câu trái/phải, active anchor | 256 | 64 | 320 |
| Câu direct, semantic counterfactual cùng family | 64 | 16 | 80 |
| Active anchor có mask không rỗng | 251 | 61 | 312 |
| Active anchor có mask rỗng | 5 | 3 | 8 |

320 câu trái/phải gồm **150 left_of, 170 right_of**. Các câu direct không thuộc mẫu số anchor accuracy; slot không hoạt động của chúng không có danh tính anchor để chấm. Không đọc Test-IID/OOD hoặc calibration samples. Calibrator/profiles được đọc để kiểm tra hash bundle, không fit hoặc áp policy mới.

Baseline chính xác: **P-CRA-U = V2 đóng băng + detail_cost Adapter + logistic33**. Trong S0 chỉ chạy forward checkpoint deterministic; không dùng risk để chọn case/threshold.

- Config: `new/outputs/pcrau_answerability_language_dev_20261003/config.json`.
- Checkpoint: `detail_cost/checkpoints/best/model.safetensors` trong root trên.
- Checkpoint SHA-256: `5a326a8040bfbabfadf3f57fff51fd90be3845438c9a3a8c6ee41bdd1e0fa045`.
- Config SHA-256: `b214670c6c4a679f755a9953e15bae4bbb71ee5eea1980fa8e3d64030d79f85b`.
- Eval mode, requires_grad=False, torch.inference_mode; AMP bfloat16 theo config; batch 8; seed 24082026; không MC/modality dropout.
- Môi trường: Python trong `.conda-roborefer`, PyTorch 2.13.0+cu130, RTX 2000 Ada Generation.

Artifact cuối: [pcrau_s0_anchor_audit_20261005_r2](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/summary.json). Bản không hậu tố `_r2` là diagnostic trước sửa logic đối chiếu graph, không dùng làm báo cáo cuối; xem mục 3.

## 3. Kiểm tra cấu trúc câu và annotation

Parser audit **chỉ đọc prompt** theo grammar hẹp: `locate the TARGET that is left/right of the ANCHOR.` hoặc direct `locate the TARGET.`. Không lấy object IDs/graph làm đầu vào parser, không thay parser/model production. Câu ngoài grammar trả unsupported, không fallback im lặng thành direct.

Trong subset:

- 400/400 câu parse được trong phạm vi đã khai báo.
- Target phrase khớp `candidate_target_ids` qua semantic class registry; nhóm `fruit` theo nhóm apple/orange/lemon/mango của generator.
- Anchor phrase khớp đúng `anchor_ids`, mask descriptor và thứ tự slot 0.
- 320/320 câu trái/phải có một anchor và frame `image`; direct có zero anchor.
- Manifest anchor mask path/hash khớp record mask path/hash; record hash và mask hash được kiểm tra.
- Graph target_ids là anchor; **graph source_ids là valid_target_ids**, không phải mọi candidate target.

**Chi tiết quan trọng:** ở 84 ca ABSENT, graph source_ids rỗng trong khi candidate_target_ids vẫn nêu vật được hỏi. Điều đó phản ánh không có target hợp lệ, không phải parser sai. Lần audit đầu kiểm nhầm source_ids với candidate_target_ids, làm báo 84 mismatch giả. Đã sửa phép chấm theo schema và tạo run `_r2`; không sửa annotation/dataset. Các anchor/target predictions và metrics vị trí không đổi giữa hai run.

“400/400 nhất quán” là kiểm tra cơ học trong grammar/ontology development hiện có, không phải semantic graph exact match tổng quát hoặc đánh giá câu tự do ngoài dataset.

## 4. Định nghĩa metric và quy ước tọa độ

Anchor slot 0 có logits 24×32; spatial probability dùng softmax trên 768 ô. MAP chuyển về ảnh 640×480 theo tâm ô:

```text
x = int((grid_x + 0.5) * 640 / 32)
y = int((grid_y + 0.5) * 480 / 24)
```

**Primary binding metric:** MAP rơi trong full-resolution mask của đúng anchor. Chỉ chấm trên active anchor có mask không rỗng. Mask rỗng được báo riêng, không cộng thành misses của mẫu số visible-anchor.

**Secondary metric:** ô MAP giao mask sau INTER_AREA downsample. Trong subset này, secondary hits bằng full-pixel hits; không có ca miss được cứu bởi cell overlap. Hai event vẫn khác nhau.

Lưu thêm sigmoid-max của logits, normalized entropy, fractional-mask-weighted probability mass và khoảng cách peak tới mask. Các score này chưa được calibration cho anchor existence/binding. Weighted mass có phụ thuộc occupancy của ô, không phải một probability correctness đã kiểm chứng.

Dataset geometry dùng **centroid mask**, hệ ảnh x tăng sang phải, margin **12 pixel**, strict signed_margin >0. So hai MAP với margin đó trong audit chỉ là diagnostic; MAP là điểm trong vật, chưa chắc là centroid. Không biến diagnostic này thành relation truth hoặc verifier certificate.

## 5. Kết quả binding

| Metric | Train | Dev |
|---|---:|---:|
| Visible active anchor | 251 | 61 |
| Anchor MAP đúng | 209/251 = 83,27% | 44/61 = 72,13% |
| Anchor MAP sai | 42/251 | 17/61 |
| Cell overlap | 209/251 | 44/61 |
| Truth FOUND trong câu trái/phải | 57 | 16 |
| Target MAP đúng trong FOUND | 52/57 = 91,23% | 14/16 = 87,50% |
| Anchor MAP đúng trong FOUND | 47/57 = 82,46% | 12/16 = 75,00% |
| Cả target và anchor đúng trong FOUND | 44/57 = 77,19% | 11/16 = 68,75% |

CI95% percentile bootstrap theo family, 5000 resamples, seed 24082026, cho visible-anchor hit rate:

- Train: **[77,38%; 88,49%]**, 64 family.
- Dev: **[56,25%; 86,21%]**, 16 family.

CI phản ánh phụ thuộc giữa variants; đây vẫn là development diagnostic, không phải independent generalization test. Không gộp train accuracy thành bằng chứng ngoài tập học.

### 5.1. Phân tầng variant — hits / visible anchors

| Variant | Train | Dev |
|---|---:|---:|
| clean | 59/64 | 11/15 |
| depth_corruption | 59/64 | 11/15 |
| occlusion_view_counterfactual | 51/64 | 11/15 |
| relation_counterfactual | 40/59 | 11/16 |

Train relation-counterfactual kém hơn clean đáng kể về point estimate. Không kết luận nguyên nhân chỉ từ bảng: query đổi target/anchor/relation, một số family còn đổi cached observations. Dev nhỏ và các variant cùng family tương quan.

### 5.2. Đổi vai target/anchor

So clean với relation_counterfactual trong cùng family:

- Train: 64 cặp; chỉ 56 cặp có cùng cached features. Trong **51 cặp cùng features và cả hai anchor visible**, cả hai query cùng hit anchor ở **29/51**; **5/51** cặp giữ nguyên anchor MAP dù anchor phrase đổi.
- Dev: 16 cặp cùng cached features; **15** cặp cả hai anchor visible. Cả hai query cùng hit anchor ở **6/15**; **2/15** cặp giữ nguyên anchor MAP.

Các số là diagnostic về khả năng đổi vai, không cô lập causal effect của một từ: cả target noun, predicate và anchor noun đều có thể đổi. Không lấy 64 train pairs như thể tất cả cùng observation.

## 6. Confidence không xác nhận identity/presence

Trong visible-anchor misses, anchor peak rơi vào target mask ở **14/42 train**, **2/17 dev**. Đây là một dạng nhầm vai có thể thấy trực tiếp; các misses còn lại không mặc định cùng nguyên nhân.

**33/42 train misses và 11/17 dev misses có sigmoid-max ≥0,9.** Mốc 0,9 chỉ là thống kê mô tả, không được chọn thành gate runtime.

8 empty-anchor samples thuộc **6 family**: 5 train samples thuộc 5 family; 3 dev samples thuộc cùng family `v211dev_family_000392`.

| Mẫu | Split / truth | Sigmoid-max | Softmax peak | Entropy chuẩn hóa |
|---|---|---:|---:|---:|
| `v211dev_family_000023__relation_counterfactual` | train / INSUFFICIENT | 0,995801 | 0,720037 | 0,185839 |
| `v211dev_family_000392__clean` | dev / ABSENT | 0,725649 | 0,389528 | 0,387807 |
| `v211dev_family_000392__depth_corruption` | dev / ABSENT | 0,479991 | 0,207809 | 0,469564 |
| `v211dev_family_000392__occlusion_view_counterfactual` | dev / ABSENT | 0,848972 | 0,318704 | 0,347383 |

Anchor mask rỗng nghĩa không có anchor pixels theo evaluator observation, không tự chứng minh đối tượng không tồn tại trong toàn cảnh 3D. Softmax vẫn phải chọn một peak; logits sigmoid cũng có thể lớn. Audit chưa cho thấy một confidence/entropy threshold đủ phân biệt existence hoặc identity. Không fit threshold từ 8 mẫu này.

## 7. Case thật và đối chiếu hình

Ảnh được dựng từ RGB model input, logits thật và mask evaluator hậu kiểm. Cyan = anchor mask, green = target mask, dấu đỏ = predicted anchor MAP, dấu vàng = target MAP. Không dùng mask để sửa prediction. Case được chọn theo thứ tự sample ID trong category, ưu tiên dev; chỉ minh họa failure/success, không thay thống kê toàn subset.

| Case | Câu được hỏi và kết quả | Hình |
|---|---|---|
| `v211dev_family_000001__clean` | apple right_of purple cube; cả target/anchor MAP đúng | [Binding đúng](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/cases/found_binding_hit.png) |
| `v211dev_family_000030__relation_counterfactual` | green cube right_of lemon; target đúng nhưng anchor peak (290,150) rơi lên green cube, ngoài lemon mask khoảng **146,86 px** | [FOUND nhưng sai anchor](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/cases/found_binding_miss.png) |
| `v211dev_family_000392__clean` | orange cube left_of orange; anchor mask rỗng nhưng anchor/target peak đều (130,170); truth ABSENT | [Anchor rỗng vẫn có peak](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/cases/missing_anchor_peak.png) |
| `v211dev_family_000097__relation_counterfactual` | train, truth INSUFFICIENT; anchor miss khoảng **174,13 px**, nhưng phép so hai predicted peaks vẫn pass | [Peak relation pass chưa đủ](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/cases/insufficient_anchor_miss.png) |

Đã xem trực tiếp bốn hình để kiểm tra overlay/nhãn/heatmap. Case thứ hai cho thấy verifier sai anchor có thể chặn target vốn đang đúng. Case thứ tư cho thấy predicate pass trên điểm dự đoán không đủ chứng minh binding hoặc answerability.

## 8. Contract và bước tiếp theo

Contract v0 được ghi riêng tại [S0_VERIFIER_INPUT_CONTRACT_20261005.md](S0_VERIFIER_INPUT_CONTRACT_20261005.md).

Đã chốt cho bước thiết kế tiếp:

- Prompt-only typed query; direct bypass; unsupported không đổi thành direct.
- Slot 0 là predicted anchor cho một clause trong phạm vi này; association annotation không đi vào runtime.
- Chỉ image-frame trái/phải; metric depth/TF chưa thuộc S0.
- Lưu maps/logits/evidence tuyệt đối và status, không dùng softmax peak làm chứng nhận tồn tại.
- Một bộ kiểm tra trên peak chỉ xuất diagnostic, không hard refine hoặc ghi GEOMETRY_VERIFIED.
- Giữ nguyên deterministic MAP/Adapter khi nghiên cứu evidence-only; đổi representation/MAP sau này cần Adapter/calibration contract riêng.

**Ưu tiên tiếp theo:** thiết kế explicit noun-phrase conditioning/binding và presence evidence trên train/dev, có đối chứng với slot cũ. Chưa thể kết luận cần train module nào hoặc loss nào chỉ từ audit này. Trước huấn luyện cần xác định kiến trúc nhỏ nhất và gate development; S0 không tự mở việc train.

## 9. Artifact, kiểm chứng và giới hạn

- [Script audit](../new/scripts/audit_s0_anchor_binding.py), parser self-check đã pass.
- [Summary cuối](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/summary.json).
- [Bảng CSV từng mẫu](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/anchor_rows.csv).
- [Evaluator rows](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/evaluator_rows.jsonl).
- [Predictions lưu trước hậu kiểm](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/inference_predictions.jsonl).
- [Logits thật](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/spatial_logits.npz).
- [Swap-pair rows](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/swap_pairs.jsonl).
- [Provenance/hashes](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/provenance.json).
- [Kiểm chứng cuối](../new/outputs/pcrau_s0_anchor_audit_20261005_r2/validation_checks.json): 1380 file được bảo vệ khớp hash; predictions hai run giống nhau; graph-association cuối 400/400; MAP đối chiếu artifact dev 80/80.

Forward chỉ nhận chín MODEL_INPUT_KEYS: r0/d0, thumbnails, token IDs/mask, relation IDs/mask và prompt-derived anchor activation mask. Không forward target/anchor masks, object IDs, graph hoặc truth. Logits/predictions đã lưu trước khi mở evaluator record/masks/registry.

Model tensors không đổi sau inference; hashes của checkpoint/config/calibrator/profiles, manifest/index, features và evaluator inputs đã kiểm tra trước/sau. Target MAP trên **80/80 dev samples** khớp artifact `detail_cost/verified_dev_predictions.jsonl` đã có. Không chạy train, calibration, test, MC, robot, capture hoặc sửa archive/active profile.

Giới hạn: point-in-mask chưa phải toàn bộ anchor segmentation/identity quality; nhiều variant phụ thuộc cùng family; parser hẹp do grammar synthetic; chỉ một checkpoint; chưa attribution nguyên nhân, chưa calibrated presence, chưa chứng minh phrase conditioning sẽ cải thiện. S0 không tạo bằng chứng aleatoric/epistemic hoặc guarantee risk.

Lệnh tái lập audit khi có nhiệm vụ được giao, với output mới để không ghi đè:

```bash
.conda-roborefer/bin/python3.10 -B new/scripts/audit_s0_anchor_binding.py --self-check-only
.conda-roborefer/bin/python3.10 -B new/scripts/audit_s0_anchor_binding.py --output new/outputs/pcrau_s0_anchor_audit_reproduction
```
