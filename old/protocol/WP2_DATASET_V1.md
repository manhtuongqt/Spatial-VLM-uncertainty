# WP2 — Dataset v1

## 1. Kết luận

WP2 đã hoàn thành ở mức **prototype 50 family × 5 variant = 250 sample** cho đề tài:

> **Nghiên cứu phương pháp LLM/VLM Agent lập kế hoạch vòng kín với hiệu chuẩn độ bất định không gian và can thiệp thích ứng cho thao tác gắp đặt bằng tay máy UR3.**

Dataset chính thức nằm tại:

`datasets/roborefer_dataset_v1_prototype_20260818_155606/`

Kết quả cuối:

- 50 `family_id` và 250 record;
- 30 family phụ thuộc depth, 20 family còn lại bao phủ direct grounding, quan hệ 2D, ambiguous/absent, shape và metric comparison;
- 100 raw capture thật từ Gazebo Fortress ở `VIEW_POSE` đã khóa;
- 250/250 record qua JSON Schema, hash, mask QC và oracle-isolation;
- 85 record có quan hệ projective/depth được đối chiếu trực tiếp với median depth hoặc centroid từ capture và đều pass;
- replay offline 2.334 file dẫn xuất không có một mismatch SHA-256 nào;
- không sử dụng 10 scene WP0/WP1 làm nguồn train;
- chưa huấn luyện P-CRA-U hoặc P-CRA-F.

WP2 vì vậy mở đường cho **WP3 — P-CRA-U prototype**. `GO_PCRA_F` của WP1 vẫn chỉ là tín hiệu sơ bộ; WP2 không mở huấn luyện P-CRA-F.

## 2. Dataset contract đã khóa

Mỗi record có bốn lớp tách biệt:

1. `inference_payload`: chỉ có JPEG RGB thực sự gửi model, relative-depth PNG thực sự gửi model, prompt, coordinate suffix và cờ depth;
2. `sensor_evidence`: RGB PNG gốc, metric depth NPY, relative depth sạch, camera info, TF và timestamp;
3. `evaluator_only`: semantic/instance labels, target/anchor/interior/graspable/reachable/valid-depth masks, relation graph, target set, answerability và expected intervention;
4. `provenance`: family seed, generator version, perturbation, capture backend và SHA-256 code/config/checkpoint.

Validator duyệt đệ quy key và path của `inference_payload`. Các key như `target_mask`, `answerable`, `oracle`, `relation_graph`, `expected_intervention` hoặc đường dẫn evaluator đều làm gate fail.

Schema:

- `protocol/dataset_v1.schema.json`;
- `protocol/family_manifest_v1.schema.json`.

## 3. Family và split

| Nhóm family | Số family | Số sample | Phụ thuộc depth |
|---|---:|---:|---|
| `nearer_farther` | 12 | 60 | Có |
| `front_behind_camera` | 8 | 40 | Có |
| `multi_anchor_depth_order` | 5 | 25 | Có |
| `occlusion_depth_evidence` | 5 | 25 | Có |
| `direct_grounding` | 5 | 25 | Không |
| `relation_2d` | 5 | 25 | Không |
| `ambiguous_absent` | 4 | 20 | Không |
| `shape_comparison` | 3 | 15 | Không |
| `metric_comparison` | 3 | 15 | Không |
| **Tổng** | **50** | **250** | **30 family depth-dependent** |

Split được gán theo `family_id`, không theo frame:

| Split prototype | Family | Sample |
|---|---:|---:|
| Train | 30 | 150 |
| Dev | 8 | 40 |
| Calibration | 6 | 30 |
| Test | 6 | 30 |

Mọi clean/counterfactual/view của cùng family nằm trong đúng một split. Split `test` ở đây là **prototype test split**, chưa phải locked Test-IID/Test-OOD cuối luận văn.

Mỗi family có đúng năm variant:

1. `clean`;
2. `semantic_counterfactual`;
3. `relation_counterfactual`;
4. `depth_corruption`;
5. `occlusion_view_counterfactual`.

Semantic/relation counterfactual có thể dùng cùng raw scene nhưng đổi query và valid target set. Depth corruption giữ RGB, làm hỏng đúng relative-depth model input. Occlusion được render thành một raw Gazebo capture riêng, không vẽ giả lên RGB.

## 4. Smoke gate và failure audit của generator

WP2 không scale ngay ở lần đầu. Lịch sử QC được giữ lại để báo cáo:

| Mốc | Kết quả | Phát hiện/hành động |
|---|---|---|
| Attempt 1 | Reject | `FOUND` nhưng target/interior mask rỗng do vật nằm trong vùng robot tự che |
| Attempt 2 | Reject trước scale | Static audit phát hiện nguy cơ chọn trùng target/second-anchor/occluder |
| Attempt 3 smoke | Pass | 3 family đại diện × 5 variant, replay/leakage/mask pass |
| Full capture | Pass | đủ 100 raw capture và hash |
| Full QC | Amend trước training | sửa anchor absent-family và nhãn quan hệ theo evidence; split/layout/raw capture không đổi |
| Final gate | Pass | 50 family, 250 sample, 30 depth-dependent |

Các amendment đều xảy ra **trước training**, không xem output model, không đổi split và không đổi raw capture. Manifest trước/sau amendment được giữ trong `report_assets/checkpoints/`.

## 5. Số liệu QC

Raw Gazebo capture:

- 100/100 capture có RGB, registered metric depth, semantic labels, camera info và TF;
- RGB–depth–label max spread: 0 đến 0,033 giây, trung bình 0,01188 giây;
- RGB gray standard deviation: 57,25 đến 64,44;
- valid depth fraction: 1,0 trong simulator;
- depth hữu hiệu toàn tập: khoảng 0,168 đến 0,485 m.

Record/mask:

- 250/250 record hợp lệ theo schema;
- tổng target-mask: 3.841.724 pixel;
- tổng interior-mask: 3.311.856 pixel;
- trạng thái: 146 `FOUND`, 92 `INSUFFICIENT_EVIDENCE`, 6 `AMBIGUOUS`, 6 `ABSENT`;
- expected intervention: 146 `EXECUTE`, 92 `REOBSERVE`, 6 `ASK_USER`, 6 `ABSTAIN`;
- `FOUND` bắt buộc target/interior và mọi anchor khác rỗng;
- `EXECUTE` bắt buộc graspable/reachable mask khác rỗng;
- `ABSENT` bắt buộc target mask rỗng;
- interior ⊆ target, reachable ⊆ graspable ⊆ target.

Quan hệ:

- 85 record `nearer/farther/front/behind/between/right/left` được kiểm bằng median metric depth hoặc mask centroid;
- 11 record shape/metric/direct có nhãn phi projective được báo riêng;
- 154 record không áp dụng relation check do absent/ambiguous/depth-corruption/occlusion state hoặc không có anchor;
- 0 mismatch ở gate cuối.

Replay:

- 250 record được materialize lại từ raw capture đã khóa;
- 2.334 file dẫn xuất được so sánh byte-for-byte;
- 0 mismatch;
- “replay deterministic” không đồng nghĩa Gazebo rerender lại pixel giống hệt. WP2 chỉ tuyên bố deterministic cho quá trình offline từ raw capture bất biến.

## 6. Artifact dùng cho báo cáo

Folder riêng:

`datasets/roborefer_dataset_v1_prototype_20260818_155606/report_assets/`

Nội dung:

- `checkpoints/`: schema/family lock, smoke gate, raw capture, các QC amendment và full gate;
- `figures/`: phân bố family/split/intervention, mask area, montage 9 task family, depth-corruption examples và QC gate history;
- `tables/`: CSV phân bố family/split, capture QC, mask QC và relation QC;
- `gui_previews/`: 250 ảnh overlay target/anchor/reachable dành riêng cho evaluator;
- `ASSET_INDEX.json`: SHA-256 và kích thước từng report asset.

GUI offline:

`datasets/roborefer_dataset_v1_prototype_20260818_155606/WP2_QC_GUI.html`

GUI lọc theo split/category/family/variant, hiển thị đồng thời instruction, state, intervention, mask area và link đến RGB/depth/mask/record. Overlay evaluator trong GUI tuyệt đối không phải model input.

## 7. Validator và lệnh tái lập

Các thành phần:

- generator: `protocol/wp2_family_generator.py`;
- Gazebo capture: `protocol/wp2_gazebo_capture.py` và launch tương ứng;
- materializer: `protocol/wp2_materialize.py`;
- replay validator: `protocol/wp2_replay_validator.py`;
- leakage validator: `protocol/wp2_leakage_validator.py`;
- full validator: `protocol/wp2_validate.py`;
- report generator: `protocol/wp2_generate_report_assets.py`.

Chạy toàn bộ software test + smoke/full gate + leakage + replay bằng một lệnh:

```bash
./protocol/run_wp2_checks.sh
```

Kết quả cuối: 86 test pass, 5 warning từ `image_geometry` về `numpy.matrix`; smoke và full validator pass; leakage pass; replay pass.

## 8. Giới hạn phải ghi đúng trong luận văn

- `graspable_mask` và `reachable_mask` hiện là proxy bảo thủ từ visible target, valid depth và vùng XY đã khóa; chưa phải output collision/reachability đầy đủ của MoveIt. Record ghi rõ policy này.
- Valid-depth fraction 100% là đặc tính capture Gazebo hiện tại, không được suy rộng sang RealSense thật.
- Semantic camera dùng unique label theo object trong world hiện tại; cần instance-segmentation rõ hơn khi scene có nhiều instance cùng asset.
- Có lỗi `move_group` teardown sau khi capture đã hoàn tất; artifact capture vẫn đủ và đúng hash, nhưng lỗi shutdown hạ tầng vẫn cần sửa trước robot protocol dài.
- 50-family prototype đủ kiểm schema/generator/QC, chưa đủ để huấn luyện và tuyên bố tổng quát hóa mạnh.
- 20% failure audit của model trên dataset phát triển vẫn chưa thực hiện; vì vậy P-CRA-F tiếp tục là nhánh ablation có điều kiện.

## 9. Bước tiếp theo

Sang **WP3 — P-CRA-U prototype**, theo thứ tự:

1. hook và kiểm tensor `R0/D0` trước projector;
2. khóa tile mapping 640×480 ↔ 448×448 ↔ patch grid;
3. cache feature cho train/dev prototype với provenance và hash;
4. triển khai relation-query extractor đa anchor;
5. smoke heatmap/answerability/source heads trên dev;
6. chỉ sau khi P-CRA-U, calibration và failure audit ổn định mới mở P-CRA-F như ablation.

