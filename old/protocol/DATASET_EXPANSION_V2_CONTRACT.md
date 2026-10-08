# Dataset Expansion V2 — family, split, OOD và lifecycle contract

> **Quyết định:** `GO_DATASET_EXPANSION_DESIGN_LOCK`
>
> **Trạng thái:** `LOCKED_DESIGN_NO_CAPTURE_NO_TRAINING`
>
> **Hành động tiếp theo duy nhất được phép:** `GO_DRY_RUN_30_PILOT_ONLY`
>
> **Revision:** `v2.0.1` — bỏ figure planned thủ công theo yêu cầu ngày
> 2026-08-21; thay đổi reporting-only, trước mọi capture/training.

## 1. Mục tiêu và ranh giới

Contract này khóa thiết kế mở rộng dataset theo `scene-query family` cho
P-CRA-U. Nó xử lý các gap đã ghi trong `WP3_DATA_GAP_AUDIT.md`: thiếu
`AMBIGUOUS`/`ABSENT`, dev và calibration chưa đủ bốn trạng thái, prototype
test chưa phải Test-IID/Test-OOD, instruction template tái sử dụng và chưa có
asset/layout/view holdout.

Design lock này không cấp quyền:

- training development;
- capture 400 family chính thức;
- capture hoặc đọc calibration/Test-IID/Test-OOD;
- scale 2.000–4.000 family;
- dùng evaluator oracle làm input inference.

WP2 50 family/250 sample giữ nguyên `PROTOTYPE_ENGINEERING_ONLY`. WP2 có thể
dùng cho schema/loader/hook/overfit/failure audit và, nếu ghi cờ rõ ràng, warm
start train-only. Không family WP2 nào được đưa vào final dev selection,
calibration, Test-IID hoặc Test-OOD.

## 2. Đơn vị độc lập và năm variants

Đơn vị split, bootstrap và power analysis là một `scene_query_family`. Mọi
view, paraphrase, clean, corrupted và counterfactual của cùng family ở cùng
split. Không đếm năm variants như năm quan sát độc lập.

Mỗi family tiếp tục có đúng năm variants:

1. `clean`;
2. `semantic_counterfactual`;
3. `relation_counterfactual`;
4. `depth_corruption`;
5. `occlusion_view_counterfactual`.

Số state trong thiết kế phát triển là `primary_answerability_stratum` ở mức
family. Phân bố state thật ở mức variant/sample phải được audit và báo riêng
sau capture; không được thay số family bằng số sample.

## 3. Ontology answerability đã khóa

| State | Denotation trong world | Bằng chứng hiện tại | Mặc định |
|---|---:|---:|---|
| `FOUND` | đúng 1 | đủ | `EXECUTE` |
| `AMBIGUOUS` | từ 2 trở lên | đủ để biết không duy nhất | `ASK_USER` |
| `ABSENT` | 0 | đủ để kết luận rỗng | `ABSTAIN` |
| `INSUFFICIENT_EVIDENCE` | đúng 1 ở latent world | không đủ | `REOBSERVE` |

Quy tắc phân biệt bắt buộc:

- target/anchor thật sự không có hoặc không có object thỏa relation →
  `ABSENT`;
- target/anchor tồn tại nhưng bị che, depth hỏng, quá nhỏ hoặc ngoài view →
  `INSUFFICIENT_EVIDENCE`;
- có nhiều object cùng thỏa query → `AMBIGUOUS`, kể cả semantic tie,
  relation tie hoặc multi-anchor conflict;
- `FOUND` chỉ hợp lệ khi valid referent set có đúng một phần tử và evidence
  quan sát đủ.

Label và relation graph đầy đủ nằm trong `evaluator_only`. Inference payload
không được chứa target/anchor ID, mask, answerability label, valid target set,
oracle graph hoặc expected intervention.

## 4. Pilot-only trước scale

Dry-run có đúng 30 family/150 sample và split duy nhất `pilot_only`:

| Primary stratum | Family |
|---|---:|
| `FOUND` | 8 |
| `INSUFFICIENT_EVIDENCE` | 8 |
| `AMBIGUOUS` | 7 |
| `ABSENT` | 7 |

Sáu family category có 5 family/category. Sáu điều kiện `none`, asset, layout,
viewpoint, language và depth-noise có 5 family/điều kiện để smoke pipeline.

Mọi pilot family ID, seed, capture và ảnh bị loại vĩnh viễn khỏi train/dev/
calibration/test. Pilot chỉ kiểm generator, capture, label, replay, leakage và
QC; không dùng để ước lượng hiệu năng.

## 5. Development expansion đã khóa

Development đầu tiên có 400 family/2.000 planned variants:

| Split | Family | `FOUND` | `INSUFFICIENT` | `AMBIGUOUS` | `ABSENT` |
|---|---:|---:|---:|---:|---:|
| Train | 320 | 128 | 80 | 56 | 56 |
| Dev | 80 | 32 | 20 | 14 | 14 |
| **Tổng** | **400** | **160** | **100** | **70** | **70** |

Family category:

| Category | Train | Dev | Tổng | Depth-dependent |
|---|---:|---:|---:|---:|
| `direct_grounding` | 48 | 12 | 60 | Không |
| `relation_2d` | 64 | 16 | 80 | Không |
| `nearer_farther` | 56 | 14 | 70 | Có |
| `front_behind_camera` | 48 | 12 | 60 | Có |
| `multi_anchor_depth_order` | 48 | 12 | 60 | Có |
| `occlusion_depth_evidence` | 56 | 14 | 70 | Có |
| **Tổng** | **320** | **80** | **400** | **260/400** |

Shape/metric/semantic-reference không nằm trong core family category V2. Nếu
đưa trở lại, phải có head/evidence contract riêng và tạo version thiết kế mới.

Development capture chỉ được mở sau khi dry-run 30 family đạt gate. Training
chỉ được mở sau khi toàn bộ 400 family đạt schema, sensor, label, leakage,
replay và hash gate.

## 6. Asset partition và clone

Seen pool dùng các asset WP2/scene cũ cộng `ycb_lemon_01` và `ycb_mug_01`.
Asset-OOD được giữ riêng:

- `ycb_pear_01`;
- `ycb_plum_01`;
- `ycb_tuna_fish_can_01`.

Ba asset này có thể xuất hiện trong pilot smoke nhưng pilot không được tái sử
dụng. Trong dữ liệu official, chúng chỉ được xuất hiện ở Test-OOD. Chúng bị
cấm trong train, dev, calibration và Test-IID.

`ycb_bleach_cleanser_01`, `cup` và `kettle` bị loại khỏi tất cả split theo yêu
cầu cleanup scene.

Ba instance của apple/orange/banana dùng chung mesh/semantic class. Clone là
instance khác nhau trong cùng scene, không phải category/asset độc lập và
không làm tăng `n_families` hay số asset OOD.

## 7. Thiết kế Test-OOD

Năm single-axis strata:

1. unseen asset;
2. held-out layout/density/topology;
3. held-out camera elevation/azimuth;
4. held-out language-template bank;
5. held-out depth/noise generator.

Trong single-axis family, bốn trục còn lại phải dùng cấu hình IID. Ít nhất 80%
Test-OOD family là single-axis; crossed hard cases không quá 20% và phải báo
riêng. Không gộp tất cả trục rồi tuyên bố nguyên nhân cụ thể.

## 8. Calibration và locked-test lifecycle

Thiết kế được khóa bây giờ nhưng assignment/capture chưa được tạo:

| Split | Điều kiện mở |
|---|---|
| Calibration | model architecture, weights và best-dev checkpoint đã freeze |
| Test-IID | calibrator, risk event, thresholds và coverage target đã freeze |
| Test-OOD | cùng điều kiện Test-IID; mở một lần |

Không dùng calibration/test để chọn kiến trúc, loss, checkpoint, ablation hoặc
ngưỡng. Calibration và Test-IID dùng seen asset/generator distribution nhưng
family/capture/template khác nhau. Test-OOD tuân thủ partition ở Mục 7.

Quy mô candidate full nằm trong 2.000–4.000 official families và chỉ được khóa
sau development failure audit, runtime estimate và power analysis không nhìn
locked-test result. Calibration, Test-IID và Test-OOD mỗi split phải có ít nhất
60 family độc lập thuộc `AMBIGUOUS ∪ ABSENT`. Mốc 60 chỉ hỗ trợ bound tổng thể
kiểu rule-of-three quanh 5% khi có 0 lỗi; subgroup claim cần denominator riêng.

## 9. Leakage và split independence

Các identity sau không được trùng cross-split:

- `family_id` và `capture_id`;
- raw RGB/depth/semantic-label capture;
- layout seed;
- exact instruction;
- language-template family ID.

Bắt buộc kiểm thêm near-duplicate instruction, asset holdout và generator ID.
Mọi counterfactual/view/paraphrase giữ cùng family. Mọi split comparison và CI
bootstrap theo family.

## 10. Gate dry-run bắt buộc

Dry-run chỉ pass nếu:

- đủ bốn answerability states và đúng ontology;
- bao phủ target absent, anchor absent, ambiguity tie và insufficient evidence;
- family/capture/layout/template không rò rỉ;
- held-out asset không lọt vào official seen split;
- RGB, registered metric depth và semantic label đồng bộ;
- target/interior/anchor/graspable/reachable/valid-depth masks hợp lệ;
- relation và query-graph label nhất quán với metric depth/centroid;
- inference payload không có oracle;
- replay deterministic và artifact hash khớp;
- pilot không có ID/seed/capture đủ điều kiện tái dùng;
- WP0–WP3 và toàn bộ WP2 giữ nguyên hash.

Pass dry-run mới cho `GO_DEVELOPMENT_CAPTURE_400`. Fail thì quyết định là
`FIX_DATASET_V2_GENERATOR_OR_CAPTURE_PIPELINE_FIRST`.

## 11. Bằng chứng được phép ở giai đoạn này

Chỉ được cập nhật:

- Bảng 1 — dataset/statistical independence, với planned row ghi rõ
  `PLANNED_NOT_CAPTURED`;
- design-lock validation report, hash và schema test.

Không tạo F01 hay bất kỳ chart/figure planned thủ công nào trước capture. F01
chỉ được sinh tự động sau khi dataset thật đã capture và qua QC, trực tiếp từ
CSV/JSON đã hash. Figure phải có dữ liệu nguồn đi kèm, denominator là family,
phân biệt family/sample, và không được trình bày planned counts như bằng chứng
quan sát.

Bảng 2–6 giữ nguyên `NOT_RUN`. Không có grounding, answerability, calibration,
source-attribution hay ablation claim từ design lock hoặc pilot.

## 12. Artifact và change control

Artifact máy đọc được:

- `dataset_expansion_v2_spec.json`;
- `dataset_expansion_v2_asset_partition.json`;
- `dataset_expansion_v2_seed_lock.json`;
- `dataset_split_v2.schema.json`;
- `validate_dataset_expansion_v2.py`;
- `dataset_expansion_v2_design_lock.json` sau validation cuối.

Seed lock dùng hash-sort deterministic và commitment SHA-256 cho toàn bộ 30
pilot assignment và 400 development assignment mà không cần ghi 430 dòng vào
lock. Sửa count, seed, label order, asset partition, schema hoặc validator làm
design-lock hash fail.

Mọi sửa đổi trước capture phải tăng version và khóa lại với lý do. Sau khi một
locked test được mở, không được thay split đó; lỗi nghiêm trọng phải báo run bị
invalid và tạo protocol version mới, không âm thầm sửa label/split.
