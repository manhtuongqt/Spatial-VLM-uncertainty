# WP3 — Data-gap audit của Dataset WP2

> **Kết luận:** `GO_FEATURE_HOOK_SMOKE_ONLY_NO_TRAIN_NO_SCALE`
>
> **Thời điểm audit UTC:** 2026-08-21T04:34:57.308440+00:00
>
> Audit chỉ đọc đúng dataset WP2 đã khóa. Không train model, không sinh family/sample mới và không sửa artifact WP2.

## 1. Gate

| Gate | Kết quả |
|---|---:|
| `contract_lock_verified` | PASS |
| `exactly_250_records` | PASS |
| `exactly_50_families` | PASS |
| `five_variants_per_family` | PASS |
| `family_split_no_overlap` | PASS |
| `record_hashes_match_index` | PASS |
| `inference_payload_oracle_free` | PASS |
| `query_graph_anchor_capacity` | PASS |
| `query_graph_relation_capacity` | PASS |
| `rgb_depth_resolution_aligned` | PASS |
| `dataset_not_scaled` | PASS |
| `locked_inputs_unchanged_during_audit` | PASS |

## 2. Quy mô và phân bố chính

- 50 family, 250 record, 100 capture ID duy nhất;
- 30 family phụ thuộc depth;
- anchor tối đa 2, relation tối đa 1;
- 76 instruction duy nhất trên 250 record;
- image resolutions: 640x480.

### Answerability state

| State | Sample | Tỷ lệ |
|---|---:|---:|
| `ABSENT` | 6 | 2.4% |
| `AMBIGUOUS` | 6 | 2.4% |
| `FOUND` | 146 | 58.4% |
| `INSUFFICIENT_EVIDENCE` | 92 | 36.8% |

### State theo split

| Split | ABSENT | AMBIGUOUS | FOUND | INSUFFICIENT_EVIDENCE | Total |
|---|---:|---:|---:|---:|---:|
| calibration | 0 | 0 | 18 | 12 | 30 |
| dev | 0 | 0 | 24 | 16 | 40 |
| test | 3 | 0 | 17 | 10 | 30 |
| train | 3 | 6 | 87 | 54 | 150 |

## 3. Query-graph coverage

- `Kmax=3`: dataset max anchor = 2 → PASS;
- `Lmax=3`: dataset max relation = 1 → PASS;
- relation ngoài core WP3: more_elongated_than, semantic_reference, taller_than;
- reference frames: base_link, camera_color_optical_frame, image, object_semantics.

`object_semantics` và các relation shape/metric không được coi là deployable spatial relation supervision nếu chưa chuyển thành observable frame/evidence tương ứng.

## 4. Mask và instruction audit

- Target-mask fraction median: 0.049697;
- Interior/target ratio median trên target khác rỗng: 0.852224;
- Instruction word count: median 14.0, min 6.0, max 24.0;
- Record hash mismatch: 0;
- Forbidden inference finding: 0.

## 5. Data gaps bắt buộc xử lý trước calibration/locked test

| Mức | Gap | Bằng chứng | Hành động |
|---|---|---|---|
| CRITICAL | AMBIGUOUS_UNDERREPRESENTED | 6/250 sample | Tăng family ambiguous trước calibration/locked claims; không oversample variants như family độc lập. |
| CRITICAL | ABSENT_UNDERREPRESENTED | 6/250 sample | Tăng target/anchor absent families và giữ minimum negative locked evidence. |
| CRITICAL | DEV_MISSING_AMBIGUOUS_ABSENT | dev states={'FOUND': 24, 'INSUFFICIENT_EVIDENCE': 16} | Bổ sung dev families cho đủ bốn answerability states trước architecture/model selection. |
| CRITICAL | CALIBRATION_SPLIT_TOO_SMALL | 30 sample; states={'FOUND': 18, 'INSUFFICIENT_EVIDENCE': 12} | Không fit/claim multimodal calibration trên split prototype; preregister expansion theo family. |
| CRITICAL | PROTOTYPE_TEST_NOT_FINAL_LOCKED_TEST | 30 sample; states={'ABSENT': 3, 'FOUND': 17, 'INSUFFICIENT_EVIDENCE': 10} | Tạo Test-IID/Test-OOD mới sau development freeze; không dùng prototype test để chọn model. |
| HIGH | ORACLE_GRAPH_NOT_PREDICTED_GRAPH | relation_graph nằm trong evaluator_only ở toàn bộ record | Xây language/predicted query graph; oracle chỉ làm upper bound. |
| HIGH | NO_MULTI_CLAUSE_RELATION_COVERAGE | max relation count=1; Lmax contract=3 | Bổ sung multi-clause families nếu muốn giữ claim Lmax>1; nếu không, thu hẹp contract/model claim về một clause. |
| HIGH | NON_CORE_RELATION_TASKS | more_elongated_than, semantic_reference, taller_than | Tách shape/metric/semantic-reference khỏi core spatial relation head v1 hoặc định nghĩa head riêng. |
| HIGH | OBJECT_SEMANTICS_REFERENCE_FRAME | 137 record | Chuyển relation deployable sang image/camera/base evidence; không dùng semantics oracle khi inference. |
| HIGH | SIMULATOR_ONLY_DOMAIN | 100 raw Gazebo captures; chưa có real calibration/test split | Giới hạn claim simulator và thiết kế real recalibration sau offline gate. |
| MEDIUM | SOURCE_LABELS_ARE_CONTROLLED_PERTURBATIONS | source labels sinh từ generator/counterfactual | Dùng multi-label paired attribution; không tuyên bố causal diagnosis trên dữ liệu thật. |
| MEDIUM | INSTRUCTION_TEMPLATE_REUSE | 76 unique/250; 23 exact instructions xuất hiện ở nhiều split | Giữ paraphrase theo family và tạo language-template OOD; không coi random family split là language generalization. |
| MEDIUM | ASSET_AND_LAYOUT_REUSE | 9 target object IDs, 100 capture IDs trên 250 record | Test-OOD phải hold out asset/layout/view generator, không chỉ random family split. |

## 6. Quyết định

Audit cho phép bước tiếp theo là **runtime feature-hook verification và 15-sample determinism smoke**.

Audit **không** cho phép:

- tuyên bố dataset đủ calibration/generalization;
- train P-CRA-U;
- scale dataset mà chưa preregister data expansion;
- mở prototype test để chọn kiến trúc;
- dùng relation graph evaluator làm predicted graph.

## 7. Artifact

- Machine summary: `/home/dhcn/ur_ws/src/myproject/results/wp3_data_audit_20260821/audit_summary.json`;
- Record-level audit: `/home/dhcn/ur_ws/src/myproject/results/wp3_data_audit_20260821/tables/record_audit.csv`;
- Family-level audit: `/home/dhcn/ur_ws/src/myproject/results/wp3_data_audit_20260821/tables/family_audit.csv`;
- Count/crosstab tables: `/home/dhcn/ur_ws/src/myproject/results/wp3_data_audit_20260821/tables/`;
- Figures: `/home/dhcn/ur_ws/src/myproject/results/wp3_data_audit_20260821/figures/`;
- Artifact manifest: `/home/dhcn/ur_ws/src/myproject/results/wp3_data_audit_20260821/audit_manifest.json`.
