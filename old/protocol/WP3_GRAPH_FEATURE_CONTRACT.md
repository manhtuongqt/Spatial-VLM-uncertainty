# WP3 — Query Graph và Feature Contract v1

## 1. Trạng thái khóa

**Contract ID:** `wp3_graph_feature_contract_v1`

**Phạm vi khóa:** schema query graph, ranh giới oracle, vị trí feature hook, tensor/tile metadata bắt buộc, cache payload và các gate phải pass trước training.

**Trạng thái:** `CONFIG_LOCKED_RUNTIME_FEATURE_VERIFICATION_PENDING`.

Trạng thái này có nghĩa contract đã khóa theo source/config hiện có, nhưng shape và determinism của feature hook vẫn phải được đo runtime trước khi tạo feature cache chính thức. Không được diễn giải tài liệu này như bằng chứng feature hook đã chạy.

WP3 chưa cho phép:

- train P-CRA-U/P-CRA-F;
- scale dataset;
- mở calibration/test để chọn model;
- đưa oracle graph/mask vào inference cache;
- publish P-CRA target sang robot.

## 2. Mục tiêu

Khóa giao diện giữa:

```text
WP2 observable input
        ↓
RoboRefer RGB/depth towers
        ↓
R0/D0 feature hook + tile metadata
        ↓
query-centric graph / P-CRA-U sidecar
        ↓
predicted target/anchor distributions
```

Contract phải đảm bảo:

1. graph inference không chứa evaluator oracle;
2. target/anchor/relation slot có giới hạn rõ;
3. RGB/depth feature được bắt tại cùng semantic stage;
4. dynamic tile có mapping ngược về ảnh gốc;
5. thumbnail không bị coi nhầm là local tile;
6. cache feature không trộn label supervision;
7. hook không làm thay đổi output RoboRefer.

## 3. Input contract

### 3.1. Inference payload được phép

- JPEG/PNG RGB thực gửi RoboRefer;
- relative-depth PNG thực gửi RoboRefer;
- instruction chưa chứa answer oracle;
- coordinate suffix;
- cờ `enable_depth`;
- `sample_id`, input hashes và preprocessing metadata.

### 3.2. Sensor evidence chỉ dùng ở geometry stage

- registered metric depth;
- CameraInfo;
- TF snapshot và timestamp;
- RGB gốc nếu cần kiểm provenance.

Metric depth không được thay thế relative-depth input trong vision/depth tower và không được dùng để tạo oracle relation trong P-CRA inference.

### 3.3. Evaluator-only bị cấm ở inference

- target/anchor/interior/graspable/reachable masks;
- semantic/instance labels;
- Gazebo object IDs/poses;
- valid target IDs;
- oracle relation graph;
- answerability/source/expected-intervention labels;
- split/test evaluator answers.

Cache chứa bất kỳ field hoặc path nào ở nhóm này phải fail closed.

## 4. Query graph contract

Machine schema: [`query_graph_v1.schema.json`](query_graph_v1.schema.json).

### 4.1. Capacity

```text
Kmax = 3 anchor nodes
Lmax = 3 relation edges
1 target node
```

Mỗi graph có tối đa bốn node và ba edge. Slot không sử dụng được mask ở model layer, không tạo node giả trong serialized graph.

### 4.2. Graph modes

| Mode | Access | Nội dung |
|---|---|---|
| `language` | inference | mention, target/anchor role, predicate và reference frame từ instruction |
| `predicted` | inference | language graph + distribution refs/raw evidence dự đoán |
| `oracle` | evaluator-only | object IDs, mask refs và relation label từ dataset |

`oracle` không được serialize vào inference request hoặc feature cache.

### 4.3. Node semantics

- `target`: thực thể phải được ground;
- `anchor_i`: thực thể tham chiếu dùng để giải relation;
- direct query chỉ có target;
- predicted node tham chiếu một spatial distribution, không tham chiếu evaluator mask;
- raw evidence không được đặt tên `probability` hoặc `confidence` trước calibration.

### 4.4. Edge semantics

Edge luôn hướng từ target tới anchor:

```text
target ─predicate→ anchor_i
```

Tránh dùng tên `source_ids/target_ids` trong predicted graph vì “target” dễ bị nhầm với grounding target. Schema dùng `subject_node_id/object_node_ids`.

Predicate registry v1:

```text
direct
left_of / right_of
nearer_than / farther_than
front_of / behind
between_in_depth / nearer_than_both
semantic_reference
more_elongated_than / taller_than
```

Core WP3 relation set:

```text
direct
left_of / right_of
nearer_than / farther_than
front_of / behind
between_in_depth / nearer_than_both
```

`semantic_reference`, `more_elongated_than` và `taller_than` được audit nhưng chưa bắt buộc cho learned spatial relation head v1.

### 4.5. Reference frame

Allowed serialized values:

```text
image
camera_color_optical_frame
base_link
object_semantics
```

Rules:

- `left/right` mặc định phải ghi `image` hoặc camera frame;
- `near/far/front/behind` phải ghi camera frame nếu dùng metric/projective evidence;
- `base_link` chỉ dùng sau TF transform;
- `object_semantics` trong WP2 là annotation semantics và phải được chuyển thành frame deployable hoặc đánh dấu evaluator-only khi train relation geometry.

## 5. Checkpoint-derived feature contract

### 5.1. Config đã xác minh tĩnh

| Field | RGB tower | Depth tower |
|---|---:|---:|
| Encoder | SigLIP | SigLIP |
| Input tile | 448×448 | 448×448 |
| Patch size | 14 | 14 |
| Hidden size | 1152 | 1152 |
| Selected layer | -2 | -2 |
| Selected feature | `cls_patch` | `cls_patch` |
| Model dtype | BF16 | BF16 |
| Projector | `mlp_downsample_3x3_fix` | `mlp_downsample_3x3_fix` |
| LLM hidden size | 1536 | 1536 |

Nguồn khóa:

- `RoboRefer/models/RoboRefer-2B-SFT/config.json`;
- tower/projector configs và preprocessors trong checkpoint;
- `RoboRefer/llava/model/multimodal_encoder/vision_encoder.py`;
- `RoboRefer/llava/model/multimodal_projector/base_projector.py`;
- `RoboRefer/llava/mm_utils.py`.

### 5.2. Hook points

Feature hook bắt output của:

```text
model.get_vision_tower()(rgb_tiles)  → R0_raw
model.get_depth_tower()(depth_tiles) → D0_raw
```

trước:

```text
model.get_mm_projector()
model.get_depth_projector()
```

Không dùng forward hook có side effect. Adapter/hook API phải trả bản tham chiếu detached hoặc clone có kiểm soát và giữ nguyên đường generate gốc.

### 5.3. Expected tensor shape và runtime gate

Từ config:

```text
448 / 14 = 32 patches mỗi chiều
expected spatial tokens = 32×32 = 1024
expected hidden = 1152
```

Expected pre-projector shape:

```text
R0_raw, D0_raw: [N_tiles, 1024, 1152]
```

Projector 3×3 pad grid 32 thành 33 và cho kỳ vọng:

```text
R, D: [N_tiles, 11×11, 1536]
```

Tuy nhiên `mm_vision_select_feature=cls_patch` và encoder implementation phải được xác nhận runtime. Nếu token count khác 1024 hoặc không reshape được thành square grid, runtime gate fail; không tự ý bỏ token hoặc sửa projector trong locked run.

### 5.4. Dynamic tiling

Checkpoint config:

```text
image_aspect_ratio = dynamic
min_tiles = 1
max_tiles = 12
use_thumbnail = true trong dynamic_preprocess
```

Metadata bắt buộc cho từng modality:

```text
original_width, original_height
tile_size
grid_columns, grid_rows
local_tile_count
thumbnail_present
tile_index
tile_kind = local | thumbnail
resized_canvas_xyxy
original_image_xyxy
patch_grid_height, patch_grid_width
```

RGB và depth phải có cùng original resolution, grid, local-tile count và tile order. Sai khác làm sample fail.

### 5.5. Mapping local tile về ảnh gốc

Với resized canvas `(Wc,Hc)`, local tile box `(x0,y0,x1,y1)`:

```text
u_original = u_canvas × W_original / Wc
v_original = v_canvas × H_original / Hc
```

Patch cell được map qua đúng tile box và transform resize. Metadata lưu cả box trên canvas và box tương ứng ở ảnh gốc để không phải suy luận lại khi evaluate.

### 5.6. Thumbnail rule

Thumbnail là global-context tile và map lên toàn ảnh. Trong P-CRA-U v1:

- local tiles tạo primary spatial logits;
- thumbnail chỉ được dùng làm global context/gating;
- không cộng thumbnail patch logits vào local heatmap như một tile thứ 13 bình thường;
- nếu thử thumbnail spatial logits, đó phải là ablation riêng với merge rule preregister.

### 5.7. Heatmap merge rule

Primary v1:

1. sinh local patch logits theo tile;
2. resize từng tile logits về original tile box;
3. merge overlap bằng arithmetic mean của logits và coverage count;
4. softmax một lần sau khi có global canvas;
5. vùng không được local tile phủ phải fail, không điền confidence giả.

Dynamic tiling hiện không tạo overlap giữa local tiles, nhưng coverage-count metadata vẫn bắt buộc để contract không phụ thuộc giả định này.

## 6. Feature cache contract

### 6.1. Inference cache được phép chứa

```text
schema_version
sample_id / family_id / split
input RGB/depth hashes
instruction hash
checkpoint/config/source hashes
R0/D0 tensors
tile metadata
dtype/shape/device-extraction metadata
baseline raw answer / p_RR / parse status
latency / peak VRAM
```

### 6.2. Bị cấm trong inference cache

- mọi evaluator-only mask/ID/relation/answerability/source label;
- expected intervention;
- model correctness label;
- test metric;
- Gazebo oracle pose.

Supervision loader đọc evaluator labels từ dataset qua một channel tách biệt sau khi feature cache đã được khóa.

### 6.3. Storage

- tensor format ưu tiên `safetensors`;
- một manifest canonical JSON;
- SHA-256 từng shard/file;
- explicit dtype/shape;
- không pickle executable payload;
- write-once output directory;
- calibration/test chưa được cache trong WP3 smoke.

## 7. Determinism và non-interference gate

Trên 15 WP2 smoke samples, chạy hai lần độc lập cùng checkpoint/config/seed:

1. shape/dtype/tile metadata phải giống tuyệt đối;
2. input và output file hashes phải được khóa;
3. tensor comparison phải báo max-absolute và max-relative difference;
4. BF16 deterministic target là exact equality nếu runtime hỗ trợ; nếu không, tolerance phải được preregister trước khi xem sample labels;
5. baseline raw answer và parsed `p_RR` phải giống run không hook;
6. peak VRAM/latency được báo, không dùng làm pass nếu chưa khóa budget;
7. pilot/WP1/WP2 digests không đổi.

Nếu hook thay đổi baseline output, quyết định là `FIX_FEATURE_PIPELINE_FIRST`.

## 8. Graph validation gates

- schema draft 2020-12 load được;
- language graph không có `object_ids/mask_refs`;
- predicted graph không có `object_ids/mask_refs`;
- oracle graph bắt buộc `access_scope=evaluator_only`;
- tối đa 1 target, 3 anchors, 3 edges;
- node IDs duy nhất;
- edge chỉ tham chiếu node tồn tại;
- relation và reference frame thuộc registry;
- direct query không bắt buộc anchor;
- non-direct edge phải có anchor;
- raw evidence chưa được gọi là probability.

Các constraint liên-node không biểu diễn đầy đủ bằng JSON Schema phải được kiểm bằng validator code trước cache/training.

## 9. Audit gate trước feature hook

Audit đúng dataset WP2 đã khóa phải trả lời:

- 250 record/50 family/5 variant;
- split family không overlap;
- record hashes đúng index;
- inference payload không oracle;
- distribution state/category/relation/source/reference frame;
- anchor/relation capacity so với `Kmax/Lmax`;
- image resolution và mask-area distribution;
- ambiguous/absent/calibration/test data gaps;
- relation ngoài core v1;
- object/layout/capture reuse;
- dataset input digests không đổi trước/sau audit.

Audit pass không có nghĩa dataset đủ train/calibration. Nó chỉ cho phép chuyển sang feature-hook smoke.

## 10. Quyết định sau contract

Trình tự hợp lệ:

```text
LOCK CONTRACT
→ AUDIT 250 WP2 SAMPLES
→ FEATURE HOOK RUNTIME VERIFICATION
→ 15-SAMPLE DETERMINISM SMOKE
→ FEATURE CACHE TRAIN/DEV PROTOTYPE
→ P-CRA-U SMOKE TRAIN
```

Không đảo thứ tự và không scale dataset trước khi data-gap audit được đọc và scope expansion được preregister.
