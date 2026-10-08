# S0 — Contract đầu vào/đầu ra cho geometric verifier, v0

Ngày: **05/10/2026**. Trạng thái: **contract thiết kế sau audit**, chưa triển khai verifier runtime. Nguồn: [S0 audit](S0_ANCHOR_BINDING_AUDIT_20261005.md).

## 1. Mục đích và phạm vi

Verifier kiểm tra compatibility hình học giữa **predicted target và predicted anchor** theo query có cấu trúc. Phạm vi đầu tiên: một clause `left_of/right_of`, một anchor, hệ tọa độ ảnh. `direct` bypass; quan hệ depth/ternary/nhiều clause/frame khác trả unsupported trong phiên bản này.

Sau S0, anchor binding chưa đủ chắc để hard refine. MVP tiếp theo chỉ được nghiên cứu **evidence-only**, giữ MAP và Adapter deterministic. Đây là contract giới hạn phạm vi, không tự mở train, calibration hay robot.

## 2. Đầu vào observable

| Trường | Nguồn | Quy định |
|---|---|---|
| Prompt | Inference payload | Không nhận graph/object IDs evaluator |
| Typed query | Parser prompt-only | Target/anchor phrases, predicate, frame, parse status; không tự điền clause thiếu |
| Target logits/map | Frozen Sidecar | Giữ output gốc trước verifier |
| Anchor slot logits/map | Frozen Sidecar | Slot 0 trong scope một clause; capacity ba slot không đồng nghĩa hỗ trợ arbitrary graph |
| Image/grid metadata | Contract preprocessing | 640×480, grid 32×24; cùng crop/resize/frame giữa target và anchor |
| Evidence quan sát hiện có | Logits/model outputs hoặc input hợp lệ | Schema/version rõ; không gọi uncalibrated score là probability presence |

`anchor_mask` của model là **slot activation từ prompt**, không phải mask ảnh. Ground-truth pixel masks, labels, object IDs, valid-target list, relation graph và variant ID không đi vào verifier. Chúng chỉ vào evaluator hậu kiểm.

Không thêm metric depth sạch, camera intrinsics/TF hoặc sensor fields khác vào contract hiện tại. Nếu mở nhánh depth phải audit input parity và corruption provenance trước; relative depth không mặc định là mét.

## 3. Query và binding

```text
target_phrase: string
anchor_phrases: [string]        # một anchor trong horizontal scope
predicate: left_of | right_of | direct
reference_frame: image        # với left/right
parse_status: supported | unsupported | ambiguous
anchor_slot_index: 0           # không phải object ID
binding_status: unverified    # chưa có module xác minh identity/presence
```

Nếu parse failure, unsupported frame hoặc sai cardinality, không phát `direct` ngầm. Tại S0 parser chỉ là helper audit. **Cập nhật S1:** đã có [typed parser độc lập](../new/src/pcrau/query_parser.py), version `pcrau_query_parser_s1_v1`, với phrase char/token spans và conditioning masks; xem [audit S1](S1_QUERY_PARSER_AUDIT_20261005.md). Chưa nối parser vào baseline forward hoặc triển khai verifier runtime; direct tiếp tục bypass.

Binding_status không được đổi thành verified chỉ vì sigmoid-max cao. S0 có wrong-anchor scores ≥0,9 và empty-mask score 0,995801. Không được lấy annotation runtime để chuyển status. Module phrase-conditioned/binding mới, nếu phát triển, phải có schema và evaluator riêng.

## 4. Phép kiểm tra hình học

Hệ ảnh có origin góc trên trái, x tăng sang phải, y tăng xuống; dùng pixel tọa độ cùng frame. Reconstruction peak phải giống baseline:

```text
u_peak = int((grid_x + 0.5) * image_width / grid_width)
v_peak = int((grid_y + 0.5) * image_height / grid_height)
```

Quy ước dataset trên **mask centroid**:

```text
right_of: u_target_centroid − u_anchor_centroid − 12 > 0
left_of:  u_anchor_centroid − u_target_centroid − 12 > 0
```

12 pixel là margin của geometry annotation hiện có, không phải ngưỡng confidence đã calibration. Peak-in-mask không nhất thiết là centroid. Full-distribution mean cũng có thể bị background/nhiều modes kéo lệch. Verifier phải ghi representation đang dùng; phiên bản so peaks chỉ là `peak_relation_diagnostic`.

Nếu phát triển candidate-level hoặc soft kernel, phải định nghĩa candidate extraction/centroid/margin và chọn trên train/dev trước freeze. Không dùng GT masks hoặc semantic labels làm candidate detector. Không suy probability predicate correctness từ raw compatibility.

## 5. Đầu ra và xử lý thiếu evidence

```text
scope_status: DIRECT_BYPASS | SUPPORTED_DIAGNOSTIC | PARSE_UNSUPPORTED
binding_status: UNVERIFIED | [status của module tương lai đã kiểm chứng]
geometry_representation: peak | [representation đã khai báo]
signed_margin_px: number | null
raw_compatibility: number | null
anchor_evidence: {sigmoid_max, entropy, softmax_peak, ...}
failure_flags: [unsupported_scope, binding_unverified, nonfinite, ...]
original_target_map: unchanged
trace: inputs/slot/coordinate convention/calculation
```

MVP không phát chứng nhận `GEOMETRY_VERIFIED` chỉ từ diagnostic. Không thể biết runtime “mask anchor rỗng” bằng oracle; nếu chưa có presence detector đã kiểm chứng, ghi unverified, giữ evidence và không hard refine.

Ở biến thể soft compatibility sau này, phải giữ evidence tuyệt đối/Z trước normalization. Z nhỏ/nonfinite trả invalid; fallback và status phải rõ. Không normalize một map có tổng gần zero thành peak rồi coi quan hệ chắc chắn. Shared anchors/clauses ngoài scope không được giả định độc lập.

## 6. Policy/calibration boundary

- S0 không thay calibrated risk, ngưỡng 0.2611932834526145 hoặc active profile.
- Evidence-only pilot có thể lưu diagnostic nhưng chưa dùng nó đổi quyết định khi chưa có calibration/protocol mới.
- Nếu dùng evidence mới trong risk, freeze schema và verifier rồi fit calibrator riêng trên calibration theo family.
- Nếu đổi final MAP, event error phải dùng MAP mới; region calibration phải fit lại nếu map đổi.
- Nếu đổi Sidecar features/moments hoặc Adapter inputs, không mặc định checkpoint Adapter cũ phù hợp.
- Diagnostic failure không tự tương đương semantic/depth/occlusion causal source; REOBSERVE/ASK_USER vẫn là action nhận thức, chưa chứng minh recovery.

## 7. Điều kiện mở rộng

Ưu tiên tiếp theo là phrase–slot binding/presence evidence có kiểm chứng trên train/dev. Chỉ nghiên cứu MAP refinement sau khi có development gate định lượng và ablation regression/utility. Module runtime, thresholds và claims phải được khóa trước calibration/IID reevaluation.

Contract này chốt đầu vào, tọa độ, status và ranh giới inference/evaluator. Nó chưa chứng nhận chất lượng một verifier đã chạy, chưa tạo uncertainty probability cho quan hệ và chưa mở robot execution.
