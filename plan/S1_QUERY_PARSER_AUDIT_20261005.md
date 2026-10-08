# S1 — Parser target–relation–anchor và kiểm chứng train/dev

Ngày: **05/10/2026**. Trạng thái: **parser và audit đã chạy; nhánh anchor mới chỉ được thiết kế, chưa train**.

**Cập nhật ở bước được giao tiếp theo:** wrapper anchor và runner preparation đã triển khai, preflight kỹ thuật pass; xem [báo cáo anchor shadow](S1_ANCHOR_SHADOW_PREFLIGHT_20261005.md). Nội dung bên dưới ghi trạng thái tại lần parser audit; chưa có pilot có học ở cả hai bước.

## 1. Kết luận

Parser độc lập đã tách được target phrase, anchor phrase, predicate và span theo tokenizer hiện tại cho **1.020/1.020 câu thuộc phạm vi direct/trái/phải** trong manifest train/dev. **980 câu ngoài phạm vi trả unsupported**, không bị chuyển thành direct. Kiểm chứng span/conditioning masks qua **6.800 presentation** gồm 6.400 train có prefix augmentation và 400 dev không augmentation.

**Kết quả này xác nhận association câu–token–annotation trong grammar hiện có; chưa cải thiện một anchor prediction nào.** Baseline forward chưa nhập parser mới; không có model forward, train, fit calibration, Test-IID/OOD hoặc robot trong S1.

Thiết kế thí nghiệm tiếp theo và gate trước train nằm tại [S1 phrase-conditioned anchor experiment](S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md).

## 2. Các file chạy được

- [query_parser.py](../new/src/pcrau/query_parser.py): API prompt-only `parse_query`, `PhraseSpan`, `TypedQuery`; version `pcrau_query_parser_s1_v1`.
- [test_query_parser.py](../new/tests/test_query_parser.py): 9 nhóm kiểm tra độc lập về vai trò, span, prefix, scope và truncation.
- [audit_s1_query_parser.py](../new/scripts/audit_s1_query_parser.py): parse trước, ghi predictions ra đĩa, sau đó mới đọc registry/record/mask để hậu kiểm.
- [summary.json](../new/outputs/pcrau_s1_query_parser_20261005_r3/summary.json): kết quả cuối.
- [phrase_token_annotation.csv](../new/outputs/pcrau_s1_query_parser_20261005_r3/phrase_token_annotation.csv): bảng đủ 2.000 dòng, có phrase/span, predicate, status, annotation IDs và mask association.
- [evaluator_rows.jsonl](../new/outputs/pcrau_s1_query_parser_20261005_r3/evaluator_rows.jsonl): bảng hậu kiểm với từng check cụ thể.
- [parsed_query_predictions.jsonl](../new/outputs/pcrau_s1_query_parser_20261005_r3/parsed_query_predictions.jsonl): prompt-only outputs trước hậu kiểm; không chứa oracle IDs/masks.
- [presentation_predictions.jsonl](../new/outputs/pcrau_s1_query_parser_20261005_r3/presentation_predictions.jsonl): toàn bộ prefix/token spans đã kiểm tra.
- [provenance.json](../new/outputs/pcrau_s1_query_parser_20261005_r3/provenance.json), [validation_checks.json](../new/outputs/pcrau_s1_query_parser_20261005_r3/validation_checks.json): hashes và kiểm chứng.

Chạy lại audit bằng Python trong `.conda-roborefer`, dùng `--output` trỏ tới thư mục mới. Runner từ chối ghi đè thư mục tồn tại. Không cần GPU hoặc checkpoint inference.

## 3. Contract parser v1

Input duy nhất: `prompt: str`, `max_tokens=72`, `max_anchors=3`. Không nhận record, sample ID, ontology, RGB/depth, mask hoặc graph annotation. Audit gắn sample ID bên ngoài parser để đối chiếu.

Grammar English có kiểm soát:

```text
locate the TARGET.
locate the TARGET that is left of the ANCHOR.
locate the TARGET that is right of the ANCHOR.
```

Có thể có prefix/suffix hướng dẫn như dataset và augmentation hiện hành. Một command duy nhất; case-insensitive. Hỗ trợ khoảng trắng/hyphen trong noun phrase. Đây là parser cú pháp cho template, chưa phải parser ngôn ngữ tự nhiên tổng quát.

Output `TypedQuery`:

| Trường | Ý nghĩa |
|---|---|
| `status`, `reason` | supported / unsupported / ambiguous, lý do khi thất bại |
| `predicate` | direct / left_of / right_of; null khi thất bại |
| `reference_frame` | image cho trái/phải theo scope đã khóa; null cho direct |
| `target`, `anchors` | PhraseSpan chứa text chuẩn hóa, char và token start/end |
| `command_char_span` | Vị trí command trong full prompt |
| `total_tokens`, `max_tokens` | Cho biết giới hạn/truncation |
| `target_token_mask` | 72 Boolean, chọn đúng token target |
| `anchor_token_masks` | 3×72 Boolean, v1 chỉ slot 0 có thể hoạt động |
| `anchor_slot_mask` | 3 Boolean; direct có zero active anchor |

Tất cả span **zero-based, half-open `[start,end)`**, tính trên full prompt trước truncation; char span tham chiếu chuỗi gốc, phrase text chuẩn hóa theo `pcrau.text.words`. Token tương ứng regex `[a-z0-9]+`, vocab hashing 8192; **không phải tokenizer ngôn ngữ của backbone VLM**.

Token mask chọn đúng vị trí, không tìm theo tên phrase ở mọi vị trí. Từ “apple” ở lời dẫn không được gộp với “apple” trong target. Prefix augmentation làm dịch span, giữ nguyên danh tính phrase và token IDs đã chọn.

Ví dụ dùng API, không cần load model:

```python
from pcrau.query_parser import parse_query

query = parse_query("locate the apple that is right of the purple cube.")
assert query.supported
assert query.target.text == "apple"
assert query.anchors[0].text == "purple cube"
trace = query.to_dict()  # spans và Boolean conditioning masks
```

Trái/phải lấy hệ ảnh theo contract phạm vi S1, x tăng sang phải. Parser không suy tư thế vật/người quan sát. Direct trong annotation dùng `object_semantics`; output không giả định direct là một phép kiểm tra tọa độ nên `reference_frame=null` là chủ ý.

Ngoài grammar, depth/ternary/multiple relations, frame ngoài phạm vi, command phủ định, nhiều command, noun phrase có từ cú pháp bị cấm hoặc phrase bị cắt bởi max_tokens: trả failure và **xóa tất cả phrase/conditioning masks**. Không fallback âm thầm sang direct. Hướng dẫn dư sau phrase có thể bị cắt nhưng phrase đầy đủ vẫn hợp lệ; `total_tokens` cho biết điều đó. Trong 6.800 presentation thực tế không có truncation.

V1 chỉ hỗ trợ một anchor; capacity 3 bảo toàn hình dạng tương thích. Các câu shared-anchor/multiple-clause chưa có contract parse hoạt động; phải unsupported. Không dùng tên version để claim full noun–edge graph parser.

## 4. Dữ liệu và mẫu số

Manifest: `old/protocol/pcra_u_development_train_manifest.json`. Archive thực: `old/roborefer_dataset_v2_1_1_development_400_20260824`. Các đường dẫn lịch sử `datasets/...` được resolve bằng `ArchiveLayout`, không sửa manifest/archive.

| Nhóm | Train | Dev | Tổng |
|---|---:|---:|---:|
| Family độc lập | 320 | 80 | 400 |
| Mẫu gốc | 1.600 | 400 | 2.000 |
| Direct supported | 560 | 140 | 700 |
| Trái/phải supported | 256 | 64 | 320 |
| Supported, có hậu kiểm annotation | 816 | 204 | 1.020 |
| Unsupported ngoài scope | 784 | 196 | 980 |
| Presentation kiểm tra span | 6.400 | 400 | 6.800 |
| Supported presentation | 3.264 | 204 | 3.468 |

Scope coverage là **51%** mẫu gốc; không gọi parser accuracy 100% trên mọi loại quan hệ. Trong phạm vi, 1.020/1.020 annotation checks pass. 980 mẫu ngoài phạm vi đều có masks bất hoạt. Không hậu kiểm graph/masks của chúng trong audit này.

Augmentation dùng bốn prefix đã có: rỗng, `Please `, `In this scene, `, `Can you `. Train 6.400 presentation vẫn **320 family độc lập**, không tăng lượng cảnh độc lập. Dev giữ nguyên 400 câu.

## 5. Bảng case thật

Các span dưới đây lấy từ full prompt trong artifact, không chỉ từ đoạn command rút gọn. IDs và pixel count thuộc evaluator, không phải parser output.

| Sample / variant | Target; token span | Anchor; token span | Predicate | Annotation hậu kiểm |
|---|---|---|---|---|
| `000001__clean` | apple `[9,10)` | purple cube `[15,17)` | right_of | target `ycb_apple_01`, anchor `cube_purple_01`; anchor 3.647 pixel |
| `000001__relation_counterfactual` | purple cube `[9,11)` | apple `[16,17)` | left_of | target `cube_purple_01`, anchor `ycb_apple_01`; anchor 4.941 pixel |
| `000001__semantic_counterfactual` | pink cube `[7,9)` | không có | direct | target `cube_pink_01`, anchor IDs rỗng |
| `000392__clean` | orange cube `[9,11)` | orange `[16,17)` | left_of | candidate `cube_orange_01`, valid target rỗng; anchor `ycb_orange_01`, 0 pixel |

Sample ID đầy đủ có prefix `v211dev_family_`. Case 000392 chứng minh cần giữ phrase anchor khi mask rỗng: câu vẫn yêu cầu “orange”, parser không được dùng oracle để bỏ anchor. Việc có câu hỏi anchor không đồng nghĩa có evidence nhìn thấy anchor.

Case 000001 clean có target char span `[50,55)`, anchor `[77,88)`. Case đổi vai có target `[54,65)`, anchor `[86,91)`. Tokenization và vị trí phrase thực sự thay đổi đúng theo vai trò.

## 6. Phép hậu kiểm annotation

Sau khi ghi prompt predictions, evaluator mới:

1. So target phrase với `candidate_target_ids` qua semantic class registry và policy của generator: nhiều candidate thuộc nhóm apple/orange/lemon/mango → fruit; một semantic class → tên class; nhóm khác → object.
2. So anchor phrase theo thứ tự với `anchor_ids` và mask `object_id`.
3. So predicate/frame với spatial label và relation graph.
4. So manifest/record mask paths/hashes và hash thật; đếm pixel nhìn thấy cho active anchors.

**Graph source_ids là valid_target_ids**, không phải candidate_target_ids. ABSENT có thể có candidate được yêu cầu nhưng valid set rỗng. **Direct vẫn có một graph clause với predicate direct, target_ids rỗng và frame object_semantics**; zero anchor không có nghĩa graph annotation rỗng.

Mọi check khớp trong 1.020 câu. Active anchors có mask rỗng: train **5 mẫu / 5 family**, dev **3 mẫu / 1 family**. Giữ cùng định nghĩa S0; không gộp direct vào negative anchor.

## 7. Kiểm chứng và bảo toàn

- **9/9 nhóm test pass**: phrase lặp, đổi vai, prefix và selected token IDs, direct, các quan hệ unsupported/negation/frame, nhiều command, required phrase truncation, case/hyphen, capacity/invalid arguments.
- Span/conditioning checks pass ở 2.000 mẫu gốc và 6.800 presentation. Original train nằm trong train presentations, không cộng thành 8.800 quan sát độc lập.
- **1.353 file được kiểm tra hash trước/sau**: bundle hiện hành, source baseline, manifest/registry và các record/mask đọc trong phạm vi. Không đổi checkpoint/config/calibrator/profiles, active selection, tokenizer, model forward hoặc archive.
- Script không forward mô hình; chưa kiểm thử một module phrase-conditioned anchor vì module ấy chưa được triển khai ở nhiệm vụ này.

Lịch sử artifact: thư mục không hậu tố là lần diagnostic dừng vì evaluator giả định sai rằng direct graph phải rỗng; đã sửa evaluator theo record thật. `_r2` đã pass với parser trước bổ sung từ khóa từ chối quan hệ ngoài phạm vi. `_r3` chạy lại sau hardening và là **artifact cuối**. Không xóa/ghi đè các thư mục cũ. Predictions gốc và prefix của `_r2`/`_r3` được đối chiếu hash; nội dung dataset outputs không thay đổi.

## 8. Phần đã làm và phần chưa có bằng chứng

**Đã triển khai/đánh giá:** prompt-only parsing trong grammar hẹp; token span/mask khớp tokenizer; phrase–annotation–mask association trên toàn bộ subset supported train/dev; fail-closed trên các loại quan hệ khác của dataset và các test đã nêu.

**Mới thiết kế:** phrase-conditioned residual query, đối chứng whole-text conditioning, loss riêng cho empty visible masks, gate pilot trước train.

**Chưa chứng minh:** anchor hit cải thiện, đổi vai định vị đúng hơn, score nhận biết thiếu evidence được calibration, tác động có lợi tới target/answerability/risk, geometric verification hoặc uncertainty decomposition. Parser đúng giải quyết contract đầu vào; binding trong ảnh vẫn cần thực nghiệm có học.
