# S1 — Nhánh anchor chạy song song và kiểm chứng trước optimizer

Ngày: **05/10/2026**. Trạng thái cuối: **S1_SHADOW_PREFLIGHT_PASS**. Đã triển khai wrapper, chuẩn bị runner M1/M2 và chạy kiểm chứng kỹ thuật trên train/dev. **Không tạo optimizer cho checkpoint thật, không optimizer step, chưa pilot có học.**

Nguồn protocol: [thiết kế phrase-conditioned anchor v1](S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md), tiếp nối [parser audit](S1_QUERY_PARSER_AUDIT_20261005.md). File thiết kế v1 được giữ nguyên làm bản khóa trước train; trạng thái triển khai mới nằm trong báo cáo này.

## 1. Kết luận

**Wrapper tái tạo anchor logits cũ với sai số tối đa 0 cho cả M1/M2 ở FP32 và bfloat16, đồng thời giữ toàn bộ outputs baseline giống chính xác.** Chỉ 33.024 tham số MLP mỗi arm được phép học. Backward probe có gradient hữu hạn tới lớp cuối MLP, không tới baseline; weights/stats không thay đổi.

Gate latency engineering đạt: M1 tăng **10,42%**, M2 tăng **10,20%**, dưới trần 20% trên phép đo sidecar đã khóa. Đây là chi phí local với cached features và batch 8, chưa phải latency VLM/robot end-to-end.

**Kết quả xác nhận đường tính và ranh giới kỹ thuật, chưa chứng minh binding cải thiện.** Ở khởi tạo zero, map mới bằng map cũ theo thiết kế; chưa có kết quả M1/M2 sau học, chưa geometric verification, calibration hoặc uncertainty mới.

## 2. Code và artifact cuối

| File | Vai trò |
|---|---|
| [anchor_shadow.py](../new/src/pcrau/anchor_shadow.py) | Wrapper AnchorShadow; capture T/q_old/F bằng temporary hooks; M1/M2 residual queries; baseline output riêng |
| [anchor_shadow_experiment.py](../new/src/pcrau/anchor_shadow_experiment.py) | Protocol cố định, prompt-only observable loader, family/prefix batching, loss, runner API và selection/early-stop bookkeeping |
| [preflight_anchor_shadow.py](../new/scripts/preflight_anchor_shadow.py) | Chạy frozen checks, backward-only probe và latency; không có CLI tự chạy train |
| [test_anchor_shadow.py](../new/tests/test_anchor_shadow.py) | 10 nhóm test synthetic không đọc archive/test data và không optimizer step |
| [summary.json](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/summary.json) | Kết quả cuối, gate kỹ thuật và latency |
| [identity_checks.jsonl](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/identity_checks.jsonl) | So sánh từng output tensor và hash output theo batch/mode |
| [inference_predictions.jsonl](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/inference_predictions.jsonl) | Prompt/parsed spans, trạng thái, MAP/scores thật của M0/M1/M2 ở step zero |
| [gradient_probe.json](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/gradient_probe.json) | Loss và gradient norms trên train probe |
| [pooling_validation.json](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/pooling_validation.json) | Kiểm tra độc lập tensor thực sự đi vào MLP theo phrase span/valid whole text, 16 presentation train/dev |
| [locked_protocol.json](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/locked_protocol.json) | Protocol/hashes/trainable names trước mọi backward |
| [family_batch_plan.json](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/family_batch_plan.json) | Lịch epoch 0 cho vòng train tương lai, chưa chạy epoch |
| [initial_mlp.safetensors](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/initial_mlp.safetensors) | Weights khởi tạo chung M1/M2, chưa được train |
| [latency.json](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/latency.json), [provenance.json](../new/outputs/pcrau_s1_anchor_shadow_preflight_20261005_r3/provenance.json) | Measurements gốc, runtime, input/source hashes |

Artifact không hậu tố là diagnostic dừng ở lỗi stride; `_r2` pass sau sửa stride. `_r3` chạy lại sau bổ sung các guard của runner và là **artifact cuối**. Không ghi đè/xóa các bản trước. Các predictions `_r2`/`_r3` được đối chiếu hash, không đổi.

## 3. Contract wrapper và API

`AnchorShadow(baseline, conditioning="phrase" | "whole_text", seed=24082026)` nhận đúng `PCRAUTargetV2` hidden_dim 128. Constructor đóng băng toàn bộ baseline, đặt eval và xóa gradients baseline cũ. Khi gọi `.train()` wrapper, MLP được chuyển train nhưng mọi module baseline vẫn eval. MLP bắt buộc FP32; có thể gọi head đóng băng với AMP.

Input forward: **đúng chín MODEL_INPUT_KEYS cũ** và một danh sách full prompts tách riêng. Không nhận supervision, object IDs hoặc graph. Wrapper kiểm tra token IDs/mask, relation IDs/mask và activation mask trong batch khớp lại chính prompt đó; truyền span của câu khác sẽ bị từ chối.

```python
from pcrau.anchor_shadow import AnchorShadow

# baseline đã load đúng selected checkpoint; batch chỉ chứa observable inputs.
m1 = AnchorShadow(baseline, conditioning="phrase")
m2 = AnchorShadow(baseline, conditioning="whole_text")
capture = m1.capture(batch, prompts)
map_m1 = m1.predict(capture)
map_m2 = m2.predict(capture)  # cùng một frozen capture và cùng baseline instance
original = capture.baseline  # target/answerability/evidence cũ
```

`m1(batch, prompts)` trả dict có `baseline`, `anchor_logits_shadow`, `active_slots`, `queries`, `scope_status`, `conditioning`, `version`. Shadow tensor không ghi đè `baseline["anchor_logits"]`. Baseline graph/Adapter luôn sử dụng anchor cũ; không có prediction mới đi ngược vào forward cũ.

Capture:

- T: contextual tokens của Sidecar encoder, `[B,72,128]`.
- q_old: anchor queries sau cross-attention + LayerNorm, `[B,3,128]`.
- F: fused visual features, `[B,24,32,128]` từ RGB/depth cached features trước projector.
- Phrase masks/active slots: parser prompt-only, không oracle.
- Head AMP mode: được ghi cùng capture để replay head đúng dtype ngay cả khi predict được gọi ở ngoài autocast context ban đầu.

T/q/F detached, không gradient baseline. Capture được thực hiện với no_grad và tạm tắt inference_mode để tensor có thể được autograd save trong bước residual head. Không mutate captured tensors. Hooks chỉ sống trong một call, được dọn trong finally cả khi call lỗi. Wrapper/cùng baseline instance chỉ hỗ trợ sử dụng tuần tự, không concurrent capture.

MLP FP32: Linear128→128, GELU, Linear128→128; W2 weight/bias zero. Query mới cộng residual đã cast về dtype q_old. Head weights đóng băng nhưng phép gọi head nằm ngoài no_grad khi cần backward.

`DIRECT_BYPASS` và `PARSE_UNSUPPORTED` giữ nguyên mọi map cũ; slots không active cũng giữ nguyên từng tensor value bằng torch.where. Trạng thái `SUPPORTED_SHADOW` chỉ mô tả scope có nhánh thử nghiệm, **không phải identity/geometry verified**.

## 4. Phạm vi và mẫu số thật

Giữ manifest development/archive và selected Adapter bundle. Không đọc calibration samples/Test-IID/OOD; calibrator/profiles chỉ hash-check, không apply risk hoặc fit.

| Nhóm kiểm chứng | Mẫu gốc | Family | Presentation mỗi precision |
|---|---:|---:|---:|
| Horizontal train, đủ bốn prefix | 256 | 64 | 1.024 |
| Toàn bộ dev | 400 | 80 | 400 |
| Tổng | 656 | 144 | 1.424 |

Chạy hai precision: FP32 và AMP bfloat16 → **2.848 lượt kiểm chứng**, vẫn chỉ 656 mẫu gốc/144 family trong phạm vi trên. Không gọi prefix/precision passes là dữ liệu độc lập mới. Đây không phải wrapper audit trên toàn 1.600 train; parser audit trước đã đọc toàn train/dev.

Dev trong mỗi precision: **64 horizontal**, **140 direct bypass**, **196 unsupported bypass**. Train preflight chỉ horizontal nên toàn 1.024 presentation có slot 0 active. Dev horizontal thuộc 16 family như S0; các mẫu còn lại dùng kiểm tra regression/bypass.

Predictions của cả hai precision được ghi đầy đủ ra đĩa **trước khi đọc mask train** cho loss/gradient probe. Mask không đi vào model/wrapper.

## 5. Gate kỹ thuật và kết quả

| Kiểm tra | Kết quả |
|---|---|
| MLP parameter count | 33.024 mỗi arm; chỉ residual.0/2 weight/bias trainable |
| Initialization M1/M2 | State digest giống nhau: `e808b0e0b3a40dc4d26210f0a5b9bce08c4074a5da18f65e4ebc8698593e7057` |
| Step-zero identity FP32 | M1 max abs error **0**, M2 **0**; gate ≤1e-6 |
| Step-zero identity bfloat16 | M1 max abs error **0**, M2 **0** |
| Baseline regression | Tất cả output keys/tensors giống chính xác giữa plain forward và wrapper capture, cả train/dev và hai precision |
| Baseline weights/stats | State digest trước/sau giống nhau; không baseline gradient |
| Direct/unsupported/inactive | Bypass đúng; synthetic test với residual khác zero vẫn giữ map inactive và downstream baseline |
| Boundary | Oracle key, prompt–token mismatch bị từ chối |
| Hook lifecycle | Dọn sau mỗi call và injected failure; không retained hooks |
| Frozen tensors/autograd | Capture dưới outer inference_mode vẫn tạo tensor thường; backward hoạt động |
| Runner setup | 16 batch/epoch, 4 family/batch, 64 presentation/batch; cùng seed/order cho M1/M2 |
| Unit checks | **10/10 nhóm shadow + 9/9 nhóm parser pass** |

So output bao gồm target/interior/anchor, relation_edge, answerability/source, graph embedding, moments, fusion gate, modality-dropout scalar và exported answerability detail/evidence. Không chỉ so argmax hoặc labels. Equality được kiểm tra **trong cùng precision/batch**, không yêu cầu outputs FP32 bằng outputs bfloat16.

### Kiểm tra đúng noun tokens đi vào MLP

Một kiểm tra độc lập thu input của Linear đầu MLP bằng hook, so với mean trực tiếp trên `T[anchor.token_start:anchor.token_end]` cho M1 và mean trên valid token_mask cho M2. Dùng 8 train presentation có prefix và 8 dev horizontal, không masks/annotation IDs.

**16/16 pass**; M1 pooling max error **0**, M2 **2,384×10⁻⁷** so phép mean độc lập. Đây là sai số reduction FP32 của phép pooling, không phải anchor-logit identity error. Các span và độ khác giữa phrase/whole pooling có trong pooling_validation.json. Kiểm tra này được chạy bổ sung ngoài CLI preflight chính.

## 6. Lỗi stride được phát hiện và sửa

Lần preflight đầu có FP32 shadow-logit max error **3,814697×10⁻⁶**, vượt gate mặc dù delta_q bằng zero. Baseline outputs vẫn exact; lỗi nằm ở lần replay anchor head.

Query cũ là view của 7 slots, stride `(896,128,1)` cho `[B,3,128]`. Clone/cộng residual thông thường tạo tensor contiguous stride `(384,128,1)`. `Linear` trên hai layout chọn đường tính khác nhau, gây rounding khác. Probe độc lập xác nhận:

| Query replay | Max abs error |
|---|---:|
| Original layout | 0 |
| Contiguous clone / contiguous sum | 3,814697×10⁻⁶ |
| Sum copy sang original stride | 0 |

Wrapper hiện giữ detached view q_old; q_new được copy vào `empty_strided` theo đúng stride cũ. Copy có autograd nên không chặn gradient MLP. Thêm test riêng về stride và backward. **Không nới gate hoặc lấy logits cũ thay cho active shadow logits để ép pass.**

## 7. Loss và backward probe thật

Loss riêng giữ visible BCE+Dice hiện có và thêm zero-map BCE cho active slots có mask rỗng, trọng số **0,25**. Direct/inactive không thành negative. Loss FP32, không location-mass, không target/answerability loss.

Sau khi inference artifact đã ghi, đọc 256 train horizontal anchor masks: **251 nonempty, 5 empty**. Backward probe chọn hai mẫu theo thứ tự sample ID: `v211dev_family_000018__clean` visible và `v211dev_family_000023__relation_counterfactual` empty; lấy đủ bốn prefix → 8 presentation, 4 visible + 4 empty. Đây là probe gradient, không một train epoch.

| Đại lượng | M1 | M2 |
|---|---:|---:|
| Total loss tại initialization | 0,322506 | 0,322506 |
| Visible BCE+Dice | 0,311656 | 0,311656 |
| Empty BCE | 0,043398 | 0,043398 |
| W2 weight gradient norm | 0,067187 | 0,020713 |
| W2 bias gradient norm | 0,016320 | 0,016320 |
| W1 gradient norm | 0 | 0 |
| Baseline gradients / optimizer steps | 0 / 0 | 0 / 0 |

W1 gradient bằng zero ở backward đầu tiên là **hệ quả đúng của W2 zero**, không phải đứt graph. Synthetic test đặt W2 khác zero và xác nhận gradient tới W1; không optimizer step. Loss M1/M2 bằng nhau ở initialization cũng là kết quả dự kiến vì logits bằng nhau; gradient khác nhau chưa chứng minh arm nào tốt hơn.

Weights MLP/checkpoint baseline không thay đổi sau probe; gradients đã được xóa. Initial artifact vẫn là initialization, không trained checkpoint.

## 8. Runner đã chuẩn bị đến đâu?

`ShadowPilotRunner` có:

- Constructor chưa tạo optimizer; `start_optimizer` yêu cầu receipt preflight pass và đối chiếu digest initialization/baseline thực tế.
- AdamW chỉ nhận MLP params: lr 3e-4, weight decay 1e-3; clip norm 1,0 với nonfinite error.
- `run_epoch` dùng cùng family batches cho hai arms; callback cấp frozen observable capture và supervision masks riêng.
- Max 15 epoch, không scheduler; buộc ghi dev selection sau mỗi epoch trước epoch tiếp theo.
- Selection tuple: visible hits, matched swap both-hits, rồi negative mean sigmoid-max nhỏ hơn; ties giữ epoch sớm hơn. Patience 5; duplicate/nonfinite dev selection bị từ chối.
- Kiểm tra baseline state digest trước/sau epoch; baseline vẫn eval, chỉ MLP có gradients.

**Đã kiểm tra preparation, loss/backward và selection bookkeeping; chưa gọi optimizer creation hợp lệ/run_epoch để huấn luyện.** Full pilot orchestration, evaluate/save best checkpoint và báo cáo binding/ablation là nhiệm vụ tiếp theo, không được gọi đã đánh giá chỉ vì API có code.

Protocol Markdown v1 không đổi; SHA-256 **`c908d287790baa41f7e63656dc99a4a278d55b3e28e8d958dddc1c645e810932`** được ghi trước backward. Các gate anchor hit ≥49/61, swap ≥8/15, FOUND both ≥12/16 và missing-support guard vẫn giữ nguyên, **chưa được chấm như kết quả nâng cấp**.

## 9. Chi phí và bảo toàn

RTX 2000 Ada, Python 3.10.14, PyTorch 2.13.0+cu130. Batch 8 horizontal train presentations, AMP bfloat16; warm-up 20 và 100 measurement/arm, xoay thứ tự M0/M1/M2 giữa rounds, synchronize CUDA khi đo. Không chạy process GPU khác đồng thời phép timing này.

| Arm | Median ms/batch | P95 ms/batch | Tăng median so M0 |
|---|---:|---:|---:|
| M0, wrapper capture bật / branch tắt | 5,054 | 5,227 | — |
| M1 phrase | 5,581 | 5,804 | **10,42%** |
| M2 whole text | 5,570 | 5,779 | **10,20%** |

Parser, token matching/capture, frozen sidecar và residual head nằm trong timing. M0 dùng cùng wrapper capture setup. Không disk feature loading, backbone forward hoặc robot. Peak CUDA allocated trong measurement là **241.916.416 byte** cho process có các tensors/models resident; chưa đo memory tăng riêng từng arm hoặc optimizer state khi train.

**732 file được kiểm tra hash không đổi** trong preflight: bundle hiện hành, baseline/new source, protocol, manifest/index, cached features và train masks đã đọc. Snapshot baseline source/bundle trước triển khai cũng được đối chiếu cuối nhiệm vụ. Không sửa model.py/text.py/dataset.py/losses.py, active profile, archive/dataset, calibration hoặc LaTeX; không reset/commit/push.

## 10. Bước kế tiếp và giới hạn claim

Gate kỹ thuật và latency pass, nên đã có cơ sở triển khai orchestration và chạy **pilot M1/M2 có học** theo protocol v1 khi được giao. Cần lưu selected residual checkpoint, paired anchor/swap metrics, từng ca zero-mask và regressions, rồi chấm gate trước khi mở verifier.

Chưa được claim anchor tốt hơn, hiểu quan hệ hình học, phát hiện thiếu anchor đã generalize hoặc uncertainty đã calibration. Branch chưa nối vào baseline graph/Adapter, nên answerability/risk path vẫn cũ; không có cải thiện risk từ các kết quả step-zero này. Gate kỹ thuật pass không thay thế gate binding và claim khoa học sau train.
