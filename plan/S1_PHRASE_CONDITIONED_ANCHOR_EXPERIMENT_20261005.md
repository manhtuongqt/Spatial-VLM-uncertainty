# S1 — Thiết kế thí nghiệm phrase-conditioned anchor, khóa trước train

Ngày: **05/10/2026**. Protocol v1: **đã ghi thiết kế và gate; chưa triển khai nhánh model, chưa train**. Nối tiếp [S0 audit](S0_ANCHOR_BINDING_AUDIT_20261005.md) và [S1 parser audit](S1_QUERY_PARSER_AUDIT_20261005.md).

## 1. Mục tiêu và giả thuyết có thể bác bỏ

**Mục tiêu vòng đầu:** kiểm tra việc đưa token của đúng anchor noun phrase vào query có giúp learned anchor head tìm đúng vật và đổi vai đúng trên cùng cảnh hay không. Đây là thí nghiệm binding độc lập, chưa nối anchor mới vào graph/Adapter hoặc dùng hình học sửa target MAP.

H1: phrase conditioning cải thiện anchor hit so với head hiện tại. H2: lợi ích đến từ chọn đúng noun span, vượt một nhánh cùng số tham số dùng whole-text pooling. H3: giám sát zero-map trên active anchors không có pixel nhìn thấy giảm evidence giả mà không làm mất binding visible anchors.

H3 không được gán hết cho phrase conditioning: M1 và M2 phải có cùng loss empty-mask; nếu cần kết luận riêng về loss, phải có vòng ablation sau. Mô hình cũ đã được train với loss khác nên M1 thắng M0 chưa cô lập được nguyên nhân.

## 2. Baseline và nguồn dữ liệu khóa

Baseline M0: **P-CRA-U hiện hành = V2 đóng băng + detail_cost Adapter + logistic33**. Bundle lấy đúng `new/outputs/active_experimental_profile.json`; không thay support scorer hoặc V3.

| Artifact | SHA-256 |
|---|---|
| checkpoint | `5a326a8040bfbabfadf3f57fff51fd90be3845438c9a3a8c6ee41bdd1e0fa045` |
| config | `b214670c6c4a679f755a9953e15bae4bbb71ee5eea1980fa8e3d64030d79f85b` |
| calibrator | `19e0c35424144cf753109978b9d08ddaa16edcadbe0c1dc1515103fb88b8de7d` |
| profiles | `df39fa55843d657111a2c0bee2fee68aad1f2c9ada4c2fc257527eb420f6c9ff` |

Manifest gốc giữ nguyên; parser S1 chọn phạm vi theo prompt. Không chọn subset bằng việc head cũ hit/miss. Association/masks chỉ dùng làm train supervision hoặc dev evaluation.

| Pilot binding | Train | Dev |
|---|---:|---:|
| Family trái/phải | 64 | 16 |
| Câu horizontal | 256 | 64 |
| Active anchor có pixel | 251 | 61 |
| Active anchor zero pixel | 5 / 5 family | 3 / 1 family |
| Presentation nếu dùng 4 prefix train hiện có | 1.024 | 64, không augment |

Direct không có anchor: bypass, không biến thành negative mask để train. Các loại depth/ternary ngoài grammar cũng bypass. Các câu khác vẫn dùng để kiểm tra regression của baseline trên toàn 400 dev; không mở rộng objective pilot.

**Mẫu số visible-anchor khác answerability FOUND.** Câu ABSENT có thể thiếu target nhưng anchor vẫn nhìn thấy; không dùng nhãn ABSENT làm nhãn anchor absent. Zero pixel là absence of visible support trong ảnh này, không nhất thiết absence of physical object.

## 3. Nhánh model đề xuất

### 3.1. Những gì code hiện tại thực sự có

`ContextualQueryGraph` có hashed token embedding + position embedding + TransformerEncoder. Anchor query là learned slot, cross-attend tới toàn bộ text rồi LayerNorm. `SlotSpatialHead` dùng feature projection, query projection và query bias để xuất logits. Không có noun span mask trong forward hiện tại.

`PCRAUTargetV2` tạo fused features từ RGB/depth feature trước projector 24×32×1152 và relation context; hidden dimension 128. Anchor maps hiện tại đi tiếp vào node moments/graph và global heads. Vì thế thay trực tiếp anchor maps trong forward sẽ đổi phân phối evidence/Adapter, dù target head không đổi.

### 3.2. Residual query trong nhánh thí nghiệm riêng

Thu các tensor frozen của baseline theo observable input:

```text
T          : [B,72,128]     contextual text tokens
q_old      : [B,3,128]      anchor queries sau cross-attention + LayerNorm
F          : [B,24,32,128]  fused spatial features
m_anchor   : [B,3,72]      từ parser prompt-only
active     : [B,3]         từ parser prompt-only
```

Với slot 0 thuộc câu horizontal:

```text
h_phrase = sum(T * m_anchor) / sum(m_anchor)
delta_q  = W2(GELU(W1(h_phrase)))
q_new    = q_old + cast(delta_q, dtype(q_old))
L_new    = frozen_anchor_head(F, q_new)
```

MLP 128→128→128 với bias: **33.024 tham số trainable**. W2 weight/bias khởi tạo zero; W1 dùng cùng seed/initialization giữa đối chứng. Khởi tạo phải tái tạo anchor logits cũ trước bước optimizer đầu tiên. Nhánh mới không thêm dropout ở vòng này; baseline giữ eval, không modality dropout hoặc MC.

Phrase token đã contextualized theo toàn câu; **không gọi h_phrase là semantic embedding độc lập hoàn toàn với target/relation**. Thí nghiệm kiểm tra việc chọn đúng span có đóng góp thêm hay không.

### 3.3. Ranh giới kỹ thuật

- Tạo wrapper/module opt-in riêng sau khi được giao bước có học; không đổi default `PCRAUTargetV2.forward` hoặc checkpoint schema cũ.
- Có thể thu T bằng encoder forward hook, q_old bằng query_graph hook, F bằng fusion hook trong wrapper; thu tensor detach, tháo hook sau call, không mutate outputs. Phải kiểm tra hook đúng layer/dtype/shape và không rò tensor qua lần gọi. Không thay source baseline chỉ để export tensor.
- Baseline chạy trong eval + no_grad, mọi weights/stats đóng băng. **Phép gọi frozen anchor head cho q_new phải ở ngoài no_grad/inference_mode** để gradient đi về MLP. Không cập nhật head weights. Frozen tensors được chuẩn bị sao cho dùng an toàn trong autograd; không giữ inference tensors không thể save cho backward.
- Loss train được nhận mask supervision ở runner riêng; observable branch chỉ nhận tensor features/text và prompt-derived masks. Giữ `MODEL_INPUT_KEYS` cũ; không nhét oracle IDs, masks, valid count hoặc graph vào forward.
- Logits mới chỉ xuất dưới tên riêng, ví dụ `anchor_logits_phrase`; baseline vẫn xuất anchor/global/Adapter evidence cũ. Không feed logits mới vào node_projection/graph_projection hoặc answerability Adapter ở pilot.
- Direct/unsupported bypass phải có trace và không có phrase prediction được gọi verified. Hình dạng 3 slot giữ nguyên, pilot chỉ train slot 0 active.

Shadow mode cho phép đo binding mới mà toàn bộ baseline target/answerability/source/risk path vẫn dùng predictions cũ. **Baseline output bất biến ở đây là gate bảo toàn triển khai, chưa chứng minh model sau tích hợp không regression.**

## 4. Đối chứng bắt buộc

| Mã | Conditioning | Trainable | Loss |
|---|---|---:|---|
| M0 | learned anchor slot hiện tại | 0 | không train lại |
| M1 | residual từ đúng anchor phrase tokens | 33.024 | visible BCE+Dice, empty zero-map BCE |
| M2 | residual từ mean toàn bộ valid text tokens | 33.024 | giống hệt M1 |

M1/M2: cùng baseline tensors, train family/order/prefix, MLP initialization, optimizer, số epoch tối đa và checkpoint selection rule. Không thêm capacity cho M1 so M2. M2 dùng toàn bộ token_mask thực, không cả padding.

Lưu predicted anchor của cả hai câu trong các cặp clean/relation-counterfactual cùng cache và cùng nhìn thấy anchors. **Gate dùng 15 cặp dev**, baseline đúng cả hai 6/15, same peak 2/15. Train matched subset 51 cặp, đúng cả hai 29/51; không dùng con số 35/59 của subset rộng hơn thay cho nó.

Diagnostic tùy chọn trong evaluation, không phải selection metric: cùng câu/hình, đổi riêng conditioning span sang target phrase để xem prediction có đáp ứng semantic query hay chỉ vị trí/predicate. Lưu can thiệp rõ; không coi câu bị can thiệp có annotation target/anchor mới đã được xác nhận.

## 5. Loss và handling thiếu evidence

Dataset hiện tại đặt `anchor_supervision_mask = small_mask.sum()>0`; `compute_losses` chỉ gọi anchor BCE+Dice trên slots có mask đó. Vì vậy **active anchor zero-mask không có direct anchor heatmap loss**. Các loss khác có thể tác động gián tiếp qua weights chung; S0 chưa cô lập được nguyên nhân mọi peak sai.

Pilot mới:

```text
V = parser active AND annotation anchor mask nonempty
E = parser active AND annotation anchor mask empty
L_visible = masked_heatmap_loss(L_new, area_resized_mask, V)  # code BCE+Dice hiện có
L_empty   = mean BCEWithLogits(L_new[E], zeros)
L_total   = L_visible + 0.25 * L_empty
```

Nếu một nhóm không có trong batch, term của nhóm ấy bằng zero; không NaN. Mean BCE tính trên mọi pixel và mẫu empty trong batch. Visible loss giữ positive-weight clipping [1,20] và Dice hiện tại; không thêm location-mass. Không Dice trên empty, không loss target/answerability/source ở shadow branch.

Hệ số **0,25 được khóa cho pilot**, là lựa chọn kỹ thuật có thể sai, không phải giá trị tối ưu suy ra từ dữ liệu. M1/M2 dùng cùng hệ số. Không quét thêm trọng số sau xem dev rồi vẫn gọi protocol v1.

Lưu raw logits, sigmoid-max và sigmoid mean trước mọi spatial softmax. Spatial softmax vẫn phải có peak dù input hoàn toàn thiếu evidence; **peak/entropy thấp không tự xác nhận anchor tồn tại hoặc đúng danh từ**. Zero-map loss chỉ tạo negative evidence được giám sát; chưa phải calibrated existence probability.

Vòng này chưa thêm presence head: chỉ có 5 family train negative và 1 family dev negative, không đủ để claim một classifier missing-anchor đã generalize. Báo từng ca và không gộp answerability state với anchor visibility.

## 6. Protocol huấn luyện dự kiến, cố định cho vòng đầu

**Đây là protocol cho lần chạy tương lai; tài liệu không tự khởi động train.**

- Seed **24082026** cho M1/M2; seed này chỉ một pilot, chưa phải robustness nhiều seed.
- Dùng đủ 256 câu horizontal train với bốn prefix đã có = 1.024 presentation / 64 family. Không tạo cảnh/annotation mới. Dev không augment.
- Batch **4 family**, giữ đủ 4 câu horizontal và 4 prefix của mỗi family trong batch (64 presentation nếu đầy batch); shuffle family theo seed, không mix dev vào train.
- Frozen feature extraction eval/no_grad; MLP train FP32, loss FP32. Dùng dtype/cast consistent khi gọi frozen spatial head; ghi AMP/runtime thực tế nếu dùng.
- Optimizer AdamW, learning rate **3e-4**, weight decay **1e-3**, gradient clipping norm **1,0**; không scheduler.
- Tối đa **15 epoch**, mỗi epoch evaluate trên cùng 64 dev horizontal; early stop sau **5 epoch** không cải thiện tuple selection.
- Chọn checkpoint theo tuple: (visible full-pixel anchor hits, both-hit matched swap pairs, negative dev mean sigmoid-max nhỏ hơn); nếu hòa, chọn epoch sớm hơn. Gate bên dưới chấm **checkpoint đã chọn**, không chọn lại checkpoint chỉ để cứu gate.
- M1/M2 có cùng rule và trần ngân sách. Model selection trên dev là hợp lệ nhưng tăng mức độ đã quan sát dev; báo rõ mọi run, kể cả thất bại.

Các giá trị trên là thiết kế prespecified của vòng pilot, không có bằng chứng tối ưu. Nếu cần đổi architecture/loss/hyperparameters sau kết quả âm, tạo protocol v2 với lý do và lưu kết quả v1; không sửa gate hồi tố.

## 7. Gate khóa trước train

Nguồn baseline: artifact S0 `_r2`. Gate điểm được chọn cho **pilot engineering**, không phải ngưỡng phổ quát, statistical guarantee hoặc deployment approval. Cần report paired deltas, regressions và family bootstrap CI 95% (5.000 resamples, seed 24082026); CI rộng phải nói rõ. Mỗi lần bootstrap resample family, giữ các variant/cặp của family cùng nhau.

| Gate | Mức khóa v1 | Vai trò |
|---|---|---|
| Input isolation | Không oracle trong observable branch, split/hash đúng | Bắt buộc trước optimizer |
| Step-zero identity | Anchor logits mới tái tạo M0; FP32 max abs error ≤1e-6; cùng dtype/AMP phải kiểm tra và báo sai số | Bắt buộc trước optimizer; không tự nới khi fail |
| Freeze | Chỉ 33.024 MLP params trainable, hashes/stats baseline bất biến | Bắt buộc toàn run |
| Shadow regression | Baseline outputs target/interior/anchor/answerability/source/edge và exported evidence giống chính xác khi wrapper bật/tắt trên 400 dev | Bắt buộc; compare cùng môi trường/batch/dtype |
| Visible anchor hit M1 | **≥49/61 = 80,33%**, baseline 44/61 | Lợi ích binding ≥5 ca, +8,20 pp |
| Matched swap pair M1 | **≥8/15**, baseline 6/15 | Cải thiện đổi vai ≥2 cặp |
| FOUND target+anchor M1 | **≥12/16**, baseline 11/16 | Giữ lợi ích ở ca hợp lệ, target baseline giữ nguyên |
| Missing visible support guard | Mỗi 3 mẫu empty dev có sigmoid-max **không tăng quá 1e-6** so M0 | Guard descriptive; cùng một family, không chứng minh missing detection |
| Phrase-specific evidence | M1 hơn M2 **≥2 visible hits hoặc ≥2 both-hit swap pairs**, đồng thời không thấp hơn ở metric còn lại | Chỉ khi đạt mới claim lợi ích của chọn noun span |
| Chi phí branch | Median sidecar-only latency tăng **≤20%** so M0, cùng batch, thiết bị, warm-up 20 và đo 100 lần | Gate feasibility pilot; báo cả ms và memory |

Latency bao gồm parser và shadow tensor collection/MLP/head, không tính lại backbone/cache capture. Đồng bộ CUDA khi đo; M0 chạy cùng wrapper setup nhưng nhánh tắt. Không gọi đây là end-to-end robot/VLM latency. Nếu overhead không đạt, báo bottleneck; chưa promote integration dù binding có tiến bộ.

Baseline dev zero-mask sigmoid-max min/median/max: **0,479991 / 0,725649 / 0,848972**; per-case values phải lấy S0 evaluator rows, không chỉ so aggregate vì có thể che regression. Thống kê trên 3 mẫu thuộc một family không tạo đủ bằng chứng chọn threshold presence.

**Nếu gate kỹ thuật fail:** không train hoặc dừng run, sửa lỗi contract. **Nếu gate binding fail:** báo kết quả âm; chưa geometric hard refinement. **Nếu M1 thắng M0 nhưng không thắng M2:** có thể conditioning/capacity/loss có ích, chưa chứng minh noun binding là nguyên nhân. **Nếu thiếu evidence guard fail:** không gọi anchor branch đủ cho verifier thiếu-evidence.

## 8. Artifact phải lưu cho vòng có học

Folder biến thể riêng, không ghi vào active bundle. Lưu protocol này và hash trước optimizer, manifest/sample-family lists, parser/version/hash, initialization hash, frozen bundle hashes, config/seed/runtime, trainable parameter names/counts, epoch losses, checkpoint selection trace, selected MLP checkpoint, train/dev shadow predictions cho M0/M1/M2 và matched-pair metrics.

Prediction trace: original prompt, parsed phrases/spans, status, active slots, baseline/new anchor logits/MAP/scores, baseline target/answerability. Oracle mask IDs/visible counts và hit flags nằm trong evaluator artifact riêng. Chỉ minh họa case thật từ predictions; không dùng oracle để sửa map/score.

Báo theo variant/state/semantic phrase, per-family delta, lỗi trước/sau, confidently wrong counts, 3 empty dev cases và latency. Không chỉ báo tổng accuracy; đặc biệt xem M1 có kéo anchor theo target không.

## 9. Điều gì được phép kết luận sau pilot?

Nếu gate đạt, có thể kết luận: **trong grammar image-frame trái/phải trên development đã quan sát, anchor query dùng đúng noun span cải thiện binding theo đối chứng đã khóa**. Đây là cơ sở thử verifier evidence-only tiếp theo.

Chưa được kết luận explicit spatial reasoning hoàn chỉnh: chưa tính predicate hình học trên predictions, chưa có trace thiếu/conflict evidence tham gia quyết định. Chưa có uncertainty mới đã calibration, chưa có epistemic/aleatoric decomposition. Shadow branch giữ answerability/risk path cũ nên **không thể tạo ra cải thiện selective risk từ việc chỉ chạy branch này**.

Sau pilot thành công mới thiết kế vòng nối predictions vào verifier, rồi nếu tích hợp vào graph/Adapter phải xử lý distribution shift, freeze và calibration đúng event cuối. Dataset/IID cũ vẫn giữ nguyên, IID là reevaluation đã quan sát. Test-OOD, capture, MC và robot tiếp tục ngoài scope.

## 10. Điểm bàn giao hiện tại

**Đã xong:** parser chạy được, bảng phrase–token–annotation, audit train/dev và protocol/gates v1 trước train. **Chưa xong và không nằm trong lần chạy này:** wrapper/MLP/runner có học, checkpoint biến thể, kết quả M1/M2, verifier hoặc calibration mới. Bước được giao tiếp theo có thể triển khai wrapper và kiểm tra step-zero/freeze, rồi thực hiện pilot có học theo protocol nếu người dùng yêu cầu.
