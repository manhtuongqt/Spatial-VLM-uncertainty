# Năm nguồn được tính bằng gì? Cơ chế MC Dropout đã triển khai

Ngày 07/10/2026. **Đã triển khai và chạy một nhánh MC thật cho head năm nguồn
trên toàn bộ 1.600 mẫu train và 400 mẫu dev, T=20.** Cơ chế này bổ sung model averaging và
model disagreement cho từng nhãn nguồn, giữ nguyên các model/profile/artifact
trước đó. Không diễn giải hồi tố rằng kết quả source sigmoid cũ đã dùng MC.

## 1. Câu trả lời trực tiếp cho thầy

> Năm nguồn semantic, relation, spatial, depth và occlusion được mô hình hóa
> bằng một head đa nhãn, học từ features RGB-D/ngôn ngữ và nhãn nguồn trong
> dataset. Ở nhánh MC mới, em giữ dropout đã học trong source head hoạt động
> khi suy luận và chạy 20 lượt. Trung bình các sigmoid outputs cho năm source
> scores ước lượng bằng Monte Carlo. Độ bất định của dự đoán từng nhãn được
> đo bằng predictive entropy và mutual information giữa nhãn với các subnet
> được lấy mẫu. Vì vậy head và supervision xác định ý nghĩa năm nguồn; MC
> Dropout cung cấp phép lấy mẫu để ước lượng scores và bất đồng của chúng.

Khi cần làm rõ phần đã nói trước đó:

> Bản ban đầu xuất source scores từ một lần forward sigmoid. Em đã bổ sung
> và kiểm tra nhánh MC riêng; các số MC thuộc nhánh mới này. Em không gọi
> source probability và model disagreement là cùng một đại lượng.

**Cách gọi khoa học:** “Ước lượng source scores bằng MC model averaging và
đo bất định của dự đoán năm nhãn nguồn bằng MC Dropout.” Chưa gọi đây là
phân rã nhân quả uncertainty tổng thành năm nguồn độc lập.

## 2. Phần học xác định năm nguồn

[Source head](../new/src/pcrau/model.py) đã có cấu trúc:

`global_feature768D → LayerNorm → Linear → GELU → Dropout(p=0,1) → Linear5`.

Global feature ghép RGB-D fused features, text/relation context, predicted graph
và thumbnail features. Đây là representation từ baseline Sidecar; không phải
hidden state ngôn ngữ cuối của backbone. Anchor P1/verifier hiện không quay
vào source head, nên MC ở đây không bao phủ nhánh geometry/P1 mới.

Head được học bằng multi-label weighted BCE và family ranking loss. Các nhãn:

| Output | Supervision hiện có | Điều được ước lượng |
|---|---|---|
| semantic | source_labels.semantic của archive | Score về nhãn nguồn semantic |
| relation | source_labels.relation | Score về nhãn nguồn relation |
| spatial | weak label: answerability=AMBIGUOUS | Score về proxy ambiguity, không toàn bộ spatial uncertainty |
| depth | source_labels.depth | Score về nhãn nguồn depth |
| occlusion | source_labels.occlusion | Score về nhãn nguồn occlusion |

Nhiều nguồn có thể cùng có mặt, nên dùng 5 sigmoid riêng, không softmax ép tổng
bằng 1. **MC không đặt tên năm nguồn**: ý nghĩa tên đến từ supervision. Weighted
BCE/ranking cũng không bảo đảm sigmoid đã calibrated; gọi là model scores
hoặc model probabilities, không mặc định là xác suất nguồn thật đã hiệu chuẩn.

## 3. Phép tính MC thực sự

Với representation cố định h(x), mỗi mask dropout m_t tạo source logits:

`z^(t) = source_head(h(x); m_t)`, `p_k^(t) = sigmoid(z_k^(t))`.

20 mask được lấy theo RNG liên tiếp trong một lượt sampling. Không reset cùng
seed cho từng draw, không perturb ảnh để thay cho MC. Giữ p=0,1 như lúc train.

### 3.1. Năm source scores bằng MC model averaging

`S_k = (1/T) Σ_t p_k^(t)`.

Lấy **mean của probabilities**, không sigmoid của mean logits. Vì sigmoid
phi tuyến, hai cách ấy không tương đương. Đây là phần thay cho single-pass
source score trong output MC mới, nhưng chưa thay inputs của calibrator cũ.

### 3.2. Năm đại lượng bất định về dự đoán nhãn nguồn

Dùng entropy Bernoulli chuẩn hóa:

`H_b(p) = [−p ln(p) − (1−p) ln(1−p)] / ln2`.

Cho mỗi nguồn k:

- Predictive entropy: `PE_k = H_b(S_k)`.
- Expected entropy: `EE_k = (1/T) Σ_t H_b(p_k^(t))`.
- Mutual information: `MI_k = PE_k − EE_k`.
- Standard deviation: `STD_k = sqrt(mean_t[(p_k^(t)−S_k)^2])`.

Nếu cần đúng **một vector 5 chiều bất định MC**, dùng
`[MI_semantic, MI_relation, MI_spatial, MI_depth, MI_occlusion]` và ghi rõ nó
là **model disagreement về năm nhãn nguồn, điều kiện trên representation cố
định**. Giữ vector source scores S riêng để không đánh tráo ý nghĩa.

Công thức entropy/MI theo MC là adaptation Bernoulli cho head đa nhãn, tham
khảo [Gal & Ghahramani (2016)](https://proceedings.mlr.press/v48/gal16.html) và
[BALD/MC estimator của Gal, Islam & Ghahramani (2017)](https://proceedings.mlr.press/v70/gal17a/gal17a.pdf).
Không suy rằng checkpoint hiện tại đã tối ưu chính xác posterior/ELBO của toàn
VLM. Tên phù hợp của MI lúc này là conditional model-disagreement proxy.

## 4. Vì sao đây là MC thật và phạm vi của nó?

[Implementation](../new/src/pcrau/source_mc.py) capture input thật của source
head trong một deterministic forward. Tạo bản sao head có weights giống hệt,
chỉ bật Dropout của bản sao; chạy 20 lượt và lưu raw probabilities `[T,B,5]`.
Không train mạng gốc, không đổi weights hoặc bật `.train()` cho baseline.
LayerNorm/Linear giữ eval; mọi weights của bản sao requires_grad=False.

Lấy mẫu trong context fork/restore RNG. Copy head tắt dropout phải xuất logits
khớp chính xác head gốc; đã kiểm tra ở mọi batch. Các mode của head bản sao
được khôi phục sau sampling. Wrapper frozen hiện hành vẫn giữ đầy đủ guard.

Đây là **source-head-only MC**, tức phép xấp xỉ subnet ngẫu nhiên ở lớp head
cuối. Nó không sampling backbone, Transformer, fusion, target/anchor maps hoặc
P1 MLP. Không gọi kết quả là toàn bộ epistemic uncertainty của Spatial VLM,
phân rã aleatoric/epistemic thuần hoặc causal attribution. Expected entropy
được lưu như thống kê, không gán thành noise vật lý đã được nhận dạng.

Nếu occlusion score cao và MI thấp, model nhất quán dự đoán nhãn occlusion;
không suy rằng ảnh không bị che khuất. Ngược lại, MI cao nghĩa là các subnet
bất đồng về nhãn, không chứng minh nguồn ấy thực sự có mặt. Một model sai
nhất quán vẫn có MI thấp.

## 5. Kết quả kiểm tra trên dữ liệu thật

Nguồn [summary](../new/outputs/pcrau_source_mc_train_dev_20261007/summary.json).
1.600 mẫu train/320 family, 400 mẫu dev/80 family, 20 draws/source head mỗi mẫu. Chỉ inference,
không train lại, fit calibration, mở IID hoặc OOD. Labels chỉ join sau khi
observable MC outputs và raw samples đã lưu.

5 unit checks đạt: entropy/MI cases biết trước, source score cao khác MI cao,
mean probability khác sigmoid mean logits, replay/modes/weights/RNG và invalid
input rejection. Same-seed replay 20 mẫu dev khớp raw samples; seed khác đổi
masks. T20 vs T100 trên 20 mẫu đầu dev có max mean delta 0,019823 và max MI
delta 0,005406; seed khác T20 có max mean delta 0,041590. Vì vậy không xem 20 draws
là exact expectation; T100 chỉ diagnostic, không thay T20 primary.

### Dev metrics tại threshold 0,5 cố định

| Nguồn | F1 single-pass | F1 MC mean | Brier single-pass | Brier MC mean | MI trung bình |
|---|---:|---:|---:|---:|---:|
| semantic |0,538776|0,540984|0,168354|0,168795|0,003393|
| relation |0,655172|0,655172|0,132987|0,133156|0,004101|
| spatial (weak) |0,965517|0,965517|0,003556|0,003549|0,000813|
| depth |0,628931|0,641026|0,099870|0,099648|0,003181|
| occlusion |0,543779|0,533937|0,177333|0,177280|0,003765|

MC không cải thiện đều mọi nguồn. Spatial F1 cao vẫn chỉ đánh giá proxy
AMBIGUOUS, không chứng minh semantic binding hay geometric localization.
Relation MI trung bình ở dự đoán sai không cao hơn dự đoán đúng trong lượt
này; không tuyên bố MI luôn nhận biết được lỗi. Có các confident-wrong cases
và source scores chưa calibrated. Đây là mô tả dev đã dùng chọn model trong
lịch sử, không phải xác nhận độc lập.

![MC scores và MI từ hai case dev thật](../new/outputs/pcrau_source_mc_train_dev_20261007/report/source_scores_and_MI.png)

### Một output thật cho câu/ảnh occlusion

`v211dev_family_000001__occlusion_view_counterfactual`, T20:

| Nguồn | MC source score | Std | Predictive entropy | MI |
|---|---:|---:|---:|---:|
| semantic |0,058082|0,008203|0,319785|0,000854|
| relation |0,156685|0,024547|0,626319|0,003146|
| spatial |0,003874|0,001284|0,036614|0,000290|
| depth |0,546520|0,038275|0,993747|0,004274|
| occlusion |0,848108|0,021703|0,614556|0,002627|

Occlusion score cao trong khi MI nhỏ: các draws tương đối nhất quán về nhãn.
Depth score≈0,55 có entropy gần 1 nhưng MI vẫn thấp: các subnet thường cùng
lưỡng lự gần 0,5, không được gọi đó là depth sensor noise thật. Hậu kiểm case
này chỉ có nhãn occlusion dương; depth là false positive tại threshold 0,5.
Tất cả numbers là outputs thật, không chỉnh tay cho khớp lời giải thích.

## 6. Code, artifacts và cách chạy

- [MC statistics và sampler](../new/src/pcrau/source_mc.py).
- [Runner train/dev](../new/scripts/audit_five_sources_mc.py).
- [Inference CLI cho một observation](../new/scripts/infer_five_sources_mc.py).
- [Tests](../new/tests/test_source_mc.py), [reporter](../new/scripts/report_five_sources_mc.py).
- [Execution lock trước MC](../new/outputs/pcrau_source_mc_train_dev_20261007/mc_execution_lock.json).
- [Observable MC outputs dev](../new/outputs/pcrau_source_mc_train_dev_20261007/dev/observable_mc_sources.jsonl),
  [raw 20 draws](../new/outputs/pcrau_source_mc_train_dev_20261007/dev/mc_samples.npz),
  [captured context](../new/outputs/pcrau_source_mc_train_dev_20261007/dev/source_context.safetensors).
- [Per-source CSV](../new/outputs/pcrau_source_mc_train_dev_20261007/report/dev_per_source.csv),
  [case table](../new/outputs/pcrau_source_mc_train_dev_20261007/report/REAL_CASE.md),
  [source-error diagnostics](../new/outputs/pcrau_source_mc_train_dev_20261007/report/source_error_detection.json).
- [Protocol](S5_FIVE_SOURCE_MC_PROTOCOL_20261007.md).

```bash
.conda-roborefer/bin/python3.10 -B new/scripts/infer_five_sources_mc.py \
  --features PATH_TO_FOUR_TENSOR_FEATURE_FILE.safetensors \
  --prompt "Locate the apple that is right of the purple cube." \
  --output PATH_TO_NEW_JSON.json
```

CLI giữ original runtime decision và xuất riêng source_MC. T=20 cố định;
seed được ghi trong output. Không feed MC mean/MI vào risk 44 features cũ; nếu muốn
fusion với risk, cần contract và calibrator mới khớp các features MC sau freeze.
Không dùng IID đã quan sát để chọn sampler rồi gọi kết quả xác nhận mới.

## 7. Điều có thể và chưa thể bảo vệ trước thầy

Có thể bảo vệ: head/source supervision xác định năm nhãn; năm mean scores
thực sự được tính bằng Monte Carlo; mỗi nhãn có entropy/std/MI từ 20 draws; có
raw outputs, công thức, reproducibility và đánh giá đúng/sai. Không cần train
lại vì source head đã học với dropout p=0,1.

Chưa thể bảo vệ: MC tự suy ra năm nguyên nhân, năm nhãn tạo partition của
uncertainty tổng, MI đo chính xác uncertainty vật lý của từng nguồn, hoặc đã
phân rã aleatoric/epistemic toàn VLM. Không nói kết quả IID cũ đã dùng MC.

Bản này giúp phát biểu đúng cơ chế đã làm; không đổi định nghĩa để hợp thức
hóa câu trả lời cũ. Model/bundle/profile cũ, dữ liệu, LaTeX và robot giữ nguyên.
