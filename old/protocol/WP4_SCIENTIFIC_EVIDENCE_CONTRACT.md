# WP4 — Scientific evidence và reporting contract

> **Trạng thái:** thiết kế trước training; mọi metric chưa chạy phải là `NOT_RUN`, không dùng `0`, dấu `–` hoặc giá trị minh họa như thể là kết quả.
>
> **Mục tiêu:** quy định trước bảng, figure, image, đơn vị thống kê, checkpoint selection và ranh giới split cho P-CRA-U.

## 1. Nguyên tắc không thương lượng

1. Đơn vị độc lập là `scene-query family`, không phải variant/sample.
2. Tất cả bảng hiệu năng phải ghi `n_families`, `n_samples`, seed và 95% family-level confidence interval (CI).
3. Năm variants của một family luôn nằm cùng split và cùng bootstrap cluster.
4. `train` dùng tối ưu trọng số; `dev` dùng chọn kiến trúc/checkpoint; `calibration` chỉ fit calibrator sau khi model đã freeze; test chỉ mở một lần sau khi model, calibrator và threshold đã khóa.
5. Không chọn `best checkpoint`, early stop, hyperparameter hoặc ablation bằng calibration/test.
6. Mọi so sánh chính là paired trên cùng family/query. Báo effect size và CI, không chỉ p-value.
7. Bảng/figure được sinh từ prediction/metric artifact đã hash; không chép số thủ công vào báo cáo cuối.
8. Qualitative panel chọn theo strata và seed đã đăng ký, không chọn riêng các ví dụ đẹp.
9. Overfit smoke chỉ chứng minh loader/loss/gradient/checkpoint hoạt động; không được xuất hiện như kết quả generalization.

## 2. Pha chạy và quyền đọc split

| Phase | Split được đọc | Được cập nhật | Được chọn best theo | Claim được phép |
|---|---|---|---|---|
| `OVERFIT_SMOKE` | tiny train manifest | P-CRA-U sidecar | train diagnostic loss | Không có claim khoa học |
| `DEVELOPMENT_TRAIN` | train + dev | P-CRA-U sidecar | metric đã khóa trên dev | Preliminary/dev-only |
| `CALIBRATION_FIT` | calibration | calibrator, không sửa P-CRA-U | calibration objective đã khóa | Calibration-development |
| `LOCKED_TEST_IID` | Test-IID | không cập nhật gì | không chọn | Final IID |
| `LOCKED_TEST_OOD` | Test-OOD | không cập nhật gì | không chọn | Final OOD |

Checkpoint manager phải từ chối `selection_split=calibration|test_iid|test_ood`. Checkpoint model tốt nhất phải được chọn xong ở `DEVELOPMENT_TRAIN`; calibrator lưu thành artifact riêng và trỏ tới model checkpoint đã freeze.

## 3. Đánh giá sáu bảng đề xuất

### Bảng 1 — Dataset và độ độc lập thống kê: bắt buộc

Giữ các cột: split, families, samples, bốn answerability states, depth-dependent và OOD. Bổ sung:

- `status`: `READY`, `PROTOTYPE_ONLY`, `MISSING_LOCKED_SPLIT`;
- `family_definition` và `split_manifest_sha256`;
- số family theo relation/source ở bảng phụ;
- kiểm tra không trùng family/capture/template giữa các split theo đúng claim OOD.

Số state là sample-level để mô tả class balance; số family là sample size thống kê chính. Prototype WP2 hiện có 30/8/6/6 family và 150/40/30/30 sample, nhưng `test` hiện tại chỉ là prototype test, không được đổi tên thành Test-IID. Chưa có Test-OOD.

### Bảng 2 — Visual grounding chính: bắt buộc sau train

Giữ B0/B1/B2/U1/P1 và các metric đề xuất. Khóa denominator:

- `instance_accuracy`, `point_in_target`, `point_in_interior`, `2d_error`: báo trên ground-truth `FOUND` và báo riêng all-state selective result;
- `mass_in_target`, `top2_recall`: chỉ áp dụng phương pháp phát heatmap; baseline point ghi `NOT_APPLICABLE`;
- `2d_error`: khoảng cách Euclidean chuẩn hóa theo đường chéo ảnh; bổ sung pixel error ở bảng phụ nếu resolution cố định;
- latency: median, p95 và peak VRAM trên cùng hardware/batch/config.

Mỗi ô kết quả chính cần estimate và CI 95%. Bảng relation phụ phải thêm `n_families/n_samples`; không kết luận từ subgroup quá nhỏ.

Đây là bảng trả lời liệu P-CRA-U cải thiện raw grounding. Nếu P1 không hơn B1/U1, báo effect size đúng dấu; luận văn vẫn có thể mạnh nếu Bảng 3–4 chứng minh selective prediction/calibration.

### Bảng 3 — Answerability và phát hiện lỗi: bắt buộc sau train

Giữ Macro-F1, error AUPRC/AUROC, recall AMBIGUOUS/ABSENT và false FOUND. Định nghĩa primary safety error:

```text
false_found_negative = predicted FOUND
                       và truth thuộc {AMBIGUOUS, ABSENT}
false_found_rate = false_found_negative / số truth {AMBIGUOUS, ABSENT}
```

Bổ sung confusion matrix 4×4, recall `INSUFFICIENT_EVIDENCE`, balanced accuracy và CI theo family. AUPRC-error là metric ranking chính khi failure hiếm; AUROC là phụ.

### Bảng 4 — Calibration: bảng thesis-primary sau calibration

Giữ Brier, NLL, ECE, AURC, risk@80% coverage và coverage@5% risk. Bổ sung:

- calibration slope/intercept;
- CI family bootstrap;
- reliability diagram và risk–coverage curve;
- số family calibration/test;
- threshold và checkpoint hash đã freeze;
- subgroup calibration cho relation/depth/occlusion/OOD nếu đủ family.

Risk dùng một event đã khóa, mặc định `selected grounding incorrect`. ECE không đứng một mình. So sánh risk phải tại cùng coverage; so sánh coverage phải tại cùng risk target.

### Bảng 5 — Source attribution: bằng chứng phụ nhưng có giá trị

Giữ paired delta score. Bổ sung CI của delta, diagonal-dominance rate và off-target leakage:

```text
diagonal_margin = delta(correct_source) - max(delta(other_sources))
```

Pair phải là clean/counterfactual cùng family. Đây là controlled simulator attribution, không gọi là causal diagnosis trên dữ liệu thật.

### Bảng 6 — Ablation: bắt buộc nhưng giới hạn phạm vi

Ba ablation tối thiểu:

1. bỏ relation conditioning;
2. bỏ depth feature/counterfactual supervision (nên tách hai dòng nếu compute cho phép);
3. scalar risk so với multimodal risk.

Mỗi ablation dùng cùng split, seed policy, training budget và checkpoint-selection rule. Báo paired delta + CI. Nếu CI chứa 0 hoặc effect rất nhỏ, giới hạn claim về thành phần đó.

## 4. Bằng chứng còn thiếu phải bổ sung

### Bảng 7 — Training và checkpoint selection

| Run/seed | Best dev step | Selection metric | Train loss | Dev metric | Gradient finite | Resume exact | Checkpoint verified |
|---|---:|---|---:|---:|---:|---:|---:|

Bảng này phát hiện cherry-pick checkpoint và lỗi resume. Overfit smoke phải có loss component, gradient norm, learning rate và memory curve theo step.

### Bảng 8 — Efficiency và reproducibility

| Method | Trainable params | Total params | FLOPs/ước lượng | Feature ms | Head ms | End-to-end median/p95 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|

Không so latency giữa các máy/config khác nhau. Ghi GPU, CUDA, PyTorch, dtype, batch size, tile count và warm-up.

### Bảng 9 — Family-level statistics

Cho từng primary contrast lưu: metric, method A/B, paired family count, point estimate delta, bootstrap CI, test statistic/p-value nếu dùng, multiple-comparison correction và conclusion giới hạn.

### Bảng 10 — OOD/subgroup robustness

Chỉ sinh sau khi Test-OOD thực sự được khóa. Báo IID/OOD gap theo relation, target size, occlusion, depth quality và language template; subgroup quá nhỏ phải gắn `INSUFFICIENT_FAMILIES`.

### Bảng 11 — Selective execution và reobserve: bắt buộc cho RQ5

| Policy | Coverage | Accepted error | False accept | Wrong-object execution | Reobserve rate | Reobserve recovery | Utility/cost |
|---|---:|---:|---:|---:|---:|---:|---:|

So sánh tối thiểu `always_execute`, geometry gate, calibrated P-CRA-U và calibrated P-CRA-U + reobserve. Báo risk tại cùng coverage hoặc coverage tại cùng risk. Tách perception false accept khỏi controller timeout; replay offline là bằng chứng chính trước robot thật.

### Bảng 12 — Spatial confidence region/3D uncertainty: optional theo gate

Chỉ mở khi output spatial distribution và metric depth đã ổn định. Báo empirical coverage, coverage gap, normalized region area, coverage–area curve, 3D localization error và covariance calibration. Vùng phủ gần toàn ảnh không được xem là tốt chỉ vì coverage cao.

## 5. Figure bắt buộc

| ID | Figure | Pha đầu tiên được phép | Vai trò |
|---|---|---|---|
| F01 | Dataset balance theo family/state/relation/source | Trước train | Chứng minh coverage/split |
| F02 | Total/component loss + gradient norm + LR theo step | Overfit/train | Chẩn đoán optimization |
| F03 | Dev metric và checkpoint-selection step | Development | Chứng minh rule chọn model |
| F04 | Qualitative RGB + GT mask + B1 point + U1/P1 heatmap/modes | Development | Giải thích spatial output |
| F05 | Answerability confusion matrix + error PR curve | Development/test | Safety failure |
| F06 | Reliability diagram trước/sau calibration | Calibration/test | Calibration quality |
| F07 | Risk–coverage curve với cùng trục/denominator | Calibration/test | Figure chính selective prediction |
| F08 | Paired source-delta heatmap/forest plot | Development/test | Source specificity |
| F09 | Ablation paired-delta forest plot | Development/test | Thành phần đóng góp |
| F10 | IID–OOD/subgroup gap | Locked test | Robustness |
| F11 | Selective risk/utility và reobserve outcome | Calibration/test | Nối UQ với quyết định robot |
| F12 | Spatial region/3D covariance coverage–size | Locked test, optional | Set-valued/3D uncertainty |

Figure phải lưu dữ liệu nguồn dạng CSV/JSON cạnh SVG/PNG và ghi hash vào manifest.

## 6. Image panel định tính

Panel phải có tối thiểu các strata: FOUND-correct, FOUND-incorrect, AMBIGUOUS, ABSENT, INSUFFICIENT_EVIDENCE, relation-dependent, depth corruption, occlusion và OOD. Mẫu được chọn bằng hash-sort với seed khóa trước khi render.

Mỗi panel hiển thị, nếu hợp lệ:

- RGB và depth visualization;
- target/anchor mask chỉ trong evaluator overlay;
- RoboRefer point;
- target heatmap, anchor heatmap và top modes;
- answerability prediction;
- raw source scores (không gọi là calibrated probability);
- calibrated grounding risk chỉ sau calibration;
- sample/family ID và checkpoint hash.

Không dùng evaluator overlay làm input model.

## 7. Statistical protocol

- Primary CI: cluster bootstrap theo family, resample family rồi giữ toàn bộ variants của family; ít nhất 2,000 bootstrap replicates cho báo cáo cuối.
- Primary comparisons: paired family bootstrap trên difference.
- Nhiều seed: báo mean và between-seed spread; prediction-level bootstrap không thay thế seed variability.
- Metric/subgroup ít family: báo raw numerator/denominator và `INSUFFICIENT_FAMILIES`, không chỉ phần trăm.
- Multiple comparisons: xác định primary contrasts trước; exploratory contrasts gắn nhãn exploratory và hiệu chỉnh Holm nếu báo p-value.
- Không bootstrap năm variants như năm family độc lập.

## 8. Checkpoint policy

Checkpoint là directory bất biến gồm:

```text
checkpoints/step_000000100/
  model.safetensors
  training_state.pt
  metadata.json
  manifest.json
checkpoints/last.json
checkpoints/best_<metric>.json
```

Yêu cầu:

- lưu qua temporary directory cùng filesystem rồi atomic rename;
- `model.safetensors` tách khỏi optimizer/RNG pickle;
- SHA-256 + size cho mọi file;
- từ chối model tensor NaN/Inf;
- lưu model, optimizer, scheduler, AMP scaler, epoch/step, RNG Python/NumPy/Torch/CUDA, config và identity hashes;
- verify manifest trước resume;
- resume phải khớp run/config/dataset/split/code/feature-contract identity;
- `best` chỉ theo `train` trong overfit diagnostic hoặc `dev` trong development;
- không tự xóa checkpoint; pruning là thao tác riêng, có manifest và phê duyệt;
- chỉ rank 0 ghi checkpoint trong distributed run;
- `last` phục vụ resume, `best` phục vụ model selection; không coi `last` mặc nhiên là model cuối.

## 9. Cấu trúc run chuẩn

```text
results/pcra_u_runs/<run_id>/
  RUN_CARD.md
  config/
  checkpoints/
  logs/
  metrics/
  predictions/
  tables/
  figures/
  images/
  manifests/
  checks/
```

Checkpoint lớn nằm trong run gốc. `results/experiment_archive` chỉ nhận report, bảng, figure, image và hash dẫn xuất; không nhân đôi checkpoint/tensor.

## 10. Điều kiện chuyển bước

Ngay lúc này chỉ được:

1. khóa overfit smoke contract/manifest;
2. implement loader/model/loss/checkpoint;
3. chạy tiny train-only overfit diagnostic;
4. kiểm tra loss/gradient/resume/checkpoint integrity.

Chưa được điền Bảng 2–6 bằng kết quả overfit, chưa scale dataset và chưa fit calibration. Dataset expansion theo family phải giải quyết thiếu AMBIGUOUS/ABSENT ở dev/calibration và tạo Test-IID/Test-OOD mới trước claim cuối.
