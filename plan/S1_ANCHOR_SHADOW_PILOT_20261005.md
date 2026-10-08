# S1 — Pilot có học M0/M1/M2: kết quả và quyết định gate

Ngày thực hiện: **05/10/2026**. Theo yêu cầu chạy pilot có học của người dùng.

**Pilot đã hoàn thành; gate nâng cấp FAIL.** M1 dùng anchor noun span đạt
**48/61 anchor hits**, **8/15 cặp matched đúng cả hai câu** và
**12/16 FOUND đúng cả target–anchor**. M2 dùng toàn câu đạt **44/61, 6/15,
11/16**, bằng baseline ở ba metric này. M1 cải thiện nhưng chưa đạt gate
**49/61** đã khóa; không coi thiếu một ca là đủ để thông qua.

Hai residual checkpoint được lưu riêng. Baseline, dataset, active profile và
protocol v1 được giữ nguyên. Chưa mở verifier, MC, IID, OOD hoặc calibration.

## 1. Protocol và cách chạy thực tế

Protocol gốc: [S1 phrase-conditioned experiment](S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md),
SHA-256 `c908d287790baa41f7e63656dc99a4a278d55b3e28e8d958dddc1c645e810932`.
File thiết kế được bảo toàn, nên câu “chưa train” trong đó ghi trạng thái
**lúc khóa protocol**, không phải trạng thái sau pilot này.

M0 là P-CRA-U đang được chọn: V2 đóng băng + detail_cost Adapter + logistic33.
M1/M2 chỉ thêm residual MLP 128→128→128, **33.024 tham số mỗi nhánh**, dùng
frozen anchor head để xuất map riêng. M1 mean-pool contextualized tokens của
anchor phrase; M2 mean-pool tất cả valid text tokens. Các tokens M1 vẫn có
context toàn câu; đây không phải embedding noun độc lập hoàn toàn.

| Thành phần | Cấu hình thực tế, giống nhau giữa M1/M2 |
|---|---|
| Initialization | Seed 24082026, cùng state digest; lớp cuối zero |
| Train | 64 family, 256 câu horizontal, bốn prefix = 1.024 presentation |
| Dev selection | 16 family, 64 câu horizontal, không augmentation |
| Batch | 4 family, 64 presentation; 16 optimizer steps/epoch |
| Observable tensors | Cùng cached RGB/depth features, text và prompt-derived parser masks |
| Frozen forward | Eval/no_grad; cùng capture baseline được chia sẻ cho hai nhánh |
| Precision | MLP/loss FP32; frozen extractor/head AMP bfloat16 trên CUDA |
| Loss | Visible BCE+Dice hiện có + 0,25 × empty zero-map BCE |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-3, clip norm 1; không scheduler |
| Trần ngân sách | 15 epoch/nhánh; patience 5 theo dev selection |
| Selection | Lexicographic: anchor hits, matched both-hit pairs, negative empty mean sigmoid-max; hòa chọn sớm |

Orchestration đã thêm evaluate mỗi epoch, lưu checkpoint khi tuple cải thiện,
early stopping riêng từng nhánh, reload best rồi chấm lại. Trong log evaluate
của một nhánh có cả predictions của nhánh còn lại tại thời điểm ấy; chỉ tuple
của **nhánh đang được evaluate** được dùng chọn checkpoint cho nhánh đó.

| Nhánh | Epoch thực chạy | Optimizer steps thực chạy | Epoch được chọn | Steps tại checkpoint chọn | Lý do dừng |
|---|---:|---:|---:|---:|---|
| M1 | 15 | 240 | 11 | 176 | Đạt trần 15 epoch |
| M2 | 6 | 96 | 1 | 16 | Năm epoch liên tiếp không cải thiện |

Hai nhánh có cùng **trần ngân sách và stop rule**, không có cùng số steps
thực chạy. Đây là kết quả early stopping đã khóa. Không ép M2 chạy thêm hay
chọn checkpoint khác sau khi xem gates. Thời gian vòng train khoảng 40,76 giây,
không gồm toàn bộ khâu chuẩn bị, chấm cuối và đo latency.

Runtime: Python 3.10.14, Torch 2.13.0+cu130, CUDA 13.0,
NVIDIA RTX 2000 Ada Generation. Khởi tạo residual state digest cả hai nhánh:
`e808b0e0b3a40dc4d26210f0a5b9bce08c4074a5da18f65e4ebc8698593e7057`.

## 2. Metric và mẫu số

Anchor hit là MAP pixel center của grid 24×32 rơi trong **full-resolution
anchor mask**. Với ảnh 640×480, center cell là `(20x+10, 20y+10)`.
Cell-overlap được lưu riêng; không dùng nó thay full-pixel hit.

Visible-anchor không đồng nghĩa truth FOUND. Câu thiếu target vẫn có thể có
anchor nhìn thấy. Empty mask nghĩa là thiếu visible support của anchor tại
ảnh này, không khẳng định vật không tồn tại trong thế giới.

| Tập | Family | Câu | Visible anchor | Empty anchor | FOUND | Cặp matched |
|---|---:|---:|---:|---:|---:|---:|
| Train, original prefix | 64 | 256 | 251 | 5, thuộc 5 family | 57 | 51 |
| Dev | 16 | 64 | 61 | 3, thuộc 1 family | 16 | 15 |

Cặp matched lấy clean/relation_counterfactual cùng cached features và cả hai
anchor có pixel. Cờ `anchor_phrase_changed` là diagnostic, không phải filter
thêm vào protocol. Một cặp train có cùng noun phrase ở hai câu vẫn thuộc subset
51 cặp; không mặc định mọi cặp matched là can thiệp đổi noun độc lập.

## 3. Kết quả checkpoint đã chọn

| Metric dev | M0 | M1 phrase | M2 whole-text |
|---|---:|---:|---:|
| Anchor hit | 44/61 (72,13%) | **48/61 (78,69%)** | 44/61 (72,13%) |
| Matched pair đúng cả hai câu | 6/15 | **8/15** | 6/15 |
| FOUND đúng target–anchor | 11/16 | **12/16** | 11/16 |
| FOUND target hit | 14/16 | 14/16 | 14/16 |
| FOUND anchor hit | 12/16 | 13/16 | 12/16 |
| Cặp matched cùng peak | 2/15 | 1/15 | 2/15 |
| Anchor sai và raw sigmoid-max ≥0,9 | 11 | 6 | 10 |
| Anchor sai nhưng peak nằm trong target mask | 2 | 1 | 2 |

Raw sigmoid-max ≥0,9 chỉ là diagnostic confidently wrong, chưa phải calibrated
probability. M1 vẫn còn **13 visible-anchor misses**, trong đó **6** có score
≥0,9. Không coi giảm số confidently wrong là đã giải quyết uncertainty.

| Metric train | M0 | M1 phrase | M2 whole-text |
|---|---:|---:|---:|
| Anchor hit | 209/251 (83,27%) | 227/251 (90,44%) | 210/251 (83,67%) |
| Matched pair đúng cả hai câu | 29/51 | 39/51 | 29/51 |
| FOUND đúng target–anchor | 44/57 | 49/57 | 44/57 |
| FOUND target hit | 52/57 | 52/57 | 52/57 |
| Anchor sai và sigmoid-max ≥0,9 | 33 | 21 | 32 |

Train là dữ liệu đã học, nên CI/train improvements chỉ mô tả fitting,
không chứng minh generalization.

## 4. Chấm gate v1, không thay gate sau khi thấy kết quả

| Gate | Yêu cầu | M1 thực tế | Kết luận |
|---|---|---|---|
| Kỹ thuật | Input isolation, freeze, baseline exact, bypass | Đạt | PASS |
| Visible anchor | ≥49/61 | **48/61** | **FAIL** |
| Matched pairs | ≥8/15 | 8/15 | PASS |
| FOUND both | ≥12/16 | 12/16 | PASS |
| Empty dev | Mỗi ca không tăng quá 1e-6 so M0 | 3/3 không tăng | PASS |
| Phrase so M2 | Không kém ở hits/pairs; hơn ≥2 hits hoặc pairs | +4 hits, +2 pairs | PASS |
| Latency | Median overhead ≤20% | M1 +10,84%; M2 +11,52% | PASS |

**Gate tổng FAIL.** Có tín hiệu chọn noun span có ích trong pilot, nhưng chưa
đủ điều kiện mở verifier theo v1. Chưa promote hoặc nối maps mới vào
graph/Adapter. Target/answerability/source/risk không đổi vì vẫn đi đường M0;
đây là bảo toàn shadow mode, không phải bằng chứng không regression sau tích hợp.

## 5. Paired corrections, regressions và CI theo family

| Tập | Đối chứng | Ca sửa | Ca làm sai | Net hit |
|---|---|---:|---:|---:|
| Dev | M1−M0 | 4 | 0 | +4 |
| Dev | M1−M2 | 4 | 0 | +4 |
| Dev | M2−M0 | 0 | 0 | 0 |
| Train | M1−M0 | 18 | 0 | +18 |
| Train | M1−M2 | 17 | 0 | +17 |
| Train | M2−M0 | 2 | 1 | +1 |

Family paired delta được lưu cho **đủ 80 family × 3 đối chứng** tại
[family_paired_deltas.csv](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/paired_report_r2/family_paired_deltas.csv).
Bootstrap resample family 5.000 lần, seed 24082026; mọi variant/cặp trong
family đi cùng nhau. Bảng dưới dùng đơn vị **percentage points**.

| Dev, M1−M0 (M1−M2 giống nhau) | Delta | CI percentile 95% |
|---|---:|---:|
| Visible anchor hit rate | +6,56 pp | [0,00; 18,03] pp |
| Matched both-hit rate | +13,33 pp | [0,00; 33,33] pp |
| FOUND both-hit rate | +6,25 pp | [0,00; 20,00] pp |

CI đều chạm 0. Dev vừa dùng chọn checkpoint, CI có điều kiện trên checkpoint
đã chọn, **không sửa selection bias**; chỉ một seed và 16 family. Chưa có
bằng chứng thắng ổn định ngoài development này. M2−M0 CI [0;0] ở ba metric
trên chỉ phản ánh hit flags giống nhau trong mẫu, không khẳng định mô hình
tương đương trên mọi dữ liệu.

Train M1−M0: anchor +7,17 pp, CI [4,02;10,80]; matched pairs +19,61 pp,
CI [9,26;31,25]; FOUND both +8,77 pp, CI [2,04;16,39]. Các số này không được
dùng thay CI dev.

## 6. Các case thật được sửa và lỗi còn lại

Tọa độ là predicted MAP `(x,y)` trên ảnh 640×480; hit được hậu kiểm bằng mask,
mask không được đưa vào prediction. IDs dưới đây khớp artifacts, không đổi tên.

| Sample ID | Parsed target → relation → anchor | M0/M2 MAP | M1 MAP | Hit M0/M1/M2 |
|---|---|---|---|---|
| `v211dev_family_000030__relation_counterfactual` | green cube → right_of → lemon | (290,150) | (150,270) | Sai/đúng/sai |
| `v211dev_family_000154__clean` | apple → left_of → lemon | (390,170) | (290,90) | Sai/đúng/sai |
| `v211dev_family_000154__depth_corruption` | apple → left_of → lemon | (390,170) | (290,90) | Sai/đúng/sai |
| `v211dev_family_000154__occlusion_view_counterfactual` | apple → left_of → lemon | (390,170) | (290,90) | Sai/đúng/sai |

**Bốn ca sửa chỉ thuộc hai family và đều có anchor `lemon`**. Ba variants của
family 154 không tạo ba cảnh độc lập. Truth của family154 ở ba dòng này là
ABSENT: sửa anchor không biến câu thiếu target thành FOUND. Chỉ ca family030
là FOUND và tăng metric FOUND both. Không gọi đây là cải thiện khả năng
chịu depth/occlusion tổng quát.

Không có hit regression M1 trên subset đánh giá này; tuy nhiên map/score của
nhiều ca đã sai vẫn thay đổi. Ví dụ `000042__clean`, anchor green cube,
vẫn sai ở (350,230), score tăng từ 0,97364 lên 0,99142.
`000081__relation_counterfactual`, anchor mango, vẫn sai, score M1 0,99840;
`000383__relation_counterfactual`, anchor apple, vẫn sai, score M1 0,99993.

M2 có train regression thật ở `v211dev_family_000050__occlusion_view_counterfactual`:
target fruit, anchor orange; M0/M1 ở (390,170) đúng, M2 chuyển sang (290,370)
sai. Hai train corrections của M2 là `000077__occlusion_view_counterfactual`
và `000355__occlusion_view_counterfactual`. Dùng đầy đủ prefix
`v211dev_family_` khi tra artifact.

Trace, IDs, mask/RGB paths, scores và hit flags mọi ca thay đổi:
[changed_cases.csv](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/changed_cases.csv).
Danh sách 13 dev misses còn lại:
[remaining_dev_misses.jsonl](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/paired_report_r2/remaining_dev_misses.jsonl).

## 7. Stratification dev: lợi ích tập trung ở đâu?

| Truth | Visible anchors | M0 hit | M1 hit | M2 hit |
|---|---:|---:|---:|---:|
| FOUND | 16 | 12 | 13 | 12 |
| ABSENT | 21 | 8 | 11 | 8 |
| AMBIGUOUS | 6 | 6 | 6 | 6 |
| INSUFFICIENT_EVIDENCE | 18 | 18 | 18 | 18 |

Ba ca empty đều thuộc ABSENT và không nằm trong visible-anchor denominator.
Theo variant, clean/depth/occlusion đều từ 11/15 lên 12/15;
relation-counterfactual từ 11/16 lên 12/16. Theo phrase, `lemon` từ **0/10 lên
4/10**; mọi phrase dev khác giữ nguyên số hits. Vì vậy không được kết luận
binding đã tốt cho mọi noun; lemon vẫn còn 6/10 misses.

## 8. Evidence khi anchor mask rỗng

Dev ba ca thuộc cùng `v211dev_family_000392`, raw sigmoid-max:

| Variant | M0 | M1 | M2 | M1 guard |
|---|---:|---:|---:|---|
| clean | 0,725649 | 0,090093 | 0,749087 | PASS |
| depth_corruption | 0,479991 | 0,075858 | 0,510252 | PASS |
| occlusion_view_counterfactual | 0,848972 | 0,706596 | 0,860604 | PASS |

M1 giảm từng ca; M2 tăng từng ca. Nhưng M1 occlusion vẫn 0,7066, và chỉ
một family dev negative: chưa chứng minh presence detection generalization.

Train năm ca rỗng là relation_counterfactual của năm family riêng:

| Family suffix | M0 | M1 | M2 |
|---|---:|---:|---:|
| 000023 | 0,995801 | 0,999596 | 0,996055 |
| 000061 | 0,309024 | 0,300746 | 0,364778 |
| 000072 | 0,214691 | 0,577019 | 0,214691 |
| 000173 | 0,332852 | 0,073696 | 0,380759 |
| 000287 | 0,557364 | 0,975577 | 0,677474 |

M1 train empty mean tăng **0,481946→0,585327**, tăng ở **3/5 ca**; max lên
0,999596. Empty loss là mean BCE toàn map, trong khi guard đo max một pixel:
hai mục tiêu khác nhau. Chưa có ablation để quy nguyên nhân; không kết luận
H3 “giảm evidence giả nói chung” đã thành công chỉ từ dev guard PASS.

## 9. Latency và kiểm chứng kỹ thuật

| Nhánh | Median ms | P95 ms | Median overhead |
|---|---:|---:|---:|
| M0 | 5,466 | 5,849 | — |
| M1 | 6,059 | 6,584 | +10,84% |
| M2 | 6,096 | 6,711 | +11,52% |

Cùng batch8 cached features, setup wrapper/parser, warm-up20 và 100 lần/nhánh,
đồng bộ CUDA, luân phiên thứ tự đo. Không gồm feature loading/backbone hoặc
robot, không phải end-to-end latency. Peak allocated GPU của phiên đo chung
ba nhánh/shared baseline là **117.985.792 bytes**; không phải memory riêng từng
nhánh. Raw timing nằm ở [latency.json](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/latency.json).

Step-zero trong run khớp M0 trước optimizer. Reload selected safetensors tái
tạo chính xác dev metrics đã chọn. Cuối run, mọi baseline tensor output trên
**toàn bộ 400 dev**, direct/unsupported/inactive maps, weights/stats và active
profile được kiểm tra bất biến. Frozen model state digest trước/sau:
`66a2d17434f7ef18123d14db542b2d618130427e46d46d8bd9c5a3646296a9a2`.
Không có oracle mask/ID trong observable branch; mask chỉ vào train loss hoặc
evaluator hậu kiểm. Calibrator/profiles chỉ được kiểm tra bundle hash, không
đọc calibration samples, fit hoặc chấm test.

**16 kiểm tra scoped** của wrapper và pilot evaluator đã pass, gồm gradient,
freeze, input isolation, zero-init, AMP/stride, early-stop selection, mask-empty
guard từng ca, family bootstrap, pair eligibility và safetensors reload.
Môi trường không có pytest; các hàm test không cần fixture được thực thi trực
tiếp bằng Python `runpy`, không cài dependency mới. Không chạy suite toàn repo.

## 10. Sửa lỗi reporting có truy vết, không sửa kết quả học

Khi audit cuối, phát hiện helper evaluator v1 dùng filter
`visible AND same_cache AND anchor_phrase_changed`, thêm điều kiện không có
trong protocol. Nó bỏ cặp train family000050, khiến mẫu số 50 thay vì 51.
Các arm đều sai ít nhất một câu ở cặp này, nên tử số 29/39/29 không thay đổi.
Dev cả 15 cặp đều đổi phrase, nên mọi dev selection/gate/prediction/checkpoint
không bị ảnh hưởng. Train loss không dùng filter pair, cũng không bị ảnh hưởng.

Đã sửa helper để dùng `visible AND same_cache`; giữ phrase-change làm diagnostic
và thêm kiểm tra chống regression. **Không train lại, không chọn lại checkpoint,
không sửa artifact gốc.** Script reporting chỉ đọc saved evaluator rows, tính
lại pairing/CI và lưu ở `paired_report_r2/`. Raw logits/predictions, epoch history,
summary v1 và residual safetensors giữ nguyên. Source thực sự đã train được
archive cùng SHA-256 trong `source_snapshot/`, phân biệt với source đã sửa.

Train paired rate/CI dùng bản r2; summary v1 giữ để audit lỗi cũ, không dùng
làm báo cáo cuối. Reporting r2 kiểm tra dev metrics/CI/selection không đổi,
checkpoints khớp freeze lock và mọi protected file ngoài hai source đã archive
giữ hash cũ. Báo cáo này dẫn tới cùng quyết định **FAIL**, không thay đổi gate.

## 11. Checkpoints, outputs và code bàn giao

Artifact root:
`new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/`.

| Sản phẩm | File |
|---|---|
| M1 residual best | [mlp.safetensors](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/M1/checkpoints/best/mlp.safetensors) |
| M2 residual best | [mlp.safetensors](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/M2/checkpoints/best/mlp.safetensors) |
| Báo cáo JSON cuối, r2 | [summary.json](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/paired_report_r2/summary.json) |
| Epoch selection/loss trace | [epoch_history.json](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/epoch_history.json) |
| Observable predictions, 256 train + 400 dev | [selected_inference_predictions.jsonl](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/selected_inference_predictions.jsonl) |
| Raw target/anchor logits | [selected_spatial_logits.npz](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/selected_spatial_logits.npz) |
| Evaluator labels/hits, 320 horizontal | [evaluator_rows.jsonl](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/evaluator_rows.jsonl) |
| Matched pairs r2 | [swap_pairs.jsonl](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/paired_report_r2/swap_pairs.jsonl) |
| Freeze/bundle/checkpoint hashes | [freeze_lock.json](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/freeze_lock.json) |
| Reporting provenance | [provenance.json](../new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/paired_report_r2/provenance.json) |

SHA-256 M1 checkpoint:
`40e3e26a7e524ca0187f92204d7268ad69f824c9e015370ea2b6dd1160aa36c9`.
SHA-256 M2 checkpoint:
`2a9a5995e978db8dc58e75f263f40e0bd4845d2af3840e78e8a603084eae9a50`.
Đây là **residual-only**, cần wrapper và đúng baseline bundle đã khóa, không
phải hai model checkpoint đầy đủ thay được active profile.

Code mới: [runner orchestration](../new/scripts/run_anchor_shadow_pilot.py),
[evaluator](../new/src/pcrau/anchor_shadow_pilot.py),
[reporting từ traces](../new/scripts/report_anchor_shadow_pilot.py),
[scoped tests](../new/tests/test_anchor_shadow_pilot.py). Runner từ chối output
folder đã tồn tại; reporting cũng từ chối overwrite `paired_report_r2`.

## 12. Kết luận và hướng xử lý tiếp theo

**Đã triển khai và đánh giá:** pilot residual M1/M2, checkpoint selection và
đối chứng matched M0/M1/M2. Có tín hiệu noun-span selection giúp binding trong
scope horizontal, nhưng lợi ích dev tập trung ở lemon/hai family và CI rộng.
Gate tối thiểu binding vẫn FAIL. H3 về giảm evidence giả toàn diện chưa đạt
bằng chứng; train negatives có regressions cần giải thích.

**Chưa đạt:** binding đủ gate để mở verifier, explicit geometric reasoning,
uncertainty evidence mới tham gia decision, calibrated risk mới hoặc phân rã
aleatoric/epistemic. Branch mới chưa thay đổi selective risk/PIT của P-CRA-U.

Bước nghiên cứu phù hợp là diagnostic **13 dev misses và 5 train negatives**
từ traces đã lưu: phân biệt peak nhầm vật, pixel-center lệch mask, evidence yếu
và điều kiện hóa còn bị target/context chi phối. Nếu cần thử tiếp, viết
protocol v2 có giả thuyết cụ thể và giữ v1 thất bại này; không nới 49 xuống48,
không kéo dài thêm epoch/seed để cứu gate hoặc mở IID tìm cấu hình.
Vòng diagnostic/thí nghiệm mới và verifier chưa được thực hiện trong pilot này.
