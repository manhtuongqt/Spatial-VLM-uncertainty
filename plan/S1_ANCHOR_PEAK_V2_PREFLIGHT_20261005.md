# S1 — Peak-aware loss v2: triển khai và preflight

Ngày: **05/10/2026**. Kết quả: **S1_PEAK_V2_PREFLIGHT_PASS**.

Đã triển khai objective riêng và kiểm tra frozen inference/backward trên train/dev.
**Chưa tạo optimizer, chưa cập nhật weights, chưa train C1/P1/P2.** Preflight
chứng minh cơ chế loss và boundary chạy đúng; chưa chứng minh anchor grounding
cải thiện. Kết quả có học gần nhất vẫn là M1 **48/61**, chưa đạt gate **49/61**.

## 1. Sản phẩm và contract

- [Module loss/arm factory](../new/src/pcrau/anchor_peak_experiment.py).
- [11 kiểm tra kỹ thuật](../new/tests/test_anchor_peak_experiment.py).
- [Script preflight](../new/scripts/preflight_anchor_peak_v2.py).
- [Protocol v2 khóa trước preflight](S1_ANCHOR_PEAK_PROTOCOL_V2_LOCK_20261005.md)
  và [config máy đọc](../new/configs/anchor_peak_pilot_v2_20261005.json).
- [Artifact và hướng dẫn đọc](../new/outputs/pcrau_s1_anchor_peak_v2_preflight_20261005/README.md),
  [summary](../new/outputs/pcrau_s1_anchor_peak_v2_preflight_20261005/summary.json).

Baseline, `AnchorShadow`, parser và loss v1 được giữ nguyên. Module v2 chỉ
được gọi riêng; không thay default forward hoặc active profile. Full masks,
center support, visibility và pixel counts đi vào `PeakSupervision` cho loss,
không đi vào capture/query/head. Predictions được lưu trước khi đọc oracle masks.

| Arm | Conditioning | Objective |
|---|---|---|
| C1 | Anchor noun span | Loss v1 exact |
| P1 | Anchor noun span | Loss v1 + 0,10 × peak auxiliary |
| P2 | Toàn valid text | Cùng objective P1 |

Mỗi arm có residual MLP **33.024 tham số**, cùng seed **24082026**, cùng
zero-output initialization; không warm-start từ learned M1/M2. P1−C1 kiểm tra
đóng góp objective, P1−P2 kiểm tra noun-span specificity khi objective khớp.

## 2. Loss thực tế

Dense objective vẫn là visible BCE+Dice và **0,25 × mean empty BCE** như v1.
Area occupancy vẫn dùng OpenCV INTER_AREA. Auxiliary dùng **logits FP32**:

```text
A = các grid centers nằm trong full anchor mask
visible, có inside và outside:
    peak term = relu(1.0 + max_outside - max_inside)
full mask rỗng:
    peak term = softplus(max_all)
L_peak = mean các term trên eligible active slots trong batch
```

Full visibility được quyết định từ **số full-mask pixels**, không phải center
support. Với ảnh640×480/grid24×32, centers là `(20x+10,20y+10)`, khớp evaluator.
Visible nhưng không có center: bỏ riêng ranking, giữ BCE+Dice và mẫu số. Mask
phủ toàn grid không có outside cũng bỏ riêng ranking. Inactive không tham gia.
Ties chọn first index row-major; batch không eligible trả connected zero.

Ca thật `v211dev_family_000077__relation_counterfactual` có **8 mask pixels**,
không có grid center: vẫn visible, không bị chuyển thành empty hoặc drop sample.

## 3. Kiểm tra synthetic và gradient

**11/11 kiểm tra PASS**, bao gồm wrong dominant peak, satisfied margin,
empty logits ±1000, tiny mask, center khác area overlap, inactive/no-eligible,
no-outside, reduction/ties, C1 exact loss+gradient v1, P1/P2 cùng objective,
reject inconsistent/nonfinite supervision và frozen/oracle boundary.

- Khi ranking vi phạm margin, gradient làm tăng inside peak và giảm outside peak.
- Empty penalty có gradient dương tại max logit, nên descent hạ dominant peak;
  các logit rất âm có gradient nhỏ như softplus quy định.
- Tại margin đã thỏa, ranking gradient bằng0; không ép vô hạn logits.
- Khởi tạo W2zero làm **W1gradientzero là đúng**, W2 nhận gradient hữu hạn.

Backward thật dùng 3 câu train × 4 prefix =12 presentations: một visible có
center, một empty, một tiny-visible. Với P1/P2: dense visible8/empty4;
auxiliary eligible8 = ranking4 + empty4; skip tiny4. Chạy cả FP32/bfloat16,
và backward riêng auxiliary P1/P2 để xác nhận gradient không chỉ đến từ dense loss.

| Precision | C1 total | P1/P2 total ở init | L_peak |
|---|---:|---:|---:|
| FP32 | 0,597923338 | 0,859909534 | 2,619862080 |
| AMP bfloat16 | 0,598536730 | 0,860974312 | 2,624376059 |

Probe visible thật đã thỏa margin nên visible auxiliary ở probe bằng0; hướng
sửa wrong winner được kiểm tra trên synthetic maps. Không suy rằng mọi visible
train đã đạt margin từ probe này. Empty term mean FP32 là5,239724159. Mọi
gradient được kiểm tra hữu hạn; baseline không có gradient, W2 có gradient,
MLP/baseline state digest giữ nguyên sau backward và clearing gradients.

## 4. Identity, dữ liệu và boundary

| Kiểm chứng | Kết quả |
|---|---|
| Train inference | 1.024 presentations /64 family, cả hai precision |
| Dev inference | Toàn400mẫu /80 family, cả hai precision |
| Baseline tensor outputs | Exact với plain M0 trong FP32 và bfloat16 |
| C1/P1/P2 step-zero anchor error | Max absolute error **0,0** mỗi arm/mode |
| Dev scope | 64supported,140direct-bypass,196unsupported |
| Direct/unsupported/inactive | Map baseline exact; hooks được tháo |
| Oracle field rejection | Center support/full visibility/counts/full mask bị reject ở capture |
| Train support subset | 256câu/64family:251visible,5empty;250visible có center |
| Dev support subset | 64câu/16family:61visible có center,3empty |
| Family plans | Khớp v1 đủ15epoch; split và prefixes giữ nguyên |
| Protected hashes trong run | 806files giữ nguyên |

1.024 train presentations là augmentation của256câu trên64family, không phải
1.024 quan sát độc lập. 400dev dùng cho identity/regression; binding gate dùng
61visible trong64câu horizontal/16family. Không trộn hai mẫu số.

## 5. Latency

Cached sidecar forward, batch8, CUDA sync, warmup20 và100lần đo luân phiên
M0/C1/P1/P2 trên RTX2000Ada. Không gồm backbone, loss/backward hay disk I/O.

| Arm | Median ms | P95 ms | Median overhead vs M0 |
|---|---:|---:|---:|
| M0 | 5,180 | 5,437 | — |
| C1 | 5,746 | 5,973 | 10,94% |
| P1 | 5,732 | 5,970 | 10,65% |
| P2 | 5,726 | 6,016 | 10,54% |

Đều dưới gate20%. Peak loss chỉ dùng khi học, không thêm phép tính inference.
Các khác biệt nhỏ giữa C1/P1/P2 ở zero-init không chứng minh arm nào nhanh hơn
hoặc grounding tốt hơn. Đây không phải latency end-to-end.

## 6. Lock và giới hạn kết luận

| Lock | SHA-256 |
|---|---|
| Protocol v2 | `0c4efd907413523bf273044c9520662750bbbca0b2cb72218fe712b02531a412` |
| Config v2 | `f53e0f9c4bfd567c8791b7216d0a702535ce7a197f2aedcd72f4da97416c7e88` |
| Residual state init, cả ba arm | `e808b0e0b3a40dc4d26210f0a5b9bce08c4074a5da18f65e4ebc8698593e7057` |
| Frozen model state trước/sau | `66a2d17434f7ef18123d14db542b2d618130427e46d46d8bd9c5a3646296a9a2` |

File `initial_mlp.safetensors` chỉ là **init chưa học**, không phải best
checkpoint. Config, protocol, source, split/family plans và init đã được ghi
trước các backward; mọi optimizer counters vẫn0. Bảo toàn historical
checkpoints/audit, active bundle/calibrator/profiles và dữ liệu.

Đã chuẩn bị arm factory, loss dispatch và plans. **Epoch orchestration C1/P1/P2
chưa triển khai**: bước pilot sau phải nối loss vào runner riêng, kiểm tra lock
trước optimizer, dev mỗi epoch, best checkpoint/reload, early stopping và paired
report. Không dùng runner v1 trực tiếp rồi mặc định rằng nó gọi loss v2.

Gate có học vẫn49/61,8/15swap,12/16FOUND-both, từng dev empty không tăng,
P1 vượt P2 theo lock và báo P1−C1, fixed/broken và family CI. Chưa chấm các
gate này bằng preflight. Max-based objective có sparse gradient/tie sensitivity;
mask association, negative sample count nhỏ và một seed vẫn là giới hạn.

Không có train mới, verifier, MC, IID/OOD, calibration fit/samples hoặc robot.
Loss peak là **grounding/evidence supervision**, chưa phải estimator phân rã
aleatoric/epistemic hay calibrated error probability.
