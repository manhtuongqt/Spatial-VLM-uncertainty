# S1 — Đề xuất protocol v2: kiểm tra objective trực tiếp cho anchor peak

Ngày **05/10/2026**. **DRAFT — chưa triển khai/chạy.** Dựa trên
[failure audit](S1_ANCHOR_FAILURE_AUDIT_20261005.md). Protocol và kết quả v1 giữ
nguyên. File này không cấp quyền tự train, mở verifier hoặc calibration/test.

## 1. Một giả thuyết được thử

**H_peak:** Khi residual đã phản ứng với noun span, dense BCE+Dice và mean
empty-BCE chưa trực tiếp bảo đảm dominant peak đúng instance hoặc không tạo
peak evidence cao ở empty anchors. Thêm **một auxiliary term case-conditioned
trên peak** có thể cải thiện correctness/evidence mà không tăng capacity.

Bằng chứng định hướng:13 dev misses đều có grid-center support, khoảng cách
MAP tới mask53–596 px;3 FOUND misses có best-positive rank12/5/3;3 train
negatives tăng cả max score và mean BCE. Audit không cô lập nguyên nhân và
không chứng minh H_peak đúng. Có thể residual/head hoặc visual/context
representation là bottleneck, đặc biệt same-class multiple instances và tiny
peripheral objects.

Không cùng lúc đổi kiến trúc, embedding, parser, sampling, presence head,
augmentation, learning rate, epoch budget, MC hoặc geometry. Không hard-mine
train bằng13 dev misses, không chuyển dev family sang train.

## 2. Ranh giới giữ nguyên

- Cùng active baseline bundle/hash, wrapper33.024 params và frozen head.
- Cùng64 train family/256câu/1024presentation;16dev family/64câu.
- Cùng seed24082026, initialization, four-prefix schedule và family batches.
- AdamW lr3e-4, wd1e-3, clip1, max15epoch, patience5, không scheduler.
- MLP/lossFP32, frozen AMP và shadow outputs riêng. q_new vẫn q_old+MLP(pool).
- Selection bằng dev tuple v1, không chọn checkpoint theo train negatives hoặc
  theo việc checkpoint nào cứu gate. Các mode can thiệp target/whole của audit
  không được dùng làm model candidates.
- Mask full/area-downsample chỉ vào loss/evaluator; không đi vào observable
  tensor capture/MLP/head hoặc sample metadata để model đoán family.

## 3. Đối chứng

| Arm | Conditioning | Objective | Vai trò |
|---|---|---|---|
| M0 | Historical frozen anchor | Không train | Baseline gốc |
| C1 | Anchor span | V1 visible BCE+Dice +0,25mean-empty BCE | Rerun control để kiểm tra replay learning/run |
| P1 | Anchor span | Objective C1 +0,10L_peak | Thử H_peak |
| P2 | Whole text | Objective C1 +0,10L_peak | Cùng capacity/loss, kiểm tra noun-span specificity |

So P1−C1 để đánh giá term mới; so P1−P2 để đánh giá noun selection dưới cùng
objective. C1 phải báo selection trace và so checkpoint/metrics với v1; nếu
runtime không tái tạo được phải giải thích trước khi quy delta cho loss.
Không dùng khác epoch thực do early-stop làm lý do train thêm arm thua.

## 4. Auxiliary term đề xuất

Cho logits slot0 `L_i` trên768cells; `A` là các cells có center nằm trong
**full-resolution train anchor mask**. `E` là active anchor full mask rỗng.
Quan sát full mask chỉ dùng supervision trong runner; không có oracle forward.

```text
Nếu visible và A không rỗng:
    ell_peak = relu(1.0 + max_{j not in A}(L_j) - max_{i in A}(L_i))
Nếu empty:
    ell_peak = softplus(max_j L_j)
Nếu visible nhưng A rỗng:
    skip riêng ell_peak; vẫn train BCE+Dice gốc

L_peak = mean ell_peak trên eligible active samples của batch
L_total_v2 = L_total_v1 + 0.10 * L_peak
```

Margin1, coefficient0,10 được đề xuất **trước thử nghiệm v2**, không phải
hyperparameters tối ưu có bằng chứng. Không quét rồi chọn sau xem dev dưới
tên một run prespecified. Một lần thực hiện đầu tiên khóa hash file/config
thực tế trước optimizer; nếu sửa thiết kế này, ghi revision trước train.

Ý nghĩa:

- Visible term trực tiếp ràng buộc ordering của cell peaks phù hợp primary
  point-in-mask event. Outside gồm target, wrong class, wrong same-class
  instance và nền; không cần thêm semantic labels vào objective.
- Empty term phạt evidence tại cell logit cao nhất; giữ mean-BCE cũ để không
  bỏ supervision toàn map. Softplus stable; không spatial-softmax vì luôn có
  peak ngay cả khi thiếu evidence.
- Đây không phải presence probability, semantic identity classifier hoặc
  uncertainty decomposition. Một cell peak đúng cũng chưa có segmentation
  quality, geometric relational consistency hoặc robot safety.

Một train case `v211dev_family_000077__relation_counterfactual` có8maskpixels
nhưng A rỗng. Không gán empty, không drop sample/family, không đổi denominator.
Trong subset hiện tại250/251 visible train samples có A; cả61 visible dev có A.
Term ranking không cứu được lỗi unrepresentable này, phải ghi riêng.

Risks của term:

- Hard max tập trung gradient vào ít cells, có thể bất ổn/ties; cố định flatten
  order/cách max, giữ FP32 loss và gradient clipping.
- Có thể thắng metric một cell nhưng heatmap/segmentation kém hơn; báo BCE/Dice,
  mass, các maps và confidently wrong, không chỉ hit rate.
- Có thể đẩy một correct cell lên thay vì hạ wrong evidence; empty guard và
  per-case absolute logits vẫn bắt buộc.
- Visible/empty cases chia sẻ residual; cải thiện positive có thể làm negatives
  tệ hơn. Chưa chứng minh term cân bằng được tradeoff.
- Wrong-instance apple có thể cần relation/context mà frozen F/query không
  cung cấp đủ. Nếu term thất bại, không mặc định cần tăng coefficient.

## 5. Preflight trước optimizer v2

Chỉ bổ sung kiểm tra cần cho objective mới:

1. Synthetic maps: peak sai thắng phải có gradient hạ wrong/tăng correct;
   khi margin đủ thì ranking termzero; empty peak cao có finite penalty.
2. Full-mask có pixels nhưng không center: chỉ skip auxiliary, không empty;
   direct/unsupported không có auxiliary; batch không eligible term=zero.
3. Oracle mask/center sets nằm riêng runner/loss; observable capture schema
   vẫn9keys+prompt, không label/ID/variant trong conditioning.
4. W2zero tái tạo M0 ở stepzero, onlyresidual trainable, baseline exact trên
   400dev, unchanged dataset/active bundle. Không nới identity tolerance.

Không sửa residual/checkpoint v1 để thực hiện preflight. Tạo outputs/version
riêng sau khi có nhiệm vụ triển khai v2.

## 6. Gate và cách bác bỏ giả thuyết

**Giữ nguyên toàn bộ engineering gates v1** cho P1 best dev checkpoint:

- Anchor≥49/61; pairs≥8/15; FOUND both≥12/16.
- Mỗi3dev empty sigmoid-max≤M0+1e-6.
- P1 hơn P2≥2visiblehits hoặc≥2both-hitpairs, không thấp hơn metric còn lại.
- Input/freeze/stepzero/shadow exact; median overhead≤20% theo cùng protocol.

Để quy lợi ích cho objective mới, báo **P1−C1**: fixed/broken IDs, delta theo
family và percentile CI5000seed24082026; anchor hoặc pairs phải cải thiện,
metric còn lại không giảm. Không dùng chỉ M0 làm control loss vì objective
lịch sử khác. Nếu P1 vượt M0 nhưng không vượt C1, H_peak chưa được hỗ trợ.

Báo cả5train empty per-case peak, mean BCE và counts các cell positive. Phần
giả thuyết “không tạo evidence cao ở empty” chưa được hỗ trợ nếu max vẫn tăng
ở ca train so M0; **không dùng các ca này chọn lại checkpoint**. Chỉ5train và
1dev family negative vẫn không chứng minh missing-detection generalization.

Nếu numeric gate qua nhưng CI chạm0 hoặc corrections tập trung ở một phrase,
chỉ nói đạt pilot engineering gate, chưa thắng ổn định. Không thay49thành48
hoặc tăng epoch/seed/coeff sau xem kết quả để cứu protocol.

## 7. Artifact và quyết định tiếp theo

Lưu frozen bundle/init/data hashes; v2 objective config/hash; control learning
trace; selected residuals; observable logits; evaluator-only center sets/hits;
paired family comparisons C1/P1/P2; mọi corrections/regressions;5train+3dev
negatives; raw-score/mass/latency; tiny-mask exception. Không overwrite v1.

**Nếu P1 fail:** giữ report âm, dừng objective hypothesis; quay lại phân tích
frozen representation/instance association, chưa thêm verifier.
**Nếu P1 đạt gates và objective có lợi ích so C1:** mới đề xuất thử verifier
evidence-only với missing-anchor contract riêng. Không tự promote model hoặc
nối vào Adapter/calibrator đã fit cho baseline outputs.

Tài liệu này là sản phẩm thiết kế sau audit, **chưa có checkpoint/metric v2**.
