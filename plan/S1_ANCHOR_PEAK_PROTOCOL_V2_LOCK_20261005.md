# S1 — Protocol v2 khóa cho peak-aware preflight và pilot tương lai

Ngày: **05/10/2026**. Theo yêu cầu triển khai loss và preflight của người dùng.
Kế thừa [đề xuất v2](S1_ANCHOR_PEAK_OBJECTIVE_V2_PROPOSAL_20261005.md) và
[failure audit](S1_ANCHOR_FAILURE_AUDIT_20261005.md). File này được hash trước
preflight. **Chỉ preflight/backward được thực hiện; chưa tạo optimizer/train.**

## 1. Giả thuyết và cấu hình giữ nguyên

H_peak: một term trên dominant peak có thể bổ sung dense objective để tìm đúng
anchor instance và tránh evidence peak cao khi full mask rỗng. Chưa có bằng
chứng cải thiện có học. Không thêm presence head, MC hoặc geometric verifier.

| Hạng mục | Giá trị khóa |
|---|---|
| Baseline | Active P-CRA-U V2 + detail_cost Adapter; logistic33 hash-only |
| Wrapper/head | `AnchorShadow` v1, frozen head; residual 33.024 params |
| Seed/init | 24082026, cùng init ba arm, lớp cuối zero |
| Train | 64 family, 256 câu horizontal, 4 prefix = 1.024 presentations |
| Dev | 16 family, 64 horizontal; toàn 400 dev cho regression/bypass |
| Batch | 4 family, đủ 4 câu × 4 prefix = 64 presentations, 16 batches/epoch |
| Optimizer dự kiến | AdamW lr3e-4, wd1e-3, clip1; không scheduler |
| Ngân sách dự kiến | Max15 epoch; patience5; cùng rule mọi arm |
| Precision | Loss/MLP FP32; frozen capture/head FP32 hoặc AMP bfloat16 |
| Dev selection | Anchor hits → matched both-hit pairs → negative empty mean sigmoid-max; hòa lấy sớm |
| Bootstrap sau train | 5.000 resamples theo family, seed24082026 |

## 2. C1/P1/P2 và loss

| Arm | Pooling | Loss |
|---|---|---|
| C1 | Anchor phrase | V1 exact: visible BCE+Dice +0,25mean-empty BCE |
| P1 | Anchor phrase | C1 loss +0,10L_peak |
| P2 | Toàn valid text | C1 loss +0,10L_peak |

Không warm-start residual từ learned M1/M2 của pilot trước. Ba arm khởi tạo
giống nhau từ zero-output MLP, chia sẻ frozen baseline và cùng data/order/prefix.
C1 là control loss; P1−C1 đo objective, P1−P2 đo noun-span specificity.

`A` là cell centers có pixel thuộc full anchor mask. Với ảnh640×480 và grid
24×32, centers `(20x+10,20y+10)`; full-visible là full pixel count>0. Area mask
dùng đúng OpenCV INTER_AREA như v1, không binarize occupancy để đổi BCE+Dice.

```text
visible, A và complement(A) đều không rỗng:
    ell = relu(1.0 + max_outside(logit) - max_inside(logit))
empty full mask:
    ell = softplus(max_all(logit))
visible nhưng A rỗng:
    skip auxiliary, giữ dense BCE+Dice
visible với A phủ toàn grid, không có outside:
    skip auxiliary, giữ dense BCE+Dice

L_peak = mean ell trên eligible active slots trong batch
P1/P2 total = V1 total +0.10L_peak
```

Clarification kỹ thuật không đổi hyperparameter: ties lấy **first index khi
flatten row-major**, dùng `torch.max(dim=-1)` để gradient không chia ngẫu nhiên;
trường hợp không outside được skip riêng ranking. Không tính phép `-inf-(-inf)`
trên rows bị skip. Batch không eligible có connected zero loss/zero gradients.
Mean gộp visible/empty eligible slots, không cộng hai group-means độc lập.

Ca thật `v211dev_family_000077__relation_counterfactual`:8pixels, không có center.
Vẫn visible cho BCE+Dice, không empty, không drop sample/family hoặc đổi metric.
Auxiliary áp dụng 250/251 visible train mẫu gốc;4 presentations của ca này được
skip auxiliary. Full-mask empties 5 train/3 dev; mọi61 visible dev có centers.

## 3. Boundary và kiểm chứng trước optimizer

- Supervision object chỉ chứa area mask, center support, full visibility/pixel
  counts; chuyển vào loss riêng. Wrapper/capture/head vẫn nhận chín model keys
  và prompt; không đổi schema hoặc default forward.
- Phải reject oracle fields ở capture; parser direct/unsupported/inactive
  bypass; full-pixel/area/center flags sai nhất quán phải fail, không tự sửa.
- Step-zero FP32 max error≤1e-6 và AMP error được báo; giữ baseline outputs
  exact trên400dev. Kiểm tra thêm1.024train presentations cả hai precision.
- Backward phải finite, gradient chỉ residual; W1 gradientzero ở W2zero là
  expected. Không optimizer step, không cập nhật baseline weights/stats.
- Tiny-mask skip chỉ auxiliary; empty gradient hạ max logit; wrong dominant
  visible peak gradient hạ outside/tăng inside; satisfied marginzero; ties và
  extreme logits hữu hạn; C1 loss/gradient exact v1.
- Config/protocol/code/data/init/bundle hashes được lưu trước optimizer. Trong
  bước này chỉ chuẩn bị arm factory/loss dispatch và family plans; epoch
  orchestration/optimizer v2 chưa triển khai hay chạy.

## 4. Gates cho pilot sau này, không chấm bằng step-zero preflight

Giữ v1: anchor≥49/61, matched both≥8/15, FOUND both≥12/16; mỗi3devempty
sigmoid-max≤M0+1e-6; P1 hơnP2≥2hits hoặc pairs, không kém metric còn lại;
input/freeze/identity/bypass; median sidecar overhead≤20% cùng setup.

P1−C1 phải có binding improvement ở hits hoặc pairs và metric còn lại không
giảm để hỗ trợ H_peak. Báo fixed/broken IDs, family CI, từng5train/3devempty,
mass/logits/latency. Train-negative evidence tăng vẫn phải công bố, không dùng
để chọn lại checkpoint. Các gate lợi ích chỉ áp dụng **checkpoint dev-selected
sau pilot có học**, không gọi preflight đạt binding hoặc đủ mở verifier.

CI dev vẫn conditional on selected checkpoint; một seed, negative families ít,
planned scene intent khác observed masks là giới hạn đã audit. Không đổi mask,
truth, denominator,49→48 hoặc tăng epoch/hyperparameter sau xem kết quả.

## 5. Machine-readable lock và phạm vi

Config riêng: `new/configs/anchor_peak_pilot_v2_20261005.json`.
Code riêng: `new/src/pcrau/anchor_peak_experiment.py`.
Preflight xuất locked_protocol/config/init/family plans và hash provenance ở
folder mới, không sửa pilot v1 hoặc active bundle. Không dùng checkpoint
zero-init/preflight làm learned model candidate.

Chưa train, IID/OOD, fit calibration, verifier, capture hoặc robot. File lock
không tự cấp quyền optimizer; pilot v2 chỉ bắt đầu khi được giao riêng.
