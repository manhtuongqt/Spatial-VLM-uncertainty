# S1 — Pilot peak-aware v2: C1/P1/P2

Ngày **05/10/2026**. **Đã hoàn thành orchestration và pilot; gate nâng cấp FAIL.**

**Kết luận chính:** P1 đạt48/61anchor,8/15swap và12/16FOUND-both, bằng C1.
Peak-aware objective chưa tạo thêm ca binding đúng trên dev và chưa đạt49/61.
P1 hơn P2 nhưng bốn ca cải thiện vẫn chỉ nằm ở hai family có anchor lemon.
Không có cơ sở mở verifier từ kết quả này; giữ nguyên baseline/active profile.

## 1. Code, lock và orchestration đã chạy

- [Epoch runner v2](../new/src/pcrau/anchor_peak_runner.py) gọi đúng
  `PeakArm.loss`, không gọi nhầm runner loss v1 cho P1/P2.
- [Evaluator/gates](../new/src/pcrau/anchor_peak_pilot.py) hỗ trợ đủM0/C1/P1/P2;
  giữ matched subset visible+same-cache, không lọc theo phrase-change.
- [Orchestration](../new/scripts/run_anchor_peak_pilot.py): dev mỗi epoch,
  best save, early stopping, best reload và verify metrics, paired/family reports,
  logits/loss curves, latency và freeze/hash checks.
- [5 runner tests](../new/tests/test_anchor_peak_pilot.py) đã PASS: actual toy
  optimizer dispatch/freeze/reload; reject thiếu authorization/hash/config/init;
  selection tie/patience; objective-control và từng empty gate; family bootstrap.
  [Receipt tests](../new/outputs/pcrau_s1_anchor_peak_v2_runner_checks_20261005.json).
- [Report script](../new/scripts/report_anchor_peak_pilot.py) tái tính paired/gates/CI
  từ saved traces, không chạy mô hình hoặc train thêm.

[Protocol v2 đã hash](S1_ANCHOR_PEAK_PROTOCOL_V2_LOCK_20261005.md) và
[config preflight](../new/configs/anchor_peak_pilot_v2_20261005.json) không đổi.
[Execution addendum](S1_ANCHOR_PEAK_PILOT_V2_EXECUTION_LOCK_20261005.md) ghi quyền
train được giao ở bước này và orchestration thực tế. Các flags “chưa triển khai/
không optimizer” trong config/receipt cũ mô tả trạng thái **preflight**, không
mô tả pilot hiện tại. Execution receipt mới được ghi trước optimizer thật.

Data, augmentation, init, batch plans và ngân sách khớp lock:64family train,
256câu horizontal ×4prefix =1.024presentations;4family/batch,16batches/epoch.
Dev chọn checkpoint trên64câu horizontal/16family; toàn400dev dùng để kiểm tra
baseline/bypass. Không warm-start từ M1/M2. C1 anchor span+lossv1; P1 anchor
span+lossv1+0,10peak; P2 whole text+cùng lossP1. Margin1, empty mean-BCE
weight0,25; seed24082026, AdamW3e-4/wd1e-3/clip1, max15epoch/patience5.
MLP/lossFP32, frozen capture/head AMPbfloat16. Arm order C1→P1→P2, frozen
captures/supervision chia sẻ trong epoch, early stopping độc lập từng arm.

Selection đúng lexicographic dev hits→matched pairs→negative mean-empty
sigmoid-max; hòa giữ sớm. Không chọn bằng train, latency hoặc việc đạt gate.

## 2. Checkpoint được chọn bằng dev

| Arm | Best epoch | Đã chạy epoch/steps | Dừng vì | Anchor /61 | Swap /15 | FOUND-both /16 |
|---|---:|---:|---|---:|---:|---:|
| M0 | — | — | Frozen | 44 | 6 | 11 |
| C1 | 11 | 15/240 | Max15 | 48 | 8 | 12 |
| P1 | 8 | 13/208 | Patience5 | 48 | 8 | 12 |
| P2 | 1 | 6/96 | Patience5 | 44 | 6 | 11 |

C1 đã **train lại** từ init và best checkpoint hash trùng learned M1 lịch sử;
không copy checkpoint M1 để thay control. Đây là kiểm chứng replay điều kiệnv1.
P1 chạm48hits ởepoch5, C1 ởepoch6; cả hai không vượt48 trong ngân sách.
Đạt48sớm hơn không thay tiêu chí endpoint hoặc chứng minh đạt gate.

| Arm | Anchor train /251 | Swap train /51 | FOUND-both train /57 |
|---|---:|---:|---:|
| M0 | 209 | 29 | 44 |
| C1 | 227 | 39 | 49 |
| P1 | 226 | 38 | 49 |
| P2 | 210 | 29 | 44 |

Target FOUND hit giữ14/16dev và52/57train; branch chỉ thay anchor shadow,
không thay target/answerability/risk outputs. Ca8pixels không có center vẫn
nằm trong251visible train, không relabel empty/drop family. P1/P2 skip riêng
ranking4presentations/epoch; giữ dense visible1004 và empty20presentations.

## 3. Paired corrections/regressions và CI

| So sánh | Dev fixed/broken | Dev net hits | Train fixed/broken | Train net hits |
|---|---:|---:|---:|---:|
| P1−C1 | 0/0 | 0 | 1/2 | −1 |
| P1−P2 | 4/0 | +4 | 17/1 | +16 |
| P1−M0 | 4/0 | +4 | 18/1 | +17 |

P1−C1 không có ca dev đổi trạng thái hit, không chỉ là tổng accuracy hòa nhau.
Còn13devmisses; bốn ca được sửa soM0/P2 giống C1:

| Sample ID | Anchor phrase | M0/P2 peak px | C1/P1 peak px |
|---|---|---|---|
| `v211dev_family_000030__relation_counterfactual` | lemon | [290,150] | [150,270] |
| `v211dev_family_000154__clean` | lemon | [390,170] | [290,90] |
| `v211dev_family_000154__depth_corruption` | lemon | [390,170] | [290,90] |
| `v211dev_family_000154__occlusion_view_counterfactual` | lemon | [390,170] | [290,90] |

Train P1−C1 sửa `v211dev_family_000391__occlusion_view_counterfactual` nhưng
làm sai `v211dev_family_000060__clean` và
`v211dev_family_000200__occlusion_view_counterfactual`. Không bỏ regressions.

Bootstrap5.000resamples theo family, seed đã khóa:

| So sánh/trục | Dev delta pp | Dev CI95% pp |
|---|---:|---|
| P1−C1 anchor/swap/FOUND-both | 0/0/0 | [0;0] mỗi trục |
| P1−P2 anchor | +6,5574 | [0;18,0328] |
| P1−P2 swap | +13,3333 | [0;33,3333] |
| P1−P2 FOUND-both | +6,25 | [0;20] |

P1−C1 train anchor delta−0,3984pp, CI[−1,9685;0,8032]pp; swap−1,9608pp,
CI[−6,25;0]pp. Full results có trong summary/family CSV. CI là conditional on
dev-selected checkpoints, chưa selection-corrected; train CI không phải bằng
chứng generalization. Dev P1−C1 CI[0;0] do observed hit labels trùng, không
chứng minh hai mô hình tương đương trên quần thể. Dev P1−P2 CI chạm0 và lợi
ích tập trung hai family; chưa chứng minh noun binding tổng quát.

## 4. Mỗi ca anchor rỗng — sigmoid-max

| Split / sample | M0 | C1 | P1 | P2 |
|---|---:|---:|---:|---:|
| train /000023 relation_counterfactual | 0,995801 | 0,999596 | 0,998866 | 0,995801 |
| train /000061 relation_counterfactual | 0,309024 | 0,300746 | 0,284576 | 0,368406 |
| train /000072 relation_counterfactual | 0,214691 | 0,577019 | 0,479504 | 0,214691 |
| train /000173 relation_counterfactual | 0,332852 | 0,073696 | 0,085099 | 0,378919 |
| train /000287 relation_counterfactual | 0,557364 | 0,975577 | 0,985936 | 0,694303 |
| dev /000392 clean | 0,725649 | 0,090093 | 0,150029 | 0,735642 |
| dev /000392 depth_corruption | 0,479991 | 0,075858 | 0,061876 | 0,495606 |
| dev /000392 occlusion_view_counterfactual | 0,848972 | 0,706596 | 0,785309 | 0,851953 |

IDs viết ngắn trong bảng đều có prefix `v211dev_family_` và suffix sau`__`.
CSV lưu đủ ID, maxlogit và mean-zero-BCE. P1 giảm3/5train peaks soC1, tăng2/5;
vẫn tăng3/5soM0:000023,000072,000287. Tại000287, P1 maxlogit4,25 soC1
3,6875/M0 0,23046875; mean-zero-BCE0,088067235 so0,061304663/0,015907538.
Đây là regression thật, không chỉ sigmoid saturation.

Mean train-empty sigmoid-max: M0 0,481946;C1 0,585327;P1 0,566796;P2 0,530424.
Mean dev-empty: M0 0,684871;C1 0,290849;P1 0,332404;P2 0,694400.
P1 cả3devempties thấp hơnM0 nên qua missing gate; nhưng chỉ1/3thấp hơnC1,
mean dev cũng kém C1. **Không gọi empty handling cải thiện toàn diện.** Ba
devempties thuộc cùng một family, không phải ba family độc lập.

## 5. Gate và chi phí

| Gate | Kết quả P1 |
|---|---|
| Technical/input/freeze/bypass | PASS |
| Anchor≥49/61 | **FAIL:48** |
| Swap≥8/15 | PASS:8 |
| FOUND-both≥12/16 | PASS:12 |
| Mỗi dev empty≤M0+1e-6 | PASS |
| P1 hơnP2≥2hits hoặc pairs, metric kia không giảm | PASS:+4hits,+2pairs |
| P1 cải thiện binding soC1 | **FAIL:0hits,0pairs** |
| Median sidecar overhead≤20% | PASS |

Median/P95ms: M0 **5,705/5,956**; C1 **6,374/6,683**; P1 **6,353/6,586**;
P2 **6,364/6,659**. Overhead11,72%/11,35%/11,54%. Setup cached-feature
sidecar+parser, batch8/warm20/runs100/CUDA sync, luân phiên bốn arm; không
gồm backbone/disk/loss, không phải robot/end-to-end latency. Train loop thực
tế44,167giây, chưa gồm initial/final inference, hash checks và reporting.

![Learning curves](../new/outputs/pcrau_s1_anchor_peak_pilot_v2_20261005/paired_report/learning_curves.png)

Ngôi sao là dev-selected epoch. Raw train loss C1 và P1/P2 khác objective,
không so độ lớn trực tiếp để kết luận arm nào tốt hơn. P1 dev empty đi lên sau
epoch8 là quan sát từ curves; chưa đủ kết luận nguyên nhân overfit.

## 6. Checkpoints và bảo toàn

Root: [artifact README](../new/outputs/pcrau_s1_anchor_peak_pilot_v2_20261005/README.md).

| Arm | Checkpoint riêng | SHA-256 |
|---|---|---|
| C1 | `C1/checkpoints/best/mlp.safetensors` | `40e3e26a7e524ca0187f92204d7268ad69f824c9e015370ea2b6dd1160aa36c9` |
| P1 | `P1/checkpoints/best/mlp.safetensors` | `2bbff4da63d49d115f98a5dfd72db05ef90b200e9e6f96a586fa4c216e294347` |
| P2 | `P2/checkpoints/best/mlp.safetensors` | `b2762ed979f1b6d7e7f9eacf01a54aa85671c741783c45ffdd872e0a5e165f14` |

Ba best checkpoints reload khớp **exact dev selection metrics**. Baseline
weights/stats/grad và mọi400dev baseline tensor outputs giữ nguyên; inactive
maps exact, hooks không sót.1627protected hashes trongrun không đổi, gồm active
bundle/dataset/preflight/historical artifacts. Source snapshot và lock lưu trước
optimizer; report tái tính paired/gates/CI từ saved traces,125original runfiles
được hash và giữ nguyên khi tạo paired report. Protocol/config preflight hash
vẫn lần lượt`0c4efd90…a412`/`f53e0f9c…c7e88`.

Không IID/OOD, calibrate/MC/verifier/capture/robot hoặc chỉnh LaTeX. Calibration
chỉ kiểm tra bundle file hash, không đọc samples/fit. Checkpoint shadow không
được promote và không làm risk/answerability đã chọn tốt hơn.

## 7. Kết luận và điểm dừng

**H_peak chưa được hỗ trợ trên dev theo protocol này.** Implementation/loss
gradient đúng không bảo đảm representation+frozen head học được instance binding.
Lợi ích noun span còn xuất hiện dưới matched objective, nhưng hẹp ởlemon.
Peak penalty có hiệu ứng một số train negatives nhưng vẫn còn regressions;
không đổi gate49→48, kéo dài epoch hoặc tự tune loss weight để cứu kết quả.

Nếu tiếp tục, bước phù hợp là audit từ logits đã lưu:13devmisses còn lại,
train regressions và000287negative; kiểm tra giới hạn biểu diễn/frozen head và
gradient conflict trước khi đề xuất cơ chế mới. Chưa tự chạy audit/intervention
mới trong báo cáo này. Không mở geometric verifier khi binding gate chưa đạt.
Peak-aware loss là grounding/evidence supervision, không phải aleatoric/
epistemic estimator hoặc một kết quả uncertainty-calibration mới.
