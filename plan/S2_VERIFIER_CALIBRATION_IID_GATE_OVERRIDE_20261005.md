# S2 — Verifier, calibration và old-IID reevaluation theo ngoại lệ gate

Ngày **05/10/2026**. **Đã triển khai verifier evidence-only, fit calibration và
hoàn thành reevaluation offline trên IID cũ.** Primary P1_G44 nhận 330 ca đúng và
33 lỗi/363 quyết định, risk 9,09%; baseline 327 đúng và34 lỗi/361, risk 9,42%.
Cải thiện point estimate nhỏ; CI vẫn chạm0 và **budget 7,5% chưa đạt**.

Người dùng cho tiếp tục dù binding gate chưa đạt. Ghi rõ **gate override**,
không sửa pilot48/61 thànhPASS. Ngoài ra live Adapter numerical parity gateFAIL
trênIID; kết quả dưới dùng cached 33 baseline evidence đã khóa, không tuyên bố
live end-to-end parity hoặc promote active profile.

## 1. Code, protocol và scope đã triển khai

- [Verifier runtime](../new/src/pcrau/horizontal_verifier.py): prompt-only parser,
  target distribution và predicted anchor logits; không mask/ID/truth/variant.
- [Risk fusion/calibration](../new/src/pcrau/verifier_risk.py): schemas33/41/44,
  family CV/fit-all vàbundle/schema guards.
- [8 unit checks](../new/tests/test_horizontal_verifier.py) PASS: direction/margin,
  diffuse maps, bypass/unsupported, immutable arrays/oracle rejection,
  nonfinite handling, evaluator-field invariance vàbundle mismatch.
- [Calibration/live-parity runner](../new/scripts/run_verifier_calibration_iid.py).
- [Cached-IID continuation](../new/scripts/reevaluate_verifier_iid_cached.py):
  dùng profiles đã khóa, không gọi fit/calibrator selection sauIID.
- [Read-only reporting](../new/scripts/report_horizontal_verifier.py).

[Protocol ngoại lệ](S2_VERIFIER_GATE_OVERRIDE_PROTOCOL_20261005.md),
[config](../new/configs/horizontal_verifier_evidence_20261005.json),
[numerical preflight](S2_VERIFIER_NUMERICAL_PREFLIGHT_20261005.md) và
[cached-evidence contract](S2_IID_CACHED_EVIDENCE_REEVALUATION_20261005.md).

Phạm vi: một anchor với left_of/right_of, image frame, margin 12 px; direct bypass,
depth/ternary/multi-clause unsupported. Signed margin tính từpredicted grid
peaks. Pair compatibility tính tổng P_target(i)P_anchor(j) trên pairs thỏa margin.
Đây là normalized-map score, không xác suất quan hệ đúng hoặc presence.
Trace luôn binding/presence UNVERIFIED, không GEOMETRY_VERIFIED.
**Target MAP/Adapter inputs/outputs cached không đổi; không hard refine.**

## 2. Frozen variants vàcalibration

Primary P1_G44 được chọn **trước calibration**, dùngP1 best epoch 8; C1 best 11
vàM0 anchor cũ làm controls. Không đổi neural weights hoặc chọn lại nhánh theo IID.

| Model | Features | Ý nghĩa |
|---|---:|---|
| B33 | 33 | Refit baseline, tái hiện active logistic33 |
| P1_A41 | 41 | Thêm 3 parser flags và5 anchor evidence, chưa geometry |
| M0_G44 | 44 | Thêmgeometry từanchor cũ |
| C1_G44 | 44 | Geometry với anchor lossv1 |
| P1_G44 | 44 | **Primary**:geometry vớianchor lossv2 |

Geometry extras gồm signedpeakmargin/640, peakcompatible vàpaircompatibility.
Anchor extras gồm maxlogit/sigmoidmax/entropy/softmaxpeak/displacement vsM0.
Ngoài scope, extras geometry/anchor bằng0 vàparser flags giữ status; trace geometry
null, không giả định unsupported là direct. Runtime vector bỏ evaluation/variant.

Calibration 1000 mẫu/200 family, 5-fold family, standardization trong training fold,
logistic L2=0,01 cố định mọi variant. B33 tái hiện coefficients/mean/scale/intercept
active calibrator trong1e-12 vàcùng family folds. Policy hard_found, budget 7,5%,
Wilson upper 95%≤7,5%, ít nhất60 accepted family. Ngưỡng chọn OOF → áp fit-all;
**khóa 5 profiles trước lượt IID đầu của runner**, không fit lại sau IID parity failure.

| Model | Ngưỡng OOF | NhậnOOF | LỗiOOF | Nhậnfit-all | Lỗifit-all |
|---|---:|---:|---:|---:|---:|
| B33 | 0,261193283453 | 346 | 16 | 351 | 14 |
| P1_A41 | 0,276521446463 | 352 | 16 | 355 | 17 |
| M0_G44 | 0,276960950121 | 352 | 16 | 354 | 17 |
| C1_G44 | 0,272815935319 | 350 | 16 | 353 | 14 |
| P1_G44 | 0,288964361193 | 351 | 16 | 357 | 17 |

Mỗi profile nhận 199 family OOF. P1_G44 OOF empirical4,5584%, Wilson upper7,2757%.
Fit-all counts là resubstitution, khôngvalidation độc lập. Calibration valid 411,
IID valid 405, không trộn mẫu số. Active threshold0,2611932834526145 giữ nguyên;
B33 refit khác ởroundoff~1e-16, replay IID decisions/counts giốngbaseline.

## 3. IID cũ: kết quảprofiles đã khóa

1000 mẫu/200 family, **reevaluation đã quan sát**, khôngfresh untouched test.
Không dùng IID chọn features/nhánh/ngưỡng; ablations đều được báo, khôngpromote
model nào vì metric tốt hơn. Eventerror=(truth≠FOUND)OR(MAPngoài target).

| Model | Nhận | Đúng | Lỗi | Coverage | Risk | Valid recall /405 |
|---|---:|---:|---:|---:|---:|---:|
| B33/official baseline | 361 | 327 | 34 | 36,10% | 9,42% | 80,74% |
| P1_A41 | 363 | 329 | 34 | 36,30% | 9,37% | 81,23% |
| M0_G44 | 363 | 330 | 33 | 36,30% | 9,09% | 81,48% |
| C1_G44 | 361 | 329 | 32 | 36,10% | 8,86% | 81,23% |
| P1_G44 | 363 | 330 | 33 | 36,30% | 9,09% | 81,48% |

P1_G44 nhận thêm3ca đúng, giảm1lỗi soB33; soP1_A41 cùngcoverage, thay1ca
lỗi bằng1ca đúng. SoC1_G44 nhận thêm1đúng **và1lỗi** trong2quyết định thêm;
khôngdominance C1. M0_G44 cócùng aggregate utility vớiP1_G44, nên utility này
chưa chứng minh phrase-conditioned/peak loss lànguồn cải thiện.

Primary Wilson upperIID12,4922%, empirical9,0909%>7,5%. **Budget vẫnFAIL**.
Không lấy threshold từ đườngrisk–coverage để giảm risk hồi tố.

| Model | Brier | NLL | ECE10 | AURC |
|---|---:|---:|---:|---:|
| B33 | 0,077108 | 0,257066 | 0,031080 | 0,255748 |
| P1_A41 | 0,076986 | 0,255624 | 0,032693 | 0,255540 |
| M0_G44 | 0,076347 | 0,254789 | 0,031544 | 0,255768 |
| C1_G44 | 0,076057 | 0,253965 | 0,029838 | 0,255604 |
| P1_G44 | 0,075631 | 0,252410 | 0,030122 | 0,255313 |

Primary cải thiện nhẹprobability metrics soB33, nhưng khôngtốt nhấtECE trong
mọi comparator vàkhông đạt budget. AURC tính toàn 1000 mẫu, prevalence error59,5%;
không áp chuẩn phổ quátAURC<0,1. Hình dưới lọc hard_found eligibility, khôngcùng
eventpopulation vớiAURCtoàn tập.

![Frozen-profile risk–coverage](../new/outputs/pcrau_horizontal_verifier_iid_cached_20261005/report/risk_coverage.png)

## 4. Paired changes vàCI theo200family

P1_G44−B33: thêm4valid accepts, mất1valid accept; loại2errors, thêm1error.
Không chỉ báo net. Nhữngquyết định đổi ngoài horizontal scope vẫn cóthểđến từ
refit coefficients/parser flags; không gọi mọi case được sửa làgeometry reasoning.

Geometry ablationP1_G44−P1_A41 đổi đúng 2 sample, đều horizontal:

- Từ chối `v211iid_family_000058__occlusion_view_counterfactual`, trước là lỗi nhận.
- Nhận `v211iid_family_000119__clean`, một ca valid trước bị từ chối.

Bootstrap5000 seed24082026 giữmọi variants trongfamily; profiles cố định, không
fit lại/ngưỡng lại trongbootstrap:

| P1_G44−B33 | Point pp | CI95% pp |
|---|---:|---|
| Coverage | +0,20 | [−0,30;+0,80] |
| Correct accepts/1000 | +0,30 | [−0,10;+0,80] |
| Selective risk | −0,3274 | [−1,2937;+0,4768] |
| Valid recall | +0,7407 | [−0,25;+1,8827] |

P1_G44−P1_A41 risk−0,2755pp, CI[−0,8357;0]pp; correctacceptance rate+0,10pp,
CI[0;+0,30]pp. P1_G44−C1_G44 risk **+0,2266pp**, CI[−0,0634;+0,7723]pp.
MọiCI này chưa cho bằng chứng thắng rõ; đây làfixed-profile reevaluation, không
bao gồmuncertainty từmodel selection/calibration fit hoặc khắc phục priorIIDuse.

## 5. Binding/scope vàgrounding giữ nguyên

Dev 400: 64 horizontal, 140 direct, 196 unsupported. Calibration/IID mỗi 1000:
160 horizontal, 350 direct, 490 unsupported. Verifier không hỗ trợ 84% ca với geometry;
các ca đó vẫnđi qua baseline vàriskcalibration theo flags đã khóa.

| Split / horizontal visible | M0 anchor hit | C1anchor hit | P1anchor hit | Empty |
|---|---:|---:|---:|---:|
| Dev /61 | 44 | 48 | 48 | 3 |
| Calibration /159 | 131 | 140 | 140 | 1 |
| IID /153 | 117 | 132 | 131 | 7 |

P1 tốt hơnanchorM0 trênIIDsubset, nhưng kém C1 một ca; không dùng sốIIDnày để
chọn lạicheckpoint. Peakrelationcompatible khôngchứng minh identity đúng.
TargetPIT giữ775/835, FOUNDvalid405/420; cachedanswerability accuracy81,10%,
macroF1 0,809686 giữbaseline. Regionkhôngđổi, khôngfit/claimregion mới.
Chưa benchmark verifier end-to-end latency, chưarobot success hoặc MC.

## 6. Numerical parity vàprovenance — giới hạn quan trọng

Đã dừng guard vàgiữartifact/source snapshots ởhaiinitialfailedruns; khôngxóa:
initialdev delta1,4314e-5; runr2calibrationdelta1,3873e-5 trước fit. Devprobe400mẫu
chỉmộtcase drift, target/sourceexact. Numerical tolerance2e-5 đượcchốt trước fit/
IID. Runr3 fit/freeze xong, IID case000052relationdelta8,1539e-5 vượtbound,
**live numerical gateFAIL, không nới tolerance sauIID**.

Cached continuation dùngchínhcalibrators/profiles r3, byte/hash khôngđổi, không
refit theoIID. Trên 1000 IID: target/sourceexact; Adapter drift ở 2 case family000052,
max8,1539e-5; argmax không đổi. Risk vector dùng**cached 33 baseline fields đãhash**,
không thay bằng recomputed probabilities. Geometry/anchor inputs khôngnhậnAdapter
probabilities nên đượcđánh giá trên target đúng artifact vàpredicted anchors.

Kết quả chứng minh utility trong**cached baseline evidence + new geometry**,
chưachứng minh parity/full deployment củađường live mới. Nguyên nhân numerical
drift cụthể chưa được xác định. Khôngđánh tráo gateFAIL thànhPASS đểhoàn tất.

3300 protected files trongcachedrun giữ hash; models/residual states khôngđổi,
không optimizer/neural train. Activebundle/calibrator/profiles vàold artifacts
bảo toàn. Source/code/schema khóa vàsnapshots lưu trước fit; profiles khóa trước
lượt IID đầu, cachedcontinuation khôngđổi lựa chọn sau failure.

## 7. Artifact vàkết luận

- [Cached-IID artifact README](../new/outputs/pcrau_horizontal_verifier_iid_cached_20261005/README.md).
- [Calibrationprofiles r3](../new/outputs/pcrau_horizontal_verifier_gate_override_20261005_r3/calibration/profiles.json).
- [IIDfull summary](../new/outputs/pcrau_horizontal_verifier_iid_cached_20261005/summary.json).
- [ComparisonCSV](../new/outputs/pcrau_horizontal_verifier_iid_cached_20261005/report/comparison.csv).
- [Paireddecision changes](../new/outputs/pcrau_horizontal_verifier_iid_cached_20261005/test_iid/paired_decision_changes.jsonl).

**Có cơ chế geometry tường minh và trace thật; lợi ích risk fusion còn nhỏ, chưa đạt
đích utility/budget hoặcgiải quyết binding.** Risk 44 làestimated composite error
probability cócalibrationrecipe, không risk robot. Paircompatibility/anchorentropy
vẫn là evidence proxies, không aleatoric/epistemic decomposition hoặccausal nguồn
bất định. Không promote C1/P1/geometry theo IID. Nếu phát triển tiếp, xử lý live
parity vàbinding bằngtrain/dev; không mining/tune module từ IID rồi gọi test mới.
