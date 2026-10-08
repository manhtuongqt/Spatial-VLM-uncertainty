# S1 — Failure audit sau pilot: binding, evidence và conditioning span

Ngày: **05/10/2026**. Nối tiếp [pilot M0/M1/M2](S1_ANCHOR_SHADOW_PILOT_20261005.md).
Thực hiện theo yêu cầu audit của người dùng; **không train, backward, optimizer,
chọn checkpoint mới, calibration, IID/OOD hoặc verifier**.

## 1. Kết luận chính

**Parser/conditioning không chỉ là nhãn gắn lên output:** giữ nguyên câu, ảnh,
frozen capture và checkpoint M1, thay pooling anchor span bằng target span làm
hit trên original anchor mask dev giảm **48/61→31/61**; pooling toàn câu giảm
**48/61→44/61**. Nhánh thật sự phản ứng với lựa chọn span, nhưng chưa bảo đảm
peak tìm đúng instance.

**13 lỗi dev không phải chỉ lệch biên mask hoặc lưới quá thô.** Cả 13 anchor
mask đều có 2–12 cell centers nằm trong vật; mọi MAP cell sai đều không giao
anchor mask. Khoảng cách MAP tới mask **53,08–596,27 px**. Các lỗi gồm nhầm
target, vật khác, instance cùng class hoặc nền; 6 ca có raw sigmoid-max ≥0,9.

**Empty loss chưa kiểm soát được peak sai:** trong 3/5 train negatives có peak
tăng, mean zero-BCE **cũng tăng**. Không có bằng chứng cho lời giải thích rằng
loss đã giảm và chỉ một peak lọt qua. Một nguyên nhân có thể kiểm chứng tiếp là
objective dense hiện tại chưa trực tiếp ràng buộc dominant peak đúng danh tính
hoặc phải thấp khi không có visible anchor.

Gate v1 vẫn **FAIL: 48/61 <49/61**. Đã viết
[đề xuất protocol v2](S1_ANCHOR_PEAK_OBJECTIVE_V2_PROPOSAL_20261005.md) với một
auxiliary objective trực tiếp trên peak; **chưa triển khai hoặc chạy v2**.

## 2. Ranh giới và kiểm chứng

Nguồn run:
`new/outputs/pcrau_s1_anchor_shadow_pilot_20261005/`.
Audit cuối:
`new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/`.

- Phân tích logits thật của selected M0/M1/M2; không đổi raw confidence.
- Frozen replay theo đúng thứ tự/batch8/AMP: **toàn bộ 656 mẫu** (256 train
  original-prefix + 400 dev) tái tạo target/M0/M1/M2 anchor tensors chính xác.
  320 câu horizontal được chấm diagnostics; 336 câu dev khác chỉ replay/bypass.
- Mask, registry, IDs và semantic segmentation chỉ được mở sau khi đã lưu
  diagnostic predictions. Không đi vào capture/pool/MLP/head.
- M1 residual và M2 residual được load từ best, giữ eval và requires_grad=False;
  baseline weights/stats và hai residual states không đổi trước/sau.
- Giữ nguyên `q_old`, fused features và contextualized text tokens. Chỉ thay
  mask pooling của residual M1; không viết lại prompt/token IDs hoặc relation.
- Inactive/direct/unsupported bypass exact. Không đưa anchor mới vào graph,
  Adapter hoặc decision; không có risk/answerability improvement mới.

RGB/mask của đủ **22 focus cases** đã được xem trực tiếp qua bốn contact pages,
gồm 13 misses, 5 train empties và 4 dev corrections. Xem
[trang 1](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/inspection/page_01.png),
[trang 2](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/inspection/page_02.png),
[trang 3](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/inspection/page_03.png),
[trang 4](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/inspection/page_04.png).
Mỗi case còn có panel RGB + M0 raw logits + M1 raw logits với **shared color
scale trong cùng case**, không dùng chung scale giả giữa các case.

## 3. Bảng 13 dev misses

IDs dưới đây rút gọn `v211dev_family_`; suffix variant được viết đủ. Tọa độ là
MAP pixel-center ảnh 640×480. Rank là thứ hạng logit tốt nhất trong các ô có
**center nằm trong đúng anchor mask**, ties dùng `1 + số logit lớn hơn`.

| ID rút gọn | Anchor yêu cầu | M1 peak nằm trên | MAP | Sigmoid-max | Cách mask, px | Best đúng rank |
|---|---|---|---|---:|---:|---:|
| 000042__clean | green cube | blue cube, requested target | (350,230) | 0,99142 | 117,39 | 14 |
| 000042__depth_corruption | green cube | blue cube, requested target | (350,190) | 0,99819 | 135,33 | 10 |
| 000042__occlusion_view_counterfactual | green cube | yellow cube | (270,150) | 0,99996 | 224,66 | 25 |
| 000081__relation_counterfactual | mango | apple, requested target | (150,150) | 0,99840 | 148,64 | 12 |
| 000178__clean | lemon | nền/label không thuộc registry | (70,50) | 0,00629 | 596,27 | 147 |
| 000178__depth_corruption | lemon | nền/label không thuộc registry | (70,50) | 0,00382 | 596,27 | 160 |
| 000178__occlusion_view_counterfactual | lemon | mustard bottle | (290,150) | 0,12085 | 354,99 | 24 |
| 000241__relation_counterfactual | tomato soup can | nền/label không thuộc registry | (470,450) | 0,22001 | 315,32 | 7 |
| 000383__relation_counterfactual | apple | orange | (290,190) | 0,99993 | 53,08 | 5 |
| 000388__clean | lemon | apple, requested target | (150,50) | 0,12085 | 525,95 | 58 |
| 000388__depth_corruption | lemon | apple, requested target | (150,50) | 0,17106 | 525,95 | 49 |
| 000388__occlusion_view_counterfactual | lemon | apple, requested target | (150,50) | 0,17329 | 525,95 | 52 |
| 000395__relation_counterfactual | apple instance01 | apple instance03 | (310,310) | 0,99983 | 112,25 | 3 |

Phân loại mô tả, không phải causal attribution:

| Peak sai nằm ở đâu? | Số ca |
|---|---:|
| Trong requested target instance | 6 |
| Vật khác class, ngoài requested target | 3 |
| Cùng class nhưng sai instance | 1 |
| Nền hoặc semantic label không có registry identity | 3 |

Chỉ **1/13** peak nằm trong valid-target mask chính thức, nhưng **6/13** nằm
trong requested target instance lấy từ candidate IDs/semantic labels. Hai event
khác nhau: ca ABSENT có thể có vật được nhắc tới nhưng không có valid target.
Đây là lý do thống kê pilot `wrong_anchor_inside_target=1` không mô tả hết nhầm
vai. Semantic registry identity là hậu kiểm, không phải output model mới.

Các ca điểm cao là family042 (3 variants), 081, 383 và395. Bảy ca còn lại có
score ≤0,22001; softmax vẫn có MAP dù mọi raw logits có thể rất thấp. Không
fit threshold missing/identity từ các score này.

## 4. Không chỉ nhầm noun: instance và evidence của cảnh

### 4.1. Cùng class vẫn có thể sai anchor

Family395 có hai apple thấy rõ: annotation anchor là `ycb_apple_01`, M1 peak
trên `ycb_apple_03`. Anchor span chỉ chứa “apple”, không có instance ID. Câu đầy
đủ còn cần relation/context để chọn đúng apple. Vì vậy **noun grounding đúng
class không tự giải quyết instance disambiguation**; không quy toàn bộ lỗi
cho parser hoặc chỉ tăng semantic noun similarity.

Best center đúng đứng rank3, softmax mass có trọng số fractional mask khoảng
0,302475, nhưng wrong-instance peak vẫn thắng. Xem
[case395 với hai peak apple](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/cases/v211dev_family_000395__relation_counterfactual.png).
Family383 apple/orange cũng có correct-mask logit rank5, mass khoảng0,134707;
chưa chứng minh thêm margin loss sẽ sửa, nhưng đã có competing evidence đáng
để thí nghiệm objective.

### 4.2. Metadata scene intent khác observed pixels

**10/13** misses có anchor ID không nằm trong `object_oracle.active_scene_ids`,
nhưng actual semantic labels và anchor mask có pixel của ID đó. RGB cho thấy
green cube/anchor khác còn trong cảnh lưu trữ, hoặc lemon/tomato soup can chỉ
lộ ở rìa phải. Ví dụ family042 `state_submode=anchor_absent`, nhưng green cube
thực sự nhìn thấy và mask có2056 pixels; family178/388 lemon có424–558 pixels
ở rìa, mỗi mask chỉ có2 centers trên grid.

Đây là **mâu thuẫn giữa planned active-set/absence intent và observed visible
support**, không phải mask rỗng. Mask và semantic label ID khớp pixel-exact ở
mọi focus case; không có bằng chứng mask bị lệch registration trong các ca này.
Không dùng active-set metadata để xóa mask, bỏ mẫu khỏi mẫu số hoặc sửa truth.
Truth/candidate/valid-target labels của archive giữ nguyên; audit này chưa
kiểm định lại toàn bộ answerability labeling của dataset.

Điểm này giới hạn cách nói “anchor vắng mặt”: mô hình có thể tập trung vào vật
chính, trong khi metric yêu cầu đúng noun instance còn nhìn thấy ở vùng lưu
trữ. Chưa được lấy observed mask hiện diện làm bằng chứng truth FOUND. Cũng
không được coi source/ABSENT là nhãn absence của anchor.

## 5. Vì sao gain chỉ tập trung ở lemon?

| Visible lemon | M0 | M1 anchor | M1 dùng target span | M1 dùng toàn câu |
|---|---:|---:|---:|---:|
| Train | 30/39 | 38/39 | 22/39 | 30/39 |
| Dev | 0/10 | 4/10 | 0/10 | 0/10 |

Train không chỉ sửa lemon: M1−M0 có +8 lemon, +3 mustard bottle, +3 yellow
cube, +2 orange, +1 green cube, +1 purple cube. Các phrase khác không tăng.
Vì vậy không đúng khi nói MLP chỉ học được duy nhất lemon; **dev gain quan
sát được** mới tập trung ở lemon.

Dev bốn gains thuộc family030 (1) và154 (3 variants). Lemons được đặt rõ trong
vùng cảnh chính, có9–10 grid centers; M1 đưa peak từ target/competing object về
đúng lemon. Sáu lemon misses còn lại thuộc family178/388, chỉ2 centers, mask
ở rìa, anchor không thuộc intended active-set. M1 phần lớn **hạ evidence**, chứ
không tìm được lemon: family388 sigmoid-max từ khoảng0,90–0,96 xuống0,12–0,17,
nhưng peak vẫn trên apple; family178 clean/depth chuyển sang nền với score rất
thấp. Xem [lemon ở rìa](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/cases/v211dev_family_000178__clean.png).

Giải thích phù hợp bằng chứng: M1 có noun-selective response rõ ở các ca lemon
được bố trí chính; chưa robust với tiny/peripheral anchors và competing
instances. **Chưa cô lập nguyên nhân** do visual features, role/query prior,
loss, kích thước mask hay layout. Không kết luận lemon là tính mới hoặc lợi ích
depth/occlusion tổng quát từ hai family.

## 6. Năm train empties: logits chứng minh điều gì?

Các ca là relation_counterfactual, năm family độc lập. Tất cả anchor masks khớp
semantic-label image và bằng zero. BCE trong bảng là `mean softplus(logit)`
trên đủ768 ô, đúng zero-map BCE của objective v1, không phải train total loss.

| Family | Anchor | M1 peak trên | M0→M1 sigmoid-max | M0→M1 mean zero-BCE | Kết quả |
|---|---|---|---|---|---|
| 000023 | orange cube | yellow cube, requested target | 0,995801→0,999596 | 0,039772→0,067922 | Cả peak/loss tăng |
| 000061 | tomato soup can | nền/label không thuộc registry | 0,309024→0,300746 | 0,002740→0,002195 | Giảm |
| 000072 | mango | tomato soup can | 0,214691→0,577019 | 0,002513→0,002959 | Cả peak/loss tăng |
| 000173 | orange | apple | 0,332852→0,073696 | 0,002389→0,000281 | Giảm |
| 000287 | mustard bottle | tomato soup can | 0,557364→0,975577 | 0,015908→0,061305 | Cả peak/loss tăng |

Mean zero-BCE qua năm ca: **0,012664→0,026932**, hơn gấp đôi. Family023 có số
ô sigmoid≥0,9 từ4→10; family287 từ0→5, số ô logit dương từ2→21; không chỉ một
pixel nhiễu. Xem
[case287](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/cases/v211dev_family_000287__relation_counterfactual.png).

Tất cả checkpoint mới vẫn được chọn bằng tuple dev đã khóa, không dùng train
negative để chọn lại. So loss M0 với M1 ở đây là diagnostic; M0 không được
train theo objective v1, và logits saved chỉ là prefix gốc, không mọi prefix
hay mọi batch ở thời điểm optimizer.

### 6.1. Mismatch objective–peak là có thật, nhưng chưa phải nguyên nhân duy nhất

Zero BCE của một peak có gradient `sigmoid(logit)/768` ở mức mean một sample;
với weight0,25, hệ số tối đa là khoảng **0,0003255** trước mean thêm trên nhóm
empty. Đây là gradient theo **logit**, không phải norm gradient tham số; không
từ đó kết luận shared MLP nhận gradient nhỏ hơn visible objective bao nhiêu.

Negatives có20/1024 train presentations =1,95%, vẫn chỉ5 family. Tùy epoch,
chỉ **2–5/16 batches** có empty term kháczero. Epoch11 được chọn có4/16 batch
chứa empty; logged mean empty BCE qua batch là0,008727, contribution weight0,25
khoảng0,002182 so total mean0,407966. Cách batch-mean này khác mean BCE của5
original-prefix selected predictions ở bảng trên; không đánh tráo hai mẫu số.

Dense mean-loss dilution, ít negatives, tradeoff của shared residual và dev
selection là các **giả thuyết cạnh tranh**, chưa causal attribution. Có thể thử
penalty trực tiếp trên dominant peak, nhưng không mặc định nó sửa được binding
hay uncertainty. Không thêm presence threshold/classifier từ5 train/1 dev
negative family ở vòng audit này.

## 7. Conditioning-span intervention

Định nghĩa modes:

| Mode | Checkpoint/residual | Pooling | Phần giữ nguyên |
|---|---|---|---|
| M0 | Frozen baseline | Không residual | Câu, ảnh, capture |
| M1_anchor | M1 best epoch11 | Anchor token span | q_old, F, text encoder |
| M1_target | **Cùng M1** | Target token span | Tất cả ngoài pooling mask |
| M1_whole | **Cùng M1** | Toàn valid text | Tất cả ngoài pooling mask |
| M2_whole | M2 best epoch1 | Toàn valid text | Frozen baseline/capture |

Target span được parse từ **cùng prompt**, không phải sửa câu thành yêu cầu mới.
Vẫn dùng frozen **anchor head**. Các hit sau can thiệp dưới đây đo trên original
anchor mask; không đánh giá các maps ấy như predicted target mới.

| Hit original anchor mask | M0 | M1_anchor | M1_target | M1_whole | M2_whole |
|---|---:|---:|---:|---:|---:|
| Train visible251 | 209 | 227 | 159 | 209 | 210 |
| Dev visible61 | 44 | 48 | 31 | 44 | 44 |

| M1 intervention so anchor span | Train | Dev |
|---|---|---|
| Target span: peak thay đổi / mọi horizontal | 142/256 | 35/64 |
| Target span: original-anchor hit sửa / làm sai | 3 /71 | 0 /17 |
| Whole text: peak thay đổi / mọi horizontal | 74/256 | 22/64 |
| Whole text: original-anchor hit sửa / làm sai | 1 /19 | 0 /4 |

M1_whole khác M2_whole: một bên dùng M1 đã học với anchor span, bên kia dùng
MLP đã học với toàn câu. Không coi can thiệp test-time này là đối chứng train
matching budget hay model mới được chọn. Đây là can thiệp ngoài conditioning
distribution đã train của M1, nên suy luận nguyên nhân phải giới hạn.

Trong focus cases, family383/395 dùng target span chuyển peak về yellow cube
được nhắc làm target; family178 clean/depth chuyển từ nền về tomato soup can.
Family042 và081 vẫn peak trên target ngay cả với anchor span; không phải mọi
ca đổi span đều đổi object. Điều này bác bỏ cách giải thích đơn giản “parser
chọn span nhưng nhánh hoàn toàn bỏ qua nó”, nhưng **không chứng minh object
retrieval tổng quát hoặc explicit spatial reasoning đã hoàn thành**.

Can thiệp không sửa được ca nào trong13 dev misses theo original anchor label.
Không cherry-pick can thiệp để thay production map. Câu có “apple” với hai
instances vẫn phải giải quyết relation/context, chưa có geometric verifier.

## 8. Đề xuất v2 được rút ra từ audit

Ưu tiên thử **một auxiliary peak-aware objective**, giữ nguyên parser,
conditioning, kiến trúc33.024 params, frozen head, dataset và ngân sách:

- Visible: ràng buộc peak trên đúng anchor thắng peak ngoài anchor.
- Empty: phạt trực tiếp max logit, thay vì chỉ trông vào mean zero-BCE.
- Giữ BCE+Dice/mean-empty loss v1; thêm một term case-conditioned, không đồng
  thời thêm presence head, MC, verifier, sampling mới hay tăng capacity.

Đây là giả thuyết objective mismatch, không phải kết luận chắc chắn về nguyên
nhân. Có3 lỗi FOUND với correct cells rank3/5/12 đáng chấm riêng, nhưng không
chọn train subset từ các lỗi dev này. Tiny/peripheral anchors và thông tin
instance còn có thể là giới hạn visual/query representation mà loss không sửa.

Audit thêm toàn subset tìm **1 train visible-mask có8 pixels nhưng không có
cell center trong mask**: `v211dev_family_000077__relation_counterfactual`.
Không biến nó thành empty hoặc bỏ mẫu. Protocol v2 phải skip riêng visible
peak-ranking term khi positive-center set rỗng, giữ original BCE+Dice supervision
và mọi metric/mẫu số. Dev61 visible masks đều có center support.

Chi tiết đối chứng, công thức, gates và stop rules nằm trong file đề xuất v2.
Chưa chạy bất kỳ thí nghiệm v2; không nới gate49 xuống48 hoặc mở verifier.

## 9. Artifacts và tính trung thực của provenance

| Artifact cuối | Link |
|---|---|
| Summary frozen audit | [summary.json](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/summary.json) |
| Observable conditioning predictions | [conditioning_predictions.jsonl](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/conditioning_predictions.jsonl) |
| Raw diagnostic logits | [conditioning_logits.npz](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/conditioning_logits.npz) |
| Evaluator stats cho320 horizontal | [conditioning_evaluator.jsonl](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/conditioning_evaluator.jsonl) |
| Case classification | [failure_classification.csv](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/failure_classification.csv) |
| Train negatives chi tiết | [train_empty.jsonl](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/train_empty.jsonl) |
| Epoch loss/batch audit | [loss_history_audit.jsonl](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/loss_history_audit.jsonl) |
| Hash/input/state protection | [provenance.json](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/provenance.json) |
| Visual inspection record | [visual_inspection.json](../new/outputs/pcrau_s1_anchor_failure_audit_20261005_r2/visual_inspection.json) |
| Reproducible script | [audit_anchor_shadow_failures.py](../new/scripts/audit_anchor_shadow_failures.py) |

Lần đầu ở folder không hậu tố hoàn tất diagnostics/assertions nhưng lỗi khi
ghi provenance cuối (`sha256_file` nhận str thay vì Path). Giữ nguyên folder,
source snapshot và `RUN_STATUS.json` đánh dấu incomplete, sửa lỗi xuất artifact
rồi chạy lại **frozen diagnostic**, không train lại. Run `_r2` có provenance
đầy đủ; summary/raw predictions/logits/evaluator số liệu hai lần đã đối chiếu
khớp. Không dùng summary incomplete làm báo cáo cuối.

Audit không sửa archive, baseline modules, parser/wrapper, loss v1, checkpoints,
active profile, gate protocol hoặc LaTeX. Nội dung v2 là đề xuất cho nhiệm vụ
tiếp theo, không phải kết quả hoặc lệnh tự khởi động training.
