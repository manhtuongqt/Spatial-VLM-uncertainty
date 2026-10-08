# Kế hoạch nâng cấp P-CRA-U tham khảo GCA và FUSE — giữ nguyên dataset

Ngày lập: **05/10/2026**.

**Ưu tiên hiện hành theo yêu cầu mới của người dùng:** đưa cơ chế suy luận không
gian và hiệu chuẩn risk có định nghĩa, chạy xuyên suốt, truy vết được vào kiến
trúc đồ án nhanh. Cải thiện kết quả cũ là lợi ích thêm, không phải điều kiện
hoàn thành tích hợp. Gate49/61 và các gate utility/budget được giữ để báo cáo,
không chặn tích hợp; kết quả âm không sửa thành PASS. Các đoạn yêu cầu thắng
baseline hoặc phải đạt gate ở thiết kế ban đầu bên dưới không còn là blocker.

**Trạng thái S6 — uncertainty nhiệm vụ đã tích hợp, 07/10/2026:** đã train
ba auxiliary heads87.199params trên train/dev, MC20 ở trained fusion/task heads,
fit calibration60 riêng và reevaluate IID cũ. **337đúng+34lỗi/371nhận**, coverage
37,10%,risk9,16%; live44 trước330+33/363,risk9,09%. Brier0,072483 so0,075631;
CI risk/Brier/NLL chứa0, budget7,5% vẫn chưa đạt. Năm nhóm nay là closed-set
semantic node identity, spatial maps, left/right relation, depth mixture và
reference-support completion dưới occlusion; không phân rã năm nguồn nhân quả,
full amodal hoặc full-VLM aleatoric/epistemic.16MC/task features thật sự vào risk,
khác S5 source-head-only companion. Supplement124train/32dev pairs (chỉ9/6hidden
đáng kể) dùng ảnh archive cùng camera/cảnh; preview6cặp đã hiển thị. RGB-D thật
và feature-cache task/risk outputs exact ở1dev observation. Baseline/active
profile/LaTeX/robot/OOD giữ nguyên. Xem [kết quả S6 và giới hạn](S6_TASK_UNCERTAINTY_RESULTS_20261007.md),
[protocol](S6_TASK_UNCERTAINTY_PROTOCOL_20261007.md), [bundle release](../new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/bundle_release.json).
Các phần7–8 bên dưới là thiết kế ban đầu; S6 dùng scope hẹp nêu trong báo cáo,
chưa hoàn thành full candidate-instance posterior hoặc full amodal model.

**Trạng thái S5 — MC năm nhãn nguồn, 07/10/2026:** theo yêu cầu mới, đã
triển khai nhánh MC thật trên 1.600 train/400 dev, T=20, dropout p=0,1 của
source head đã học. Mỗi nhãn semantic/relation/spatial/depth/occlusion có MC
mean score, std, predictive/expected entropy và MI. Đây là source-head-only
sampling điều kiện trên representation baseline cố định; spatial vẫn là weak
label AMBIGUOUS. Không gọi MI là năm nguyên nhân độc lập hoặc toàn bộ epistemic
uncertainty của VLM. Nhánh MC là output đồng hành, chưa vào risk 44 features
hoặc calibrator live; các kết quả IID S4 vẫn thuộc pipeline không MC. Không
train/fit hoặc mở IID mới, không đổi active profile. Xem [phép tính và câu trả
lời cho thầy](FIVE_SOURCE_MC_METHOD_20261007.md), [protocol S5 khóa trước run](S5_FIVE_SOURCE_MC_PROTOCOL_20261007.md)
và [artifact MC](../new/outputs/pcrau_source_mc_train_dev_20261007/README.md).
F1 không cải thiện đồng đều: occlusion giảm 0,543779 → 0,533937; kết quả dev
là mô tả dữ liệu đã quan sát, chưa chứng minh MC luôn nhận biết được lỗi.

**Trạng thái mới S4 — 07/10/2026:** đã reevaluate IID cũ bằng chính bundle live
r2, forward mới toàn 1000 mẫu/200 family và không lấy cached 33 prediction evidence
cho primary. Kết quả 330 đúng/33 lỗi/363, coverage 36,30%, risk 9,09%; baseline
327 đúng/34 lỗi/361, risk 9,42%. Anchor P1 trúng 131/153 visible so M0 117/153,
paired sửa 16/broken 2. CI utility chứa 0, budget 7,5% vẫn chưa đạt. Model,
calibrator và threshold 0,2889643687106893 giữ nguyên, không fit/tune theo IID.
Xem [tổng hợp cuối cho đồ án](SPATIAL_VARIANT_FINAL_SUMMARY_20261007.md),
[protocol S4](S4_LIVE_IID_PROTOCOL_20261007.md) và
[kiểm tra artifact cuối](../new/outputs/pcrau_unified_spatial_iid_20261007/final_verification.json).
16 checks PASS, 1000 runtime decisions replay khớp, 12 case thật đã xem; 3121 protected
files giữ hash. Kết quả nay có provenance của đường live; S2 cached vẫn là
lịch sử riêng. Active profile/LaTeX/robot/OOD giữ nguyên.

**Trạng thái S3 trước lượt IID live:** đã hoàn thành [inference live xuyên suốt](S3_UNIFIED_SPATIAL_INFERENCE_20261005.md)
và [tài liệu kiến trúc](../new/docs/UNIFIED_SPATIAL_INFERENCE.md).
RGB-D train thật đi qua backbone, anchor P1, verifier và logistic44 khớp với
feature-cache path của cùng observation; risk lấy44features từ forward hiện tại.
13checks PASS; toàn train1600/dev400 và calibration1000 đã chạy; bundle reload
đạt kiểm tra. Calibrator/profile mới khóa riêng, ngưỡng0,2889643687106893.
S3 chưa mở IID; không gán số S2 cached path cho bundle live. Không thay active
profile, không robot/MC/hard MAP refinement/LaTeX. Bước tích hợp đã hoàn thành.

**Trạng thái S2 trước khi ghép live:** đã triển khai verifier evidence-only, fit calibration và reevaluate IID cũ theo ngoại lệ gate được người dùng giao. Primary P1_G44 nhận330đúng/33lỗi/363, risk9,09%; baseline327/34/361, risk9,42%. Budget7,5% chưa đạt, CI chạm0; binding gate48/61 và live numerical parity vẫnFAIL. Kết quả S2 dùng cached33baseline evidence đã khóa + verifier mới. Active profile không đổi. Các cập nhật dưới là lịch sử theo thời điểm; ngoại lệ gate không sửa kết quả pilot thànhPASS, không mở robot/MC/MAP refinement hoặcLaTeX.

**Cập nhật triển khai S0 ngày 05/10/2026:** theo yêu cầu tiếp theo của người dùng, đã hoàn thành audit frozen train/dev. Xem [báo cáo S0](S0_ANCHOR_BINDING_AUDIT_20261005.md) và [contract verifier v0](S0_VERIFIER_INPUT_CONTRACT_20261005.md). Anchor hiện tại chưa đủ cơ sở cho hard MAP refinement; chưa train lại, fit calibration, thêm MC hoặc chạy IID/robot. Các giai đoạn sau vẫn là đề xuất.

**Cập nhật S1 ngày 05/10/2026:** [parser audit](S1_QUERY_PARSER_AUDIT_20261005.md) khớp phrase–token–annotation **1.020/1.020 câu trong scope** trên toàn train/dev; 980 câu ngoài phạm vi trả unsupported. Đã kiểm tra 6.800 presentation và ghi [thiết kế phrase-conditioned anchor với gate trước train](S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md). Parser chưa nối vào baseline forward; chưa triển khai/train nhánh MLP hoặc verifier. Binding cải thiện vẫn là giả thuyết cần thí nghiệm.

**Cập nhật bước kiểm chứng anchor tiếp theo ngày 05/10/2026:** đã triển khai wrapper residual M1/M2 và chuẩn bị runner riêng; xem [báo cáo preflight](S1_ANCHOR_SHADOW_PREFLIGHT_20261005.md). Kiểm tra 1.024 train presentation + 400 dev ở cả FP32/bfloat16: step-zero error 0, mọi baseline output exact; backward chỉ MLP; latency overhead M1/M2 khoảng 10%, dưới gate 20%. Chưa tạo optimizer hoặc chạy pilot; chưa có bằng chứng binding cải thiện, chưa verifier/calibration mới. Design v1 được bảo toàn làm protocol khóa.

**Cập nhật pilot có học ngày 05/10/2026:** đã evaluate mỗi epoch, lưu/reload best residual, early stop và xuất paired report. [Báo cáo M0/M1/M2](S1_ANCHOR_SHADOW_PILOT_20261005.md): M1 best epoch11 đạt **48/61 anchor hits, 8/15 matched pairs, 12/16 FOUND both**; M2 best epoch1 đạt **44/61, 6/15, 11/16**. M1 sửa4/broken0 dev nhưng chỉ hai family, đều lemon; CI theo family chạm0. **Gate tổng FAIL** do thiếu anchor hit; chưa mở verifier. Empty dev giảm từng ca, nhưng train empty tăng ở3/5 ca. Báo cáo r2 sửa mẫu số matched train 50→51 theo protocol từ traces, không train/chọn checkpoint lại, giữ artifact/source gốc. Baseline/dataset/active profile không đổi; chưa IID/calibration/MC.

**Cập nhật failure audit ngày 05/10/2026:** [báo cáo audit](S1_ANCHOR_FAILURE_AUDIT_20261005.md) đã xem22case thật, hậu kiểm13devmisses và5trainempties, frozen replay656mẫu exact. Cùng M1 checkpoint, đổi pooling anchor→target span làm hit dev48→31, anchor→whole48→44: có span sensitivity thật, chưa bảo đảm đúng instance. Cả13misses có grid-center support, không phải chỉ lệch biên;6peak trên requested target,1sai apple instance,3vật khác,3nền/unmapped. Cả3trainnegative tăng max cũng tăng meanBCE. Ghi riêng10misses có planned active-set không chứa anchor nhưng observed pixels vẫn có, không sửa nhãn. [Protocol v2 đề xuất](S1_ANCHOR_PEAK_OBJECTIVE_V2_PROPOSAL_20261005.md) thử một auxiliary peak-aware objective, chưa triển khai/train; v1 gate vẫnFAIL, chưaverifier/IID/calibration.

**Cập nhật peak-aware v2 ngày 05/10/2026:** [báo cáo preflight](S1_ANCHOR_PEAK_V2_PREFLIGHT_20261005.md) PASS11kiểm tra; visible margin1 và empty softplus(max logit), weight0,10 giữ dense loss. C1/P1/P2 cùng init33.024params, zero-step error0 trên1.024train presentations+400dev ở FP32/bfloat16; baseline exact, gradient chỉ MLP. Ca8pixels vẫn visible, skip riêng ranking. [Protocol v2 khóa](S1_ANCHOR_PEAK_PROTOCOL_V2_LOCK_20261005.md) và [config](../new/configs/anchor_peak_pilot_v2_20261005.json) đã lưu trước backward, chưa optimizer/train. Epoch orchestration v2 chưa triển khai; binding49/61 vẫn chưa được chứng minh, chưa verifier/calibration/IID.

**Cập nhật pilot C1/P1/P2 ngày05/10/2026:** [báo cáo v2](S1_ANCHOR_PEAK_PILOT_V2_20261005.md) đã hoàn tất5runner tests, optimizer lock, dev mỗi epoch, best reload và paired report. C1 best11/chạy15epoch, P1 best8/dừng13, P2 best1/dừng6; C1 vàP1 cùng48/61,8/15,12/16, P2 44/61,6/15,11/16. P1−C1 dev fixed0/broken0; P1−P2 fixed4/broken0 vẫn chỉlemon ởhai family, family CI chạm0. P1 giảm3/5trainempty peaks soC1 nhưng vẫn tăng3/5soM0; dev3empty đều giảmsoM0,2/3kémC1. Gate49/61 và objective-improvement FAIL, không mở verifier. C1 train lại tái hiện exact checksum historicalM1; active profile/dataset/baseline giữ nguyên. [Execution lock](S1_ANCHOR_PEAK_PILOT_V2_EXECUTION_LOCK_20261005.md) bổ sung orchestration/quyền train, không sửa config/protocol preflight lịch sử.

**Cập nhật verifier/calibration/IID ngày05/10/2026:** theo yêu cầu “mở verifier/IID/calibration luôn”, đã ghi [protocol ngoại lệ](S2_VERIFIER_GATE_OVERRIDE_PROTOCOL_20261005.md) và [báo cáo S2](S2_VERIFIER_CALIBRATION_IID_GATE_OVERRIDE_20261005.md). Verifier chỉ left/right image-frame, evidence-only, binding/presenceUNVERIFIED.5calibrators33/41/44features đượcfit family-crossfit vàfreeze trước lượtIID mới, không chọn model/ngưỡng theoIID. Geometryablation sửa1falseaccept+thêm1validaccept soP1_A41; primary soB33 +3correct/−1error nhưng chưa qua budget vàCI. M0_G44 cócùngaggregate utility; chưa chứng minh peakloss lànguồn cải thiện. Numericalguard phát hiện liveAdapter delta8,1539e-5>2e-5 trênIID; giữFAIL, không nới tolerance/refit bằngIID. [Cachedcontinuation](S2_IID_CACHED_EVIDENCE_REEVALUATION_20261005.md) dùng profilesr3 nguyênbyte/hash; target/sourceexact, Adapterargmaxkhôngđổi, risk33cached. Chưa códeployment/liveparity đầyđủ hoặcpromote activeprofile.

## 0. Đích đến và tiêu chí hoàn thành



### 0.1. Đích đến trong một câu

**Xây dựng một biến thể P-CRA-U trên cùng dataset và backbone đóng băng, có parser target–relation–anchor, anchor điều kiện hóa theo noun span, kiểm chứng ràng buộc không gian trên predictions và hiệu chuẩn xác suất lỗi nhận thức; tất cả chạy trong cùng đường inference và xuất trace kiểm chứng được.**

Đích đến bắt buộc hiện hành là **cơ chế có định nghĩa, thực sự tham gia risk/decision và trace kiểm chứng được**, cùng báo cáo thực nghiệm trung thực. Lợi ích so baseline là mục tiêu bổ sung; kết quả âm vẫn hoàn thành tích hợp. Có thêm tên module hoặc scalar mà không tham gia phép tính chưa đủ. Binding được đo bằng hậu kiểm, không mặc định chính xác khi parser đúng.

### 0.2. Hệ thống cuối cùng phải làm được gì?

Với đầu vào RGB/depth và một câu thuộc phạm vi hỗ trợ, hệ thống phải:

1. **Hiểu cấu trúc yêu cầu:** xác định target phrase, anchor phrases, predicate của từng clause và hệ quy chiếu; báo rõ câu unsupported hoặc binding chưa đủ tin cậy.
2. **Tìm evidence từ ảnh:** dự đoán target/anchors từ đầu vào observable; không lấy mask hoặc object ID oracle để điền vào trace.
3. **Thực hiện phép kiểm tra quan hệ:** tính mức tương thích hình học trên predictions, giữ bằng chứng tuyệt đối và phát hiện clause conflict/anchor yếu. Kết quả này thực sự tham gia sửa lựa chọn vị trí hoặc quyết định từ chối.
4. **Đo giới hạn của dự đoán:** ghi độ mơ hồ vị trí và evidence quan hệ; phân biệt tín hiệu được đo với cách diễn giải chưa được chứng minh. Model disagreement/MC là mở rộng tùy chọn, không bắt buộc ở biến thể tích hợp hiện hành.
5. **Ước lượng đúng event lỗi:** risk cho trường hợp truth non-FOUND hoặc MAP sai, fit sau freeze; xuất quyết định nhận thức theo policy đã khóa.

Ví dụ đích đến hiện hành: với “apple right of purple cube”, hệ thống phải thể hiện được nó đang so target với anchor dự đoán cho cụm “purple cube” trong hệ tọa độ ảnh. Margin, mức tương thích và evidence tuyệt đối của anchor thực sự vào phép tính risk, từ đó có thể ảnh hưởng quyết định nhận/từ chối. Nhánh S3 không sửa target MAP hoặc hard veto. Trace giữ presence/binding UNVERIFIED; softmax peak không chứng minh có vật hay bind đúng vật.

### 0.3. Đích đến của uncertainty là gì?

**Mục tiêu bắt buộc hiện hành là calibrated predictive error risk từ observable evidence, có evidence hình học thực sự vào phép tính và quyết định.** Phải có đánh giá calibration/đối chứng và công bố cả thất bại; thắng baseline không còn là điều kiện chặn. Bất đồng mô hình là mở rộng tùy chọn, chưa triển khai ở S3.

Các tên gọi được nhắm tới là “mơ hồ định vị”, “thiếu/mâu thuẫn evidence quan hệ”, “model disagreement” và “risk đã hiệu chuẩn”. Đây là định nghĩa kiểm chứng được trong phạm vi đồ án.

**Phân rã thuần aleatoric/epistemic chưa phải deliverable được bảo đảm của kế hoạch này.** Nếu yêu cầu cuối cùng là hai đại lượng riêng mang đúng nghĩa thống kê ấy, cần mở một mục tiêu nghiên cứu bổ sung về noise/generative model, giả thiết nhận dạng và đánh giá riêng. MC Dropout + entropy + logistic, dù đạt metric tốt, vẫn chưa đủ để tuyên bố đã hoàn thành phân rã đó. Tham khảo FUSE không tự giải quyết khoảng trống này.

### 0.4. Đích đến thực nghiệm và các con số mong muốn

Baseline đang nhận **327 ca đúng và 34 ca lỗi trong 361 quyết định**, coverage **36,10%**, risk **9,42%**. Mục tiêu ứng dụng là cải thiện đánh đổi này, không chỉ tăng F1 hoặc làm heatmap sắc hơn.

| Trục | Mục tiêu cần chứng minh |
|---|---|
| Cơ chế reasoning | Parse/binding/geometry có metric và trace; ablation cho thấy verifier đóng góp vào vị trí hoặc quyết định |
| Grounding | Giữ chất lượng tổng thể và FOUND-only trong mức regression đã khóa trên dev; công bố corrections/regressions thay vì chỉ báo tổng PIT |
| Utility | Ít lỗi hơn tại coverage tương đương, hoặc nhận thêm ca đúng ở mức risk tương đương; có paired uncertainty theo family |
| Uncertainty | Evidence mới có ích ngoài answerability/entropy; báo cả probability quality và ranking tại vùng nhận |
| Chi phí | Latency/memory phù hợp budget chọn trước trên dev; lợi ích đủ bù phần MC/verifier thêm vào |

**Mốc định lượng mong muốn, không chặn tích hợp:** duy trì ít nhất **327 ca nhận đúng**, đồng thời đưa empirical selective risk IID xuống **≤7,5%**, với ngưỡng được chọn hoàn toàn trên calibration. Đây là đích bổ sung, chưa phải kết quả, cam kết hoặc bảo đảm thống kê.

Ví dụ để hiểu mốc: nếu vẫn chấp nhận 361 quyết định, cần **không quá 27 lỗi và ít nhất 334 ca đúng** để empirical risk ≤7,5%. Không dùng ví dụ này để chọn ngưỡng trên IID hoặc ép số nhận bằng thứ hạng test. Coverage thực tế của profile đã freeze có thể khác; matched-coverage analysis là phép so sánh bổ sung, không phải runtime profile được tune bằng test.

Nếu risk giảm bằng cách từ chối thêm nhiều ca đúng, chưa đạt đích utility ở trên. Nếu empirical risk đạt 7,5% nhưng CI còn vượt budget, chỉ được công bố đạt mốc thực nghiệm tại point estimate, chưa được tuyên bố đã bảo đảm budget. Nếu MAP mới thay đổi, báo lại tập valid; không mặc định mẫu số vẫn là 405.

### 0.5. Sản phẩm cuối cùng phải bàn giao

- Một **bundle biến thể riêng đã khóa**: model/config, parser/verifier, feature schema, calibrator và profiles; baseline hiện tại được bảo tồn. MC configuration chỉ cần nếu mở thí nghiệm MC sau này.
- Một **inference output có trace**: query đã parse, predicted anchors, compatibility/status, MAP trước/sau, uncertainty evidence, calibrated risk và decision. Trace phản ánh phép tính thật, không phải lời giải thích sinh sau để hợp thức hóa prediction.
- Một **báo cáo đối chứng B/R/U/R+U và ablation**, trên cùng dataset/split, có paired metrics theo family, failure cases và chi phí tính toán.
- Một **kết luận về định nghĩa**: cơ chế nào đã triển khai/đánh giá, uncertainty nào chỉ là proxy, quan hệ/frame nào chưa được hỗ trợ và mốc nào chưa đạt.

### 0.6. Khi nào được gọi là hoàn thành?

**Hoàn thành nghiên cứu/triển khai:** đã có bundle chạy đúng contract, kiểm tra kỹ thuật, protocol freeze–calibration–reevaluation, báo cáo đối chứng và kết luận trung thực. Nếu giả thuyết thất bại, kết quả âm vẫn kết thúc được vòng nghiên cứu nhưng không được gọi là nâng cấp thành công.

**Hoàn thành tích hợp theo ưu tiên mới:** có đường RGB-D/features + prompt →
predictions → parser/verifier → evidence trực tiếp → calibrator → risk/decision/trace,
đã kiểm tra train/dev và calibration khớp producer, bundle khóa, case thật và
tài liệu. S3 đã đạt mức này; reevaluation IID live hoặc robot là việc riêng,
không tự thêm thành điều kiện còn thiếu của yêu cầu tích hợp vừa giao.

**Nâng cấp thành công:** có đủ bằng chứng cho explicit relational grounding trong phạm vi hỗ trợ, uncertainty evidence thực sự hữu ích, và cải thiện đánh đổi nhận đúng–lỗi–chi phí so baseline theo tiêu chí đã khóa. Chỉ promote nếu người dùng lựa chọn sau khi xem artifact.

**Phạm vi đích đến hiện tại là quyết định nhận thức offline.** Robot motion/pick-and-place, full scene graph, reasoning 3D tổng quát và phân rã aleatoric/epistemic thuần cần protocol/bằng chứng bổ sung; không được suy là đã hoàn thành từ bundle offline này. Test-IID giữ nguyên và đã quan sát, nên kết luận cuối vẫn phải ghi reevaluation.

## 1. Quyết định đề xuất

Nâng cấp theo hai hướng bổ trợ:

1. **GCA-inspired relational grounding:** biến câu lệnh thành ràng buộc target–relation–anchor có hệ quy chiếu rõ ràng; kiểm chứng bằng hình học dự đoán; dùng kết quả kiểm chứng để điều chỉnh định vị và cung cấp bằng chứng thiếu/mâu thuẫn quan hệ.
2. **FUSE-inspired uncertainty evidence fusion:** kết hợp chất lượng quan sát, độ mơ hồ định vị và bất đồng mô hình; hiệu chuẩn thành xác suất lỗi của quyết định nhận thức.

Giữ backbone đóng băng, dữ liệu và split hiện có. Tạo biến thể nghiên cứu riêng; P-CRA-U đã chọn tiếp tục là baseline. Bắt đầu bằng module nhỏ có thể kiểm chứng, chỉ thay Sidecar/Adapter nếu development audit chứng minh cần thiết.

**Không đồng nhất phương án này với tái hiện nguyên bản GCA hoặc FUSE.** Parser, geometric verifier, MC Dropout và logistic calibration dưới đây là thiết kế thích nghi cho đồ án. Chưa có bằng chứng chúng cải thiện kết quả.

## 2. Nguồn paper và phần được tham khảo

### 2.1. GCA

**Geometrically-Constrained Agent for Spatial Reasoning**, Zeren Chen và cộng sự, CVPR 2026. Bản arXiv đầu tiên ngày 27/11/2025; không mô tả đây là bài lần đầu xuất hiện năm 2026.

- [Trang chính thức CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Geometrically-Constrained_Agent_for_Spatial_Reasoning_CVPR_2026_paper.html).
- [arXiv và lịch sử phiên bản](https://arxiv.org/abs/2511.22659).
- [Phương pháp, mục 3.2–3.3 của bản arXiv](https://arxiv.org/html/2511.22659v1).

GCA tách hình thức hóa hệ quy chiếu/mục tiêu khỏi thu nhận và tính toán hình học theo ràng buộc. Phần tham khảo là sự tách biệt đó và việc gắn ký hiệu với đối tượng trước tính toán. Hệ agent đầy đủ có điều phối công cụ, tái dựng 3D và sinh code; không cần mang toàn bộ vào đồ án. Công thức compatibility ở mục 6 là đề xuất riêng, không phải công thức GCA.

### 2.2. FUSE

**FUSE: Quantifying Uncertainty in Vision-Language Models by Bayesian Fusing Epistemic and Aleatoric Uncertainty**, Harry Zhang và Luca Carlone, ICML 2026, PMLR 306.

- [Trang xuất bản chính thức](https://proceedings.mlr.press/v306/zhang26j.html).
- [arXiv](https://arxiv.org/abs/2606.14728).
- [Phương pháp, mục 4 của bản arXiv](https://arxiv.org/html/2606.14728v1).

FUSE học probabilistic image–text embeddings bằng Gaussian Process, lấy bằng chứng phân tán từ response embeddings của các câu trả lời sinh ngẫu nhiên, rồi kết hợp Bayesian và calibration. P-CRA-U thiếu các biểu diễn/cơ chế lấy mẫu tương ứng. Mượn cách kết hợp bằng chứng đầu vào và bất đồng đầu ra; không chuyển nguyên posterior hoặc giả thiết thống kê của bài sang heatmap. MC Dropout cộng logistic là một biến thể lấy cảm hứng, không phải FUSE nguyên bản.

### 2.3. Phạm vi đọc và mức tin cậy

Đã tham khảo các mục phương pháp nêu trên và đối chiếu code/artifact nội bộ. Chưa tái hiện thực nghiệm hoặc kiểm chứng toàn bộ giả thiết/định lý của paper. Kết quả trên benchmark của paper không phải dự báo mức cải thiện của P-CRA-U.

## 3. Baseline phải bảo tồn

Nguồn quyết định hiện hành: [active_experimental_profile.json](../new/outputs/active_experimental_profile.json), [MODEL_SELECTION.md](../new/docs/MODEL_SELECTION.md) và [bàn giao 05/10](SESSION_CONTEXT_20261005.md).

**P-CRA-U hiện tại = V2 đóng băng + Adapter answerability `detail_cost`, train-only language augmentation + logistic33.** Bản trước Adapter gọi là P-CRA-U V2 lịch sử.

Root bundle: `new/outputs/pcrau_answerability_language_dev_20261003/`.

| Thành phần | Đường dẫn trong root |
|---|---|
| Config | `config.json` |
| Checkpoint | `detail_cost/checkpoints/best/model.safetensors` |
| Calibrator | `calibration_full_fit/selected_calibrator.json` |
| Profiles | `calibration_full_fit/profiles.json` |
| Kết quả IID | `test_iid_7p5/full_metrics.json`, `test_iid_7p5/SUMMARY.md` |

Policy hiện hành: predicted `FOUND` và risk ≤ **0.2611932834526145** thì `EXECUTE` ở mức nhận thức. Ngân sách calibration **7,5%**, selective risk IID **34/361 = 9,42%**; budget chưa đạt trên test. Người dùng chấp nhận đánh đổi thực nghiệm, không phải bảo đảm an toàn.

| Chỉ số baseline hiện tại | Giá trị và mẫu số |
|---|---|
| PIT target-present | 775/835 = 92,81% |
| RoboRefer đối chứng | 724/835 = 86,71%; delta +6,11 pp, CI [1,78; 10,49] pp |
| FOUND-only MAP đúng | 405/420 = 96,43% |
| Answerability accuracy / macro-F1 | 81,10% / 0,809686 |
| Coverage | 361/1000 = 36,10% |
| Nhận đúng trong ca hợp lệ | 327/405 = 80,74% |
| Composite lỗi trong tập nhận | 34/361 = 9,42% |
| AURC logistic33 / raw Adapter | 0,255748 / 0,251397 |

Các số là kết quả cũ có artifact, không phải kết quả của kế hoạch này. Adapter không thay grounding/source/edge/region. Không thay mô hình bằng support scorer, V3 hoặc checkpoint smoke.

## 4. Vấn đề cơ chế hiện tại

### 4.1. Reasoning đã có nhưng còn gián tiếp

[text.py](../new/src/pcrau/text.py) nhận diện loại quan hệ và số slot anchor từ cú pháp. Chưa có parser đầy đủ gắn từng noun phrase với node/cạnh, xử lý scope và shared anchor.

[model.py](../new/src/pcrau/model.py) có Transformer, learned query slots, fusion theo relation, target/anchor predictions, spatial moments, edge MLP và graph embedding. Không thể nói mô hình hoàn toàn không học quan hệ: relation conditioning và shared training losses có thể ảnh hưởng upstream weights.

Tuy nhiên, trong forward hiện tại:

- Target/anchor logits được tạo trước edge logits.
- Edge logits không điều chỉnh target heatmap và không đi trực tiếp vào graph projection.
- Adapter có thể dùng edge evidence để thay answerability, nhưng đây chưa phải bộ giải hình học có ràng buộc tường minh.
- Edge supervision là proxy; accuracy theo proxy chưa chứng minh kiểm tra đúng từng predicate hình học.

### 4.2. Uncertainty không vô nghĩa, nhưng tên nguồn còn yếu

Risk đã calibration có event cụ thể, nên có ý nghĩa thực nghiệm. Source head là một head đa nhãn năm đầu ra; spatial source là weak label từ AMBIGUOUS. Source scores không phải năm phép đo bất định độc lập hoặc causal attribution.

Spatial softmax là phân bố trên vị trí lưới; entropy/covariance chưa được chứng minh là aleatoric uncertainty. Heatmap sắc vẫn có thể sai, heatmap rộng có thể phản ánh diện tích vật. Chưa có nhánh MC spatial disagreement đã đánh giá trong bundle chọn.

### 4.3. Nút thắt quyết định

34 lỗi nhận hiện tại gồm **22 INSUFFICIENT_EVIDENCE, 7 ABSENT, 5 FOUND nhưng MAP sai**. Occlusion và relation variants chiếm 27/34 lỗi, nhưng đây là phân tầng quan sát, chưa xác định nguyên nhân nhân quả.

Trên toàn IID, composite error là 595/1000: 580 non-FOUND và 15 FOUND có MAP sai. Vì vậy, ranking toàn tập có thể chủ yếu phản ánh answerability. Cần đánh giá riêng định vị trong truth FOUND và lợi ích từ bằng chứng quan hệ. Chỉ sửa MAP chưa đủ xử lý phần lớn lỗi nhận.

## 5. Hợp đồng giữ nguyên dataset

| Tập | Family | Mẫu gốc | Vai trò |
|---|---:|---:|---|
| Train | 320 | 1600 | Học các thành phần cần học |
| Dev | 80 | 400 | Chọn kiến trúc, checkpoint, tham số reasoning/MC |
| Calibration | 200 | 1000 | Fit risk/profile sau model freeze |
| Test-IID cũ | 200 | 1000 | Reevaluation của biến thể đã khóa |

Mỗi family có năm variant; giữ nguyên membership và provenance. Language augmentation 6400 presentation vẫn chỉ 320 family train độc lập.

Các giới hạn:

- Không thu capture, mở OOD, sửa mask/nhãn, đổi family split hoặc làm sạch archive để tăng số.
- Không dùng family/sample/variant ID làm feature.
- Mask, object IDs và relation graph ground truth chỉ dùng ở supervision hoặc evaluator theo split; không vào inference, parser hoặc verifier.
- Không dùng test để sửa grammar, chọn kernel, chọn checkpoint, feature, MC budget, ngưỡng hoặc region mass.
- IID đã được quan sát trong quá trình phát triển; mọi kết quả tiếp theo vẫn là reevaluation, không phải xác nhận độc lập mới.
- Semantic/relation counterfactual có thể thay đổi ý nghĩa và truth. Không ép chúng bất biến như sensor noise.

Cho phép thiết kế representation mới, đọc các trường đã có để audit, dùng annotation train làm supervision và tính metric evaluator mới. Những việc đó không tạo thêm family độc lập hoặc supervision mới từ con người.

## 6. Nhánh reasoning đề xuất

### 6.1. Query có cấu trúc

Parser chỉ đọc prompt và xuất cấu trúc tối thiểu:

```text
target_phrase
anchor_phrases[]
clauses[{predicate, target_ref, anchor_refs, reference_frame}]
parse_status: supported | ambiguous | unsupported
```

Ví dụ:

```text
"locate the apple that is right of the purple cube"
target_phrase = "apple"
anchor_phrases = ["purple cube"]
clause = right_of(target, anchor_0), frame = image
```

Triển khai đầu tiên nên là grammar có phạm vi xác định trên train/dev, có kiểm tra scope và thông báo unsupported. Không dùng ground-truth graph để hoàn thiện parse ở runtime. Không mặc định câu không parse được là `direct`, vì sẽ bỏ mất ràng buộc mà không báo lỗi.

Các quan hệ ternary như `between_in_depth` và `nearer_than_both` phải có biểu diễn nhiều anchor; không thay bằng cạnh nhị phân bất kỳ. Parser nhiều clause chỉ được tuyên bố hỗ trợ khi có dữ liệu/evaluator tương ứng.

### 6.2. Binding noun phrase với predicted anchor

Đây là gate trọng yếu: có một anchor map không có nghĩa map đó chỉ đúng danh từ parser tìm được.

1. Audit ánh xạ phrase → slot → mask trên train/dev bằng annotation evaluator.
2. Thử dùng các slot hiện có nếu binding đạt chất lượng đủ cho reasoning.
3. Nếu slot không gắn đúng danh từ, thêm phrase-conditioned target/anchor queries, tận dụng Transformer và spatial heads hiện có.
4. Huấn luyện thay đổi này trên train bằng target/anchor masks đã có. Danh tính annotation chỉ dùng nối supervision với phrase; runtime chỉ nhận phrase và ảnh/features.

Không mặc định mọi record có metadata đủ để nối phrase–mask chính xác. Kiểm tra coverage và ambiguity trước; không âm thầm tạo association bằng tên file hoặc object ID ở inference.

### 6.3. Hệ quy chiếu và phạm vi hình học

Ưu tiên `left_of/right_of` trong hệ tọa độ ảnh đã thống nhất. Chuẩn hóa tọa độ theo width/height và bảo đảm target/anchor dùng cùng phép resize/crop.

Quan hệ depth chỉ mở sau audit: ý nghĩa relative depth, chiều lớn/nhỏ, registration, invalid values, phạm vi valid và đầu vào tương ứng từng variant. Relative depth không phải mét. Chưa tuyên bố object-centric/world-frame reasoning, khoảng cách Euclidean 3D hay graspability.

Đối với candidate-level reasoning, dùng candidate/centroid dự đoán từ đầu ra mô hình. Top-k modes không tự là instance IDs; evaluator masks không được trở thành candidate detector runtime. Phân biệt phép gần đúng trên điểm lưới với kiểm tra vị trí đối tượng.

### 6.4. Compatibility và điều chỉnh heatmap

Đề xuất thử nghiệm, không phải công thức nguyên bản GCA:

```text
A_l(j) = normalized predicted anchor distribution cho clause l
K_l(i,j) ∈ [0,1] = mức tương thích hình học của predicate l
C_l(i) = Σ_j A_l(j) K_l(i,j)
```

Với trái/phải, kernel có thể là hàm mềm của chênh lệch tọa độ và margin. Tham số độ mềm/margin chọn trên dev, không lấy từ test. Bản hard rule chỉ dùng làm đối chứng có cùng đầu vào.

Một phép điều chỉnh mềm có thể khảo sát:

```text
g(i) = Π_l max(C_l(i), ε)^(λ_l)
Z = Σ_i P_base(i) g(i)
P_reason(i) = P_base(i) g(i) / Z, khi Z an toàn về số
```

`λ_l` là mức ảnh hưởng lựa chọn trên dev; direct query bypass verifier. Tính bằng log-space khi cần ổn định số. Nếu Z quá nhỏ hoặc không hữu hạn, xuất trạng thái invalid/unsupported và fallback đã khai báo; không chia cho một floor rồi coi đầu ra vẫn là distribution tổng một. Đây là score fusion, **không phải phép nhân các evidence độc lập đã được chứng minh Bayesian**: P_base vốn đã được condition theo relation.

Không chỉ trả P_reason. Phải giữ Z, compatibility tuyệt đối, anchor evidence, parser status, disagreement giữa clause và độ dịch chuyển MAP. Nếu mọi vị trí tương thích kém, normalize vẫn có thể tạo một peak sắc; peak ấy không chứng minh quan hệ được thỏa mãn.

### 6.5. Anchor vắng mặt, quan hệ mâu thuẫn và nhiều ứng viên

Softmax luôn có tổng một, kể cả khi không có anchor. Vì vậy, normalized anchor map không được dùng một mình làm xác nhận anchor tồn tại. Lưu cả bằng chứng từ logits/confidence chưa chuẩn hóa, độ tập trung và kiểm tra trên negatives; calibrate mức hữu ích của chúng.

Nhiều clause không được mặc định độc lập. Nếu dùng chung anchor phải tái sử dụng cùng binding. Nếu thiếu anchor, parse unsupported hoặc các clause xung đột, verifier xuất status/evidence để policy cân nhắc từ chối; không dựng quan hệ hợp lệ bằng oracle hoặc fallback im lặng.

Một trace tối thiểu cần cho biết: parser hiểu gì, slot nào gắn với phrase nào, geometry dự đoán nào được dùng, predicate nào được tính, score trước/sau và vì sao MAP/decision thay đổi.

## 7. Nhánh uncertainty đề xuất

### 7.1. Các nhóm evidence

| Nhóm | Ví dụ feature dự kiến | Diễn giải cho phép |
|---|---|---|
| Quan sát | Depth validity nếu có đầu vào hợp lệ; anchor evidence yếu | Chỉ báo chất lượng hoặc thiếu bằng chứng |
| Hình học | Z, clause compatibility, conflict, parser support, MAP shift | Khả năng đáp ứng ràng buộc trên predictions |
| Định vị | Entropy, mode competition, candidate margin, region mass | Mơ hồ vị trí theo representation |
| Mô hình | MC disagreement, sự thay đổi candidate/MAP | Bất đồng của phần mô hình được lấy mẫu |
| Answerability | Probability bốn lớp từ Adapter | Xác suất lớp theo supervision hiện có |

Feature nào không có nguồn observable hợp lệ phải bỏ hoặc đánh dấu missing rõ ràng; không lấy annotation bù vào. Bắt đầu với ít feature có giả thuyết cụ thể, tránh mở rộng vector tùy tiện theo kết quả test.

Source scores cũ có thể là comparator/evidence bổ sung, nhưng không đổi tên chúng thành aleatoric/epistemic hoặc causal source. Không xóa các chiều source khỏi Adapter cũ vì checkpoint đã học với các chiều đó.

### 7.2. MC Dropout có kiểm soát

Giữ input features cố định; không chạy lại backbone cho từng sample. Lấy mẫu dropout ở phạm vi Sidecar đã khai báo, chạy verifier trên từng dự đoán rồi tổng hợp.

- Backbone và các thành phần không lấy mẫu giữ eval.
- Không gọi `.train()` toàn mô hình một cách tùy tiện; forward có nhánh modality dropout khi cấu hình kích hoạt.
- Xét cả dropout nội bộ Transformer/attention, không chỉ các module `nn.Dropout` nhìn thấy trực tiếp.
- Giữ preprocessing và normalization đã fit cố định; không cập nhật weights/stats.
- Ghi seed, số lượt T, phạm vi lấy mẫu và cách tổng hợp; khảo sát T nhỏ trên dev, ví dụ 5/10/20, không coi đây là thông số đã chọn.
- Pilot giữ answerability từ đường deterministic của Adapter cũ; MC chỉ cung cấp spatial disagreement. Lấy mẫu Adapter hoặc thay answerability bằng trung bình MC là biến thể khác phải đánh giá riêng.

Với P_t là normalized spatial distribution sau verifier ở lượt t:

```text
P_bar = (1/T) Σ_t P_t
H_predictive = H(P_bar)
H_conditional = (1/T) Σ_t H(P_t)
D_MC = H_predictive − H_conditional
```

D_MC đo độ bất đồng của mixture, có thể diễn giải là thông tin giữa vị trí và chỉ số lượt lấy mẫu. Trong thiết lập này, gọi nó là model disagreement hoặc proxy epistemic có điều kiện; chưa chứng minh nó là uncertainty toàn mô hình. H_conditional chưa tự là aleatoric uncertainty.

Tất cả lượt có thể đồng ý nhưng cùng sai vì parser, backbone hoặc anchor binding. Cần đánh giá các ca **confidently wrong**, không chỉ ca MC variance lớn. Spatial covariance có thể phản ánh kích thước vật hoặc nhiều điểm hợp lệ; không tự là covariance sai số định vị.

### 7.3. Không sao chép FUSE bằng cách đổi tên feature

R0/D0 pre-projector và hashed text representation hiện tại không phải cặp image–text embeddings đã aligned theo contract FUSE. Chiếu chúng về cùng số chiều không đủ chứng minh alignment.

MC heatmaps không phải response embeddings lấy từ stochastic text decoding. Không áp log-determinant/Wishart likelihood hay closed-form posterior của FUSE lên heatmaps mà chưa kiểm tra giả thiết, rank và conditioning. Không gọi logistic là Bayesian fusion.

Nếu nghiên cứu tái hiện phần GP, phải tách thành nhánh riêng: contract embeddings, objective alignment, fit trên train, chi phí suy luận latent, regularization và ablation. Dataset không cần thêm capture, nhưng ít family độc lập và sự không tương ứng image–text ở ca absent/insufficient là rủi ro đáng kể. Nhánh này chưa phải ưu tiên triển khai.

Không dùng công thức heteroscedastic Gaussian regression `exp(-s)*BCE + 0.5*s` như bằng chứng đã học đúng classification aleatoric uncertainty.

## 8. Kiến trúc và hai mức tích hợp

```text
RGB + relative depth → frozen towers/cache → Sidecar → target/anchor maps
Prompt → typed query + frame ────────────────────────┐
Maps + typed query → geometric verifier ←───────────┘
                  → P_reason / MAP_reason / geometric evidence

Controlled Sidecar MC → verifier mỗi lượt → spatial disagreement
Deterministic answerability + observable evidence + disagreement
                  → new calibrator → hard_found selective policy
```

**Mức A — pilot hậu xử lý:** giữ checkpoint V2+Adapter hiện tại; verifier đọc maps và prompt; MC xuất evidence riêng. Đường answerability deterministic không nhận feature đã thay đổi. Risk phải chấm theo MAP thực sự được chọn cho biến thể, không dùng nhãn lỗi của MAP cũ.

Trong pilot, MAP cuối lấy từ P_reason của đường deterministic; P_bar của MC chỉ dùng làm diagnostics/evidence. Nếu muốn chọn MAP từ P_bar hoặc đổi final spatial distribution, đó là ablation riêng phải khóa trước calibration. Missing/invalid verifier dùng fallback rõ ràng và giữ status trong evidence, không xóa dấu vết thất bại.

**Mức B — tích hợp có học:** nếu cần phrase-conditioned anchor hoặc learned reasoning, sửa Sidecar và train trên train. Nếu đưa maps/moments/evidence mới vào Adapter, huấn luyện/đánh giá Adapter mới; không mặc định Adapter cũ phù hợp với phân bố feature mới.

Tại cả hai mức, mọi thay đổi vị trí hoặc risk features đều yêu cầu calibrator/profile mới trước khi đánh giá quyết định. Không tái sử dụng threshold cũ. Giữ hard_found làm policy chính của so sánh đầu tiên để hạn chế đổi nhiều yếu tố cùng lúc.

## 9. Supervision và objective khi phải train lại

Pilot A không cần thêm loss. Nếu chuyển sang B, tận dụng target/anchor spatial supervision và answerability hiện có, công bố rõ head nào train/frozen và provenance checkpoint khởi tạo.

Một objective có thể nghiên cứu:

```text
L_new = L_existing + λ_phrase L_phrase_anchor + λ_geom L_geometry
```

Đây là placeholder thiết kế, không phải cấu hình đã chọn. Chỉ thêm term khi định nghĩa được target/mask hợp lệ và giả thuyết cần kiểm chứng. Không cộng loss chỉ để tạo tên contribution.

- L_phrase_anchor chỉ áp dụng khi association phrase–mask trong train được xác minh.
- L_geometry cần nhãn/predicate hình học có ý nghĩa; edge proxy hiện tại không đủ thay thế relation truth độc lập.
- Không gán mọi AMBIGUOUS/INSUFFICIENT thành cạnh sai: relation có thể đúng nhưng thiếu điều kiện trả lời.
- Giữ ca ABSENT/INSUFFICIENT trong huấn luyện quyết định, tránh chỉ tối ưu FOUND recall.
- Nếu thêm location-mass phải là ablation riêng; code optional và smoke 02/10 chưa chứng minh tăng grounding.

Model/optimizer/checkpoint lựa chọn trên train/dev. Không dùng calibration để backprop vào Sidecar/Adapter và không dùng test để mining hard cases cho run này.

## 10. Rủi ro đầu vào depth và evaluator leakage

Trong mẫu `depth_corruption` thuộc family IID đầu tiên đã đối chiếu, relative-depth PNG bị thay đổi nhưng metric-depth path/hash vẫn trỏ tới clean capture. Family là `v211iid_family_000001`; metric path đã thấy là `raw/captures/v211iid_family_000001__clean_capture/depth_metric.npy` trong dataset. Đây là phát hiện ở mẫu đã xem, **chưa phải audit toàn bộ dataset**; không dùng mẫu IID này để chọn tham số reasoning.

Nếu verifier lấy metric depth sạch trong khi baseline chỉ nhìn depth bị corruption, đầu vào không còn tương đương. Tăng performance khi đó chưa chứng minh robustness của reasoning. Không sửa dataset hoặc lấy metric clean âm thầm để khắc phục.

Trước khi mở nhánh depth:

1. Audit provenance trên train/dev và contract sensor fields.
2. Kiểm tra relative/metric depth, registration và validity từng variant.
3. Chọn observable input contract nhất quán cho baseline/comparator.
4. Nếu chưa thể bảo đảm, giữ phạm vi image-frame hoặc công bố biến thể dùng thêm modality riêng; không gộp vào so sánh cùng đầu vào.

Existing evaluator relation graphs/object IDs có thể hỗ trợ chấm parser/binding offline. Không đưa chúng vào model forward/verifier. Dùng chúng xây metric mới không có nghĩa semantic graph exact match đã được triển khai hoặc đánh giá.

## 11. Calibration và policy của biến thể mới

Giữ event:

```text
E_new = (truth_answerability != FOUND) OR (MAP_new ngoài target mask)
```

Nếu MAP thay đổi, ca hợp lệ và prevalence error có thể đổi. 405 valid của baseline không phải mẫu số cố định cho mọi mô hình. Ngược lại, tập truth FOUND 420 và target-present 835 giữ cố định theo dataset/evaluator.

Sau freeze model, parser, verifier và MC configuration:

- Trích evidence trên calibration bằng đúng đường runtime.
- Dùng protocol family-aware; standardization fit trong training fold.
- Khóa danh sách candidate calibrator nhỏ trước khi xem kết quả, ví dụ logistic baseline và logistic có các evidence mới.
- Chọn bằng calibration family CV theo tiêu chí ghi trước; không thay đổi neural architecture theo kết quả calibration rồi coi protocol vẫn nguyên vẹn.
- Nếu kế thừa OOF threshold → fit-all runtime, lưu cả hai loại predictions/counts và thừa nhận chênh lệch.
- Budget chính đề xuất giữ 7,5% để so sánh; các mức khác chỉ khi được khóa trước.
- Nếu kế thừa threshold gate của bundle hiện tại, ghi rõ Wilson upper 95% và tối thiểu 60 accepted family; mọi thay đổi gate phải khóa trước fit, không âm thầm chỉ kiểm tra empirical sample risk.
- Nếu không có threshold thỏa gate, báo không có profile đạt tiêu chí; không nới gate dựa trên test.

Wilson cấp sample không loại bỏ phụ thuộc family hoặc hiệu ứng quét ngưỡng. Không tuyên bố guarantee risk trên test từ việc đạt calibration criterion. Không chuyển sang support/split-calibration chỉ vì đó là protocol thử nghiệm mới hơn.

Risk là xác suất lỗi nhận thức, không phải xác suất va chạm/grasp failure. Giữ riêng action label với robot motion và task success. Không chia risk thành bốn khoảng để suy ra bốn loại answerability/action.

Nếu map/region thay đổi, fit lại region calibration cho biến thể sau freeze bằng cùng event đã khai báo. Mass **0.604183197**, score coverage **86,83%**, direct intersection **95,81%** và diện tích **0,5194%** vẫn chỉ thuộc artifact cũ; không mang sang biến thể mới.

## 12. Thiết kế đối chứng và metric

### 12.1. Ma trận tối thiểu

| ID mô tả | Reasoning mới | MC/evidence mới | Mục đích |
|---|---|---|---|
| B | Không | Không | Bundle P-CRA-U đã chọn |
| R | Có | Không MC | Kiểm tra parser/verifier và evidence quan hệ |
| U | Không | Có | Kiểm tra nhánh uncertainty trên map hiện tại |
| R+U | Có | Có | Kiểm tra kết hợp và interaction |

Các ID chỉ là nhãn trong kế hoạch, chưa có run tương ứng. Reasoning evidence cần cho R vẫn được ghi rõ; không coi R là biến thể chỉ đổi MAP nếu nó đồng thời bổ sung risk features.

Lưu prediction trước/sau trên cùng mẫu. Ngoài bundle B đã khóa, dùng một calibration recipe chung cho các comparator mới để giảm nhiễu từ protocol khác nhau. Nếu fit lại calibrator cho kiến trúc B làm đối chứng, ghi là comparator riêng, không sửa bundle B gốc.

### 12.2. Ablation có điều kiện

- Verifier chỉ xuất evidence so với verifier có sửa MAP: tách lợi ích từ chối và định vị.
- Có/không clause binding đúng cấu trúc; không feed oracle binding làm kết quả chính.
- Anchor map hiện tại so với phrase-conditioned anchor nếu cần train.
- Entropy-only so với entropy + MC disagreement ở cùng calibration recipe.
- Reasoning evidence đơn lẻ, MC evidence đơn lẻ và kết hợp.
- Nếu thêm neural/loss changes, so trên cùng initialization/training budget; không đổi nhiều yếu tố rồi quy toàn bộ lợi ích cho một module.

Oracle diagnostic nếu thực sự cần phải gắn nhãn evaluator-only/upper bound riêng. Không trình bày nó như runtime hoặc kết quả chính.

### 12.3. Metric phải báo

**Grounding:** PIT trên 835 target-present; FOUND-only PIT trên 420; interior/distance theo evaluator cũ; paired corrections/regressions; parser coverage và binding accuracy trên subset có thể chấm. AMBIGUOUS mask union không chứng minh chọn đúng instance.

**Answerability:** accuracy, macro-F1, confusion matrix, recall FOUND, false FOUND/non-FOUND. Không gộp answerability accuracy với MAP correctness.

**Risk và utility:** Brier/NLL/ECE, AURC, đường risk–coverage, false accepts, valid accepts, valid recall và matched-coverage comparisons. Giữ riêng PIT trong tập nhận với composite correctness trong tập nhận.

**Uncertainty diagnostics:** disagreement có phân biệt đúng/sai trong truth FOUND không; confidently wrong; phản ứng trước depth/occlusion với điều kiện semantics phù hợp; ablation source/entropy. AUROC tổng đẹp nhưng không tăng ích lợi ở vùng nhận chưa đủ.

**Chi phí:** Sidecar/verifier/MC latency, bộ nhớ và throughput; báo cache-only riêng với end-to-end từ ảnh. Không gọi realtime khi chỉ đo cached inference.

CI/paired bootstrap theo family, các variant trong family đi cùng nhau. Khóa seed/số lượt bootstrap trước đánh giá. Một cải thiện nhỏ không có CI phù hợp chưa đủ khẳng định thắng. AURC oracle phụ thuộc error prevalence và cách tính; không áp chuẩn phổ quát AURC < 0,1, không mặc định oracle 0,229231 của baseline còn đúng nếu MAP/event prevalence đổi.

## 13. Lộ trình và gate/stop rules

Các bước là roadmap cho nhiệm vụ triển khai sau này, không phải lệnh đang được thực hiện.

| Giai đoạn | Công việc và artifact dự kiến | Gate / cách dừng |
|---|---|---|
| S0 — audit | Khóa baseline, input boundary, query grammar và annotation association trên train/dev | Chưa rõ depth/association thì thu hẹp phạm vi; không đoán hoặc dùng oracle bù |
| S1 — parser | Typed clauses, unit cases cho scope/shared anchor/unsupported | Parse sai có hệ thống thì sửa bằng train/dev; không mở test tìm pattern |
| S2 — verifier pilot | Compatibility, absolute evidence, trace trước/sau MAP | Dừng sửa MAP nếu regressions vượt mức cho phép đã khóa; có thể giữ evidence-only |
| S3 trong thiết kế ban đầu — MC pilot, nay tùy chọn | Controlled sampling, seed/config, latency và disagreement diagnostics | Không bắt buộc cho tích hợp hiện hành; tên S3 trong báo cáo mới chỉ bước ghép inference |
| S4 — optional learning | Phrase-conditioned Sidecar/Adapter nếu audit yêu cầu | Dừng nếu supervision association không đủ hoặc không qua development gate |
| S5 — freeze/calibration | Bundle mới, family CV, OOF/runtime profiles | Không có threshold đạt gate thì công bố thất bại, không tune bằng IID |
| S6 — IID reevaluation | B/R/U/R+U, paired metrics và failure report | Không lặp chọn biến thể trên IID rồi gọi test độc lập |
| S7 — lựa chọn | Báo cáo tradeoff có artifact cho người dùng xem | Không tự promote hoặc đổi active profile |

Trước S2/S3, phải ghi cụ thể mức regression, improvement và latency tối thiểu chấp nhận trên dev. Tài liệu này chưa có pilot để chọn số hợp lý, nên không bịa một gate phần trăm. Gate định lượng phải được khóa trước chạy, không hồi tố sau khi xem kết quả.

Reasoning chỉ qua gate nếu binding/hình học có nghĩa và có lợi ích độc lập, hoặc evidence giúp quyết định dù MAP không đổi. Uncertainty chỉ qua gate nếu đem thông tin bổ sung hữu ích cho risk/utility ngoài baseline. Brier tốt hơn nhưng ranking/local utility tệ hơn phải công bố đủ, không gọi cải thiện toàn diện.

## 14. Bản đồ code cho nhiệm vụ triển khai tương lai

| Vị trí hiện có | Vai trò / thay đổi có thể cần |
|---|---|
| `new/src/pcrau/text.py` | Giữ parser cũ/tokenization làm baseline, không đổi trong S1 |
| `new/src/pcrau/query_parser.py` | Typed parser prompt-only S1 đã có; direct/trái/phải và status, char/token spans, chưa nối runtime baseline |
| `new/src/pcrau/anchor_shadow.py` | Wrapper opt-in M1/M2 đã có; residual query và frozen head; outputs mới tách riêng, không vào graph/Adapter |
| `new/src/pcrau/anchor_shadow_experiment.py` | Protocol/batching/loss/runner API riêng; dùng trong pilot M1/M2 |
| `new/src/pcrau/anchor_shadow_pilot.py` | Evaluator/pairing/CI/gates train-dev, không inference oracle |
| `new/scripts/run_anchor_shadow_pilot.py` | Orchestration epoch/dev/best/early-stop; đã chạy một pilot |
| `new/scripts/report_anchor_shadow_pilot.py` | Tái tính paired report từ traces, không train; correction có provenance |
| `new/scripts/audit_anchor_shadow_failures.py` | Frozen replay/span interventions, hậu kiểm masks/semantic identity và22case ảnh; không optimizer |
| `new/src/pcrau/anchor_peak_experiment.py` | Loss v2 opt-in, supervision boundary và arm factory C1/P1/P2; không optimizer |
| `new/scripts/preflight_anchor_peak_v2.py` | Frozen identity/gradient/boundary/hash/latency preflight v2; đã PASS, chưa train |
| `new/configs/anchor_peak_pilot_v2_20261005.json` | Config preflight v2 giữ nguyên objective/data/init/budget/gates và flags lịch sử; trạng thái train mới nằm ở execution lock |
| `new/src/pcrau/anchor_peak_runner.py` | Epoch runner v2 gọi đúng peak loss; lazy optimizer có receipt; reuse dev selection/patience |
| `new/src/pcrau/anchor_peak_pilot.py` | Evaluator4arm, matched pairs/family CI và C1/P1/P2 gates |
| `new/scripts/run_anchor_peak_pilot.py` | Orchestrationv2 đã chạy:dev/best/reload/early-stop/paired/latency/freeze |
| `new/scripts/report_anchor_peak_pilot.py` | Read-only report/CSV/curves từsaved traces, không thêm train/eval |
| `new/src/pcrau/horizontal_verifier.py` | Evidence-only left/right geometry runtime; không oracle/MAP change hoặc binding certification |
| `new/src/pcrau/verifier_risk.py` | Frozen33/41/44features, family-crossfit calibration vàbundle guards |
| `new/scripts/run_verifier_calibration_iid.py` | Dev/calibration freeze; live IID numericalguard FAIL được bảo toàn |
| `new/scripts/reevaluate_verifier_iid_cached.py` | Cachedbaseline33+newgeometry IID continuation; reuse profiles, không refit sauIID |
| `new/scripts/report_horizontal_verifier.py` | Read-only comparison/risk–coverage report từsaved artifacts |
| `new/src/pcrau/unified_inference.py` | Đường opt-in live44features → verifier → calibrator → risk/decision/trace; không cached prediction evidence |
| `new/src/pcrau/frozen_rgbd.py` | Worker backbone từ RGB-D đã có; tách môi trường extractor và Sidecar, không capture/robot |
| `new/scripts/infer_spatial_variant.py` | Entry point nhận ảnh hoặc bốn features và prompt, load bundle r2 đã khóa |
| `new/scripts/build_unified_spatial_bundle.py` | S3 freeze, train/dev audit, calibration khớp producer và reload; đã chạy, không neural train |
| `new/scripts/report_unified_spatial.py` | Tám case dev thật, gồm thành công, binding sai, empty anchor và false accept |
| `new/scripts/verify_unified_spatial_bundle.py` | Kiểm tra hash/schema, replay risk từ evidence đã lưu và diagnostic geometry; không fit/inference/IID mới |
| `new/scripts/reevaluate_unified_spatial_iid.py` | S4 forward 1000 IID bằng bundle r2 đã khóa, runtime trước evaluator; không fit/train/chọn ngưỡng |
| `new/src/pcrau/live_iid_analysis.py` | Paired changes, family bootstrap và metrics của models/profiles cố định |
| `new/scripts/report_unified_spatial_iid_v2.py` | Báo cáo read-only, 8 ca đổi quyết định và 4 ca giới hạn; giải path RGB workspace-relative |
| `new/scripts/verify_live_iid_artifacts.py` | Replay risk/metrics/CI, độc lập hậu kiểm 1000 target/160 anchor, hash và case/provenance |
| `new/src/pcrau/model.py` | Kiểm tra slot binding; optional phrase conditioning; không thêm annotation vào MODEL_INPUT_KEYS |
| `new/src/pcrau/dataset.py` | Association supervision train và collate contract; evaluator fields tách riêng |
| `new/src/pcrau/postprocess.py` | Điểm nối map/verifier; giữ output baseline trước hậu xử lý |
| `new/src/pcrau/answerability_evidence.py` | Phân biệt Adapter evidence cũ với vector mới; không thay dimensions checkpoint cũ |
| `new/src/pcrau/engine.py` | Runner sampling riêng, kiểm soát mode/seed; deterministic evaluator cũ giữ rõ |
| `new/src/pcrau/losses.py` | Chỉ thêm objective có target hợp lệ và ablation riêng |
| `new/src/pcrau/selective_experiment.py` | Feature schema, event theo MAP_new, calibration và decisions |
| `new/scripts/fit_full_calibration_profiles.py` | Tham khảo family-CV/OOF protocol, không chạy mặc định từ tài liệu |
| `new/scripts/decide_experimental_profile.py` | Bundle/hash/runtime contract của biến thể mới |

Các module mới cho typed parser, verifier và MC runner chỉ tạo khi được giao triển khai. Ưu tiên module tách biệt và opt-in config; không đổi default runner/demo hoặc mở rộng boundary âm thầm.

## 15. Risk register

| Rủi ro | Hệ quả | Biện pháp |
|---|---|---|
| Parser đúng nhưng anchor slot sai vật | Hình học đúng trên đối tượng sai | Audit binding; phrase conditioning nếu cần |
| Anchor absent nhưng softmax vẫn có peak | Tạo evidence giả | Giữ unnormalized evidence, negatives và status |
| Pixel relation khác object relation | Loại nhầm phần vật hoặc candidate | Tách điểm/candidate-level approximation; đánh giá regressions |
| Base map đã relation-conditioned | Double counting, sharpen sai | Verifier-strength ablation, absolute evidence, không gọi Bayesian độc lập |
| Mọi compatibility thấp nhưng normalize sắc | Confidence giả | Lưu Z/compatibility trước normalize và dùng cho risk |
| Metric clean bypass depth corruption | So sánh không cùng đầu vào | Audit provenance; chưa đạt thì hoãn depth branch |
| MC chỉ lấy mẫu một phần mô hình | Underestimate uncertainty tổng | Công bố scope, đo confidently wrong |
| Entropy/covariance chủ yếu do vật lớn | Đặt sai tên aleatoric | Candidate-aware diagnostics; dùng tên descriptive |
| Adapter cũ nhận feature đã đổi | Distribution mismatch | Pilot giữ đường deterministic; train Adapter mới khi cần |
| Feature/calibrator nhiều với ít family | Overfit, coverage/risk không ổn định | Vector nhỏ, family CV, candidates khóa trước |
| IID đã quan sát, nhiều vòng phát triển | Claim generalization quá mạnh | Ghi reevaluation; không hứa test độc lập |
| Tăng MC không tăng utility | Độ trễ tăng vô ích | Gate marginal benefit/latency trên dev |
| Relation/source proxy bị coi causal | Kết luận không có bằng chứng | Evaluator riêng và ablation; ghi weak/proxy |

## 16. Điều kiện để định nghĩa và claim đứng vững

| Claim | Bằng chứng tối thiểu | Giới hạn vẫn phải nói |
|---|---|---|
| Explicit relational grounding | Typed clauses đúng, anchor binding, predicate tính được, ảnh hưởng MAP/decision, ablation | Trong grammar/frame/dataset đã đánh giá |
| Geometrically verified prediction | Input/frame hợp lệ, trace phép kiểm tra, negative/conflict handling | Perception có thể sai; không bảo đảm truth tuyệt đối |
| Model disagreement / proxy epistemic | Controlled sampling, scope rõ, tương quan với lỗi ngoài entropy baseline | Không bao phủ uncertainty backbone/parser cố định |
| Calibrated predictive error risk | Event rõ, freeze trước fit, family-aware calibration, metrics | Chưa phải robot safety hoặc guarantee test budget |
| Aleatoric/epistemic decomposition | Generative/noise assumptions, identifiable components và đánh giá phù hợp | Chưa đạt chỉ bằng source head, heatmap entropy hoặc MC |
| FUSE reproduction | Contract embeddings/sampling/GP/fusion và protocol tương ứng paper | Không đạt bằng logistic + MC heatmaps |
| Causal source attribution | Thiết kế can thiệp/đối chứng cô lập có căn cứ | Weak source labels và variant stratification chưa đủ |

Cách viết phù hợp nếu đạt gate: **P-CRA-U bổ sung geometric relational verification có hệ quy chiếu xác định và uncertainty evidence fusion, được hiệu chuẩn cho composite grounding/answerability error.** Tính mới và mức cải thiện phải chứng minh sau thực nghiệm; không suy ra từ tên module hoặc paper tham khảo.

## 17. Artifact và kiểm chứng cần lưu khi triển khai

Lưu riêng cho từng biến thể: input/data manifest hashes; config/checkpoint/freeze lock; parser/verifier version và supported grammar; MC scope/T/seed; baseline và final MAP; clause/binding trace; absolute evidence; feature schema/normalization; calibrator/profiles; calibration OOF/runtime predictions; evaluator event/counts; paired report và latency.

Các kiểm tra kỹ thuật cần thiết khi code thay đổi:

- Inference boundary không nhận oracle mask/IDs/graph; evaluator tách riêng.
- Parser tests có shared anchor, ternary relation và unsupported handling.
- Geometry synthetic cases kiểm tra trái/phải, frame direction, missing depth và conflicting clauses; không chỉ test lại công thức bằng cùng implementation.
- Verifier bypass giữ kết quả baseline; missing/zero compatibility không NaN và không bị coi verified.
- MC không đổi weights/stats, reproducibility và deterministic path hoạt động.
- Calibration folds không tách family, normalization fit đúng fold, event dùng đúng final MAP.

Lưu case thật có truy vết; không sửa confidence, dựng ảnh hoặc lấy oracle làm prediction để minh họa. Hình/dashboard chỉ làm sau khi có output hợp lệ; dashboard 2×3 và robot handoff vẫn là nhiệm vụ khác.

## 18. Những điểm chưa biết và bước triển khai đầu tiên

Sau S0/S1 và pilot đã biết chất lượng baseline/phrase-conditioned anchor trong subset horizontal. M1 có cải thiện hẹp nhưng chưa qua gate v1; xem báo cáo pilot ở trên. Chưa biết mức regression của verifier, utility MC, latency end-to-end và độ ổn định profile mới. Chưa audit toàn bộ depth provenance hoặc có anchor-graph evaluator tổng quát. Thí nghiệm có học đầu tiên đã hoàn thành, không được mô tả như binding đã giải quyết.

Sau S2, người dùng cho tiếp tục dù binding gate FAIL. Kết quả cached-IID có
utility cải thiện nhỏ nhưng budget FAIL; binding và historical numerical parity
chưa giải quyết. Các kết quả ấy được giữ nguyên, không mining/tune từ IID rồi
gọi là xác nhận độc lập.

**Điểm dừng sau S3, trước yêu cầu S4:** bước tích hợp đã hoàn tất
theo mục tiêu cơ chế chạy được. Bundle r2 lấy evidence trực tiếp, đã kiểm tra
train/dev, calibration khớp producer, reload và một RGB-D smoke thật. Kiểm tra
cuối replay 3000 dòng, 13 tests PASS và 3300 historical hashes giữ nguyên.
Diagnostic đặt raw geometry bằng0 làm đổi9/6/9quyết định train/dev/calibration
trong cùng calibrator; đó là wiring diagnostic, không phải ablation train lại.
Xem [biên bản bàn giao](S3_COMPLETION_20261007.md). Không tự train tiếp, mở MC,
robot, promote baseline hoặc đòi thắng metric để hoàn thành tích hợp. IID live
chưa đánh giá, là việc riêng nếu người dùng yêu cầu.

**Điểm dừng hiện hành sau S4:** yêu cầu reevaluation và bản tổng hợp cuối đã
hoàn tất. Bundle live có kết quả của đúng pipeline, cùng report paired,
CI và case thật. Có thể dùng [bản tổng hợp](SPATIAL_VARIANT_FINAL_SUMMARY_20261007.md)
làm nguồn cho kiến trúc/phương pháp/kết quả đồ án; chưa sửa LaTeX. Không tự
train tiếp, fit/chọn threshold từ IID, promote model hoặc mở robot/MC/OOD.
Cải thiện nhỏ và các thất bại binding/budget vẫn công bố đầy đủ, không chặn
hoàn thành mục tiêu tích hợp hiện hành của người dùng.

Tài liệu là phương án có thể kiểm chứng, không phải lời hứa tăng PIT/F1, đạt risk budget hoặc hoàn thành robot. Lúc lập chỉ thêm Markdown; các triển khai và pilot sau đó theo yêu cầu riêng đã được ghi ở phần cập nhật đầu tài liệu. Dataset, active profile và LaTeX được giữ nguyên.
