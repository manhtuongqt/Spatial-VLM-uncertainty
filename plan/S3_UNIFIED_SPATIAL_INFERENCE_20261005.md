# S3 — Hoàn thành đường inference xuyên suốt cho biến thể đồ án

Triển khai ngày 05/10/2026; hoàn thiện bàn giao ngày 07/10/2026. Giữ tên run/file
05/10 để truy vết. **Đã chạy RGB/depth + prompt → backbone đóng băng → Adapter
baseline + anchor P1 → parser/verifier → 44 features trực tiếp → calibrator →
risk, quyết định nhận thức và trace.** Runtime không lấy 33 evidence từ
prediction cũ. Bước tích hợp đã hoàn thành theo ưu tiên mới của người dùng;
cải thiện điểm số không phải điều kiện chặn.

## 1. Sản phẩm

- [Module inference](../new/src/pcrau/unified_inference.py),
  [RGB-D bridge](../new/src/pcrau/frozen_rgbd.py),
  [CLI riêng](../new/scripts/infer_spatial_variant.py).
- [Runner freeze/audit/calibration](../new/scripts/build_unified_spatial_bundle.py),
  [tests](../new/tests/test_unified_inference.py),
  [script case reporting](../new/scripts/report_unified_spatial.py),
  [script kiểm tra cuối](../new/scripts/verify_unified_spatial_bundle.py).
- [Bundle r2](../new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json),
  [freeze lock](../new/outputs/pcrau_unified_spatial_20261005_r2/frozen/freeze_lock.json),
  [calibrator](../new/outputs/pcrau_unified_spatial_20261005_r2/calibrator.json),
  [profile](../new/outputs/pcrau_unified_spatial_20261005_r2/profile.json),
  [summary](../new/outputs/pcrau_unified_spatial_20261005_r2/summary.json).
- [Tài liệu kiến trúc, contract và cách chạy](../new/docs/UNIFIED_SPATIAL_INFERENCE.md),
  [protocol](S3_UNIFIED_INFERENCE_PROTOCOL_20261005.md),
  [case table](../new/outputs/pcrau_unified_spatial_20261005_r2/report/CASE_TABLE.md).

## 2. Định nghĩa và cơ chế đã nối

Parser tạo target–relation–anchor và spans từ prompt. Nhánh P1 được điều kiện
hóa theo anchor span, xuất map riêng bằng checkpoint best epoch 8 đã chọn từ
pilot. Verifier tính signed margin với khoảng cách 12 px và mức tương thích
cặp phân bố cho trái/phải. Các features này vào logistic44; trace ghi từng
contribution vào logit để hậu kiểm phép tính. Không sửa target MAP hoặc chặn
cứng mọi margin âm. Binding/presence vẫn `UNVERIFIED`; mức tương thích hình
học không xác nhận danh từ đã gắn đúng vật thể.

Uncertainty output chính là **xác suất lỗi nhận thức được hiệu chuẩn** cho event
truth non-FOUND OR target MAP sai. Heatmap entropy, source scores và
compatibility là evidence/proxy. Chưa phân rã aleatoric/epistemic, chứng minh
causal attribution, suy luận 3D tổng quát hoặc đo xác suất robot nguy hiểm.

Exporter giữ riêng probabilities V2 trước Adapter và probabilities cuối sau
Adapter. Edge mask chỉ từ cú pháp. Đối chiếu dataset cho thấy edge_mask vốn
cũng bằng relation_mask AND language_anchor_mask, không lấy truth. Producer
runtime tái tạo quy tắc ấy mà không đọc mask hoặc dataset.__getitem__.

## 3. Kiểm tra kỹ thuật trên toàn train/dev

| Split | Mẫu/family | Horizontal / direct / unsupported | Anchor hit / visible | Empty anchor |
|---|---|---|---|---|
| Train | 1600/320 | 256/560/784 | 226/251 | 5 |
| Dev | 400/80 | 64/140/196 | 48/61 | 3 |
| Calibration | 1000/200 | 160/350/490 | 140/159 | 1 |

13 unit checks PASS: chặn oracle fields, nonfinite/shape, bảo toàn evidence
trước Adapter, prompt-edge mask, scope/null geometry, không sửa input,
sensitivity của risk/action khi đổi geometry, schema/bundle mismatch và 8
verifier checks cũ. Masks chỉ join sau khi đã lưu toàn bộ observable traces.
Bypass/inactive slots khớp baseline chính xác trên mọi batch. Capture/plain
baseline được đối chiếu mọi tensor ở 20 mẫu đầu mỗi split; không khẳng định đã
đối chiếu mọi baseline tensor của toàn 3000 mẫu. Neural state hashes trước/sau
không đổi, không optimizer. Reload bundle tái hiện chính xác 20 mẫu calibration.

Contribution geometry khác 0 ở 256/256 train, 64/64 dev, 160/160 calibration
horizontal. Đây là bằng chứng kết nối và ảnh hưởng số học vào risk, chưa chứng
minh lợi ích nhân quả. Diagnostic đặt ba raw geometry features bằng 0 trong
cùng calibrator được lưu ở final_verification; không gọi đó là ablation train lại.

Median batch 20 gồm validation/export khoảng 63,27 ms train, 65,42 ms dev và
63,76 ms calibration; không phải latency toàn backbone hoặc benchmark robot.

So 33 features trực tiếp với export dev lịch sử: target summary, source và
relation_consistency khớp chính xác; final Adapter probability có max delta
1,43144e-5; vài mean/top-k mass khác ≤1,1921e-7. Không sửa output hoặc thay
bằng cache. Nguyên nhân cụ thể của Adapter numerical drift chưa được chứng
minh; producer và runtime của calibration mới đồng nhất. Historical parity
FAIL của S2 vẫn giữ nguyên, không đổi thành PASS từ kết quả S3.

## 4. Calibration khớp producer mới

Freeze weights/source/schema trước fit. Chỉ một primary P1_G44, L2=0,01,
5 folds theo family, cùng quy tắc chọn threshold cũ. Không model search hoặc
neural training. Ngưỡng OOF **0,2889643687106893**, nhận 351/1000, lỗi 16,
199 accepted families; empirical risk OOF 4,5584%, Wilson upper 7,2757%.
OOF Brier=0,06395613, NLL=0,21721112, ECE10=0,01505691, AURC=0,24085498.

| Áp profile fit-all | Nhận | Đúng | Lỗi | Risk | Coverage |
|---|---:|---:|---:|---:|---:|
| Train | 663 | 648 | 15 | 2,26% | 41,44% |
| Dev | 139 | 132 | 7 | 5,04% | 34,75% |
| Calibration fit-all | 357 | 340 | 17 | 4,76% | 35,70% |

Train đã dùng học; dev đã dùng chọn checkpoint; calibration fit-all là
resubstitution. Các dòng này không thay kết quả test độc lập hoặc bảo đảm
risk 7,5%. **S3 không mở IID**; không gán 330/33/363 của S2 cached path cho
bundle live. Báo kết quả IID của đường mới cần một lượt reevaluation riêng,
giữ profile đã khóa; việc ấy không phải điều kiện còn thiếu của bước tích hợp.

## 5. Smoke RGB-D thật và môi trường

Dùng train sample `v211dev_family_000027__clean` có RGB-D sẵn, không capture
mới: mango right_of apple. Đường backbone thật và feature cache của cùng
observation đều EXECUTE, risk ≈0,058214; target grid, 44 features, risk và
quyết định khớp chính xác. Xem
[comparison](../new/outputs/pcrau_unified_spatial_20261005_r2/smoke/comparison.json)
và [output RGB-D](../new/outputs/pcrau_unified_spatial_20261005_r2/smoke/rgbd_prediction.json).

Backbone load+extract ≈6,98 s, feature forward ≈1,248 s, preprocess ≈116 ms.
Sidecar cold-call đầu tiên ≈598 ms sau extractor hoặc ≈343 ms từ cache.
Đo trên một case, chưa là benchmark có warmup hoặc bằng chứng parity mọi ảnh.
Backbone inventory hash được kiểm tra trước extract; Sidecar chỉ lấy features
trước projector.

[Run r1](../new/outputs/pcrau_unified_spatial_20261005/) hoàn tất feature→decision
nhưng smoke RGB-D lỗi import torch/torchvision do user-site. Giữ artifact và
source snapshot r1. R2 chạy extractor trong worker conda torch2.5.1 bằng
`python -s`; Sidecar vẫn torch2.13 đúng runtime calibration. Không sửa package.
Source hiện tại khác r1 sau sửa bridge; dùng r2 cho entry point hiện hành.

## 6. Case thật, gồm cả giới hạn

![Dev cases thật](../new/outputs/pcrau_unified_spatial_20261005_r2/report/dev_cases.png)

- `000001__clean`: apple → right_of → purple cube; target (350,150), anchor
  (290,250), margin 48 px, compatibility 0,999901, risk 0,202758, EXECUTE.
  Hậu kiểm anchor đúng.
- Câu đổi vai cùng family: purple cube → left_of → apple; margin 68 px,
  risk 0,144897, EXECUTE. Target và anchor đều đúng.
- Depth corruption cùng family: compatibility vẫn 0,999884 nhưng risk
  0,970546, REOBSERVE. Hình học tương thích không xóa thiếu evidence quan sát.
- `000383__relation_counterfactual`: yellow cube → left_of → apple; anchor
  peak trên orange, ngoài apple mask khi hậu kiểm. Margin 8 px, compatibility
  0,958688, risk 0,152992, EXECUTE. Phép tính trên predictions có thật nhưng
  binding sai; không gọi quan hệ đã verified.
- `000042__clean`: margin −12 px, compatibility 0,300290 vẫn EXECUTE,
  risk 0,280246; truth ABSENT. Evidence-only fusion có false accept thật và
  không phải hard veto.
- `000392__clean`: anchor mask rỗng khi hậu kiểm nhưng map vẫn có peak
  (130,170); risk 0,937834, ABSTAIN. Trace giữ presence UNVERIFIED.
- Direct và unsupported có trace bypass/null geometry đúng, vẫn đi qua risk.

Hình/JSON lấy từ predictions thật; không sửa confidence hoặc masks để làm
đẹp. Truth và anchor hit nằm riêng trong evaluator-only fields của case JSON.

## 7. Kết luận để đưa vào đồ án

Đã triển khai **kiểm chứng ràng buộc không gian trên đối tượng dự đoán và
hiệu chuẩn xác suất lỗi nhận thức** tham gia quyết định. Có thể mô tả cơ chế
này trong kiến trúc với phạm vi trái/phải 2D và các giới hạn đã nêu. Không cần
train tiếp, MC Dropout hoặc ép cải thiện metric để hoàn thành bước tích hợp.

Baseline/dataset/active profile được bảo toàn. Chưa robot motion, MoveIt hoặc
pick-and-place; chưa sửa LaTeX/Prism hay `pipeline.png`. Đã bàn giao code,
bundle, trace, case thật và tài liệu riêng.
