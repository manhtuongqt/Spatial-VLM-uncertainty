# S2 — Verifier evidence-only, calibration và IID theo ngoại lệ gate

Ngày05/10/2026. Người dùng giao “mở verifier/IID/calibration luôn đi, xem như
đạt gate”. **Đây là quyền tiếp tục dù binding gate chưa đạt**, không sửa kết quả
pilot hoặc gắn PASS: P1=C1 48/61, objective-improvement FAIL vẫn giữ nguyên.
Không promote active profile. Không train neural/MC/MAP refinement/robot.

## 1. Phạm vi và biến thể cố định trước calibration/IID

Verifier theo [contract S0](S0_VERIFIER_INPUT_CONTRACT_20261005.md): chỉ một
anchor, left_of/right_of trong image frame, margin12pixels từ annotation
convention. Direct bypass; depth/ternary/multi-clause/frames khác unsupported.
Đầu vào chỉ prompt, target probability grid, predicted anchor logits và anchor
logits cũ cho displacement diagnostic. Không mask/ID/variant/truth runtime.
P1 bestepoch8 làm biến thể chính **đã chọn trước calibration**, C1 best11 và
M0 anchor đóng băng làm controls; không chọn lại nhánh theo IID/calibration.

Giữ target MAP/probability grid và Adapter outputs. Anchor branch tách riêng;
không đưa map mới vào Adapter. Binding/presence luônUNVERIFIED; sigmoid-max
cao không xác nhận đúng object. Verifier là diagnostic geometry evidence,
không xuất chứng nhận GEOMETRY_VERIFIED hoặc presence probability.

Tọa độ ảnh x phải,y xuống; grid24×32 centers(20x+10,20y+10).
Signed margin: right target_x−anchor_x−12; left anchor_x−target_x−12.
Peak compatible iff margin>0. Pair compatibility là tổngP_target(i)P_anchor(j)
trên center pairs thỏa margin. Đây là conditional normalized-map score, không
phải probability quan hệ đúng/anchor tồn tại, không giả định Bayesian independence.
Giữ cả logits tuyệt đối và entropy/peak. Không normalize để hard sửa MAP.

## 2. Feature schemas và ablation cố định

Tất cả dùng33evidence frozen baseline, thêm:

- Parser flags: horizontal_supported,direct_bypass,parse_unsupported.
- Anchor evidence: maxlogit,sigmoidmax,normalizedentropy,softmaxpeak,
  displacement-to-M0 normalized image diagonal.
- Geometry: signedpeakmargin/640,peakcompatible,paircompatibility.

Các extras bằng0 ngoài horizontal scope (parser flags vẫn phản ánh scope);
trace dùng null cho geometry chưa tính, không tự coi unsupported là direct.

| ID | Features | Vai trò |
|---|---:|---|
| B33 | Baseline33 | Refit comparator, phải tái hiện active logistic33 |
| P1_A41 | B33+parser/anchor8 | Kiểm tra evidence anchor không geometry |
| M0_G44 | B33+11verifier extras từM0 | Geometry với anchor cũ |
| C1_G44 | B33+11extras từC1 | Lossv1 control |
| P1_G44 | B33+11extras từP1 | **Primary cố định**, không chọn theo metric |

L2=0,01 cố định cho tất cả, logistic dampedNewton như recipe hiện có.
Calibration1000samples/200family,5-fold family crossfit cùng assignment recipe
baseline; fit mean/std trong training fold, fit-all riêng sauOOF. Không fit bằng
dev/IID; không tune hyper/schema sau nhìnIID. OOF→fit-all mismatch phải báo.
Không lựa chọn calibrator cạnh tranh theoIID; ablations không được tự promote.

Policyhard_found, budget7,5%, Wilson upper95%(z1,96), ít nhất60acceptedfamily.
Ngưỡng chọn trênOOF calibration rồi ápfit-all runtime, đúng recipe đã chốt.
Không nới calibration threshold gate vì binding override; nếuNone báo reject-all.

Event E=(truth≠FOUND)OR(MAP ngoài target), MAP không đổi. Tậpvalid/grounding
mẫu số giữbaseline nếu source predictions kiểm chứng đúng. Region không đổi,
không refit region hoặc mang region metrics cũ thành đóng góp verifier.

## 3. Freeze, thứ tự và đối chứng

Trước calibration: unit/preflight geometry/boundary, dev observable inference,
source/config/checkpoint/schema hashes, freeze runtime source. Reuse saved33
baseline evidence từ đúngcheckpoint; recompute observable target/answerability/
source để kiểm chứng parity. Oracle chỉ join evaluator sau saved inference.
Nếu mismatch phải dừng/ghi lỗi, không sửa predictions để khớp artifact.

Fit calibration và khóa tất cảcalibrators/profiles **trước lượt inference/apply
IID của runner mới**. Schema và kết quả lịch sử của oldIID đã được đọc khi
chuẩn bị code; không tuyên bố chưa từng xem IID hoặc xác nhận độc lập.
IID cũ1000samples/200family là reevaluation đã quan sát, không test mới untouched.
Chỉ sau freeze chạy verifier IID và apply ngưỡng đã khóa. Không dùng IID chọn
module/nhánh/ngưỡng. Official baseline risk threshold giữ0,2611932834526145,
report327correct/34errors/361accepted làm reference nếu replay khớp.

Báo từngmodel: coverage,errors,correctaccepts,validrecall,risk/Brier/NLL/ECE/AURC,
grounding/answerability invariant,scope và anchor/geometry diagnostics. Paired
familybootstrap5000 seed24082026 choP1_G44−B33/P1_A41/C1_G44; fixed thresholds,
ratio riskCI loại resamples không accepted và báo số hợp lệ. Không gọi guarantee
robot hoặc budget guarantee từ empirical/Wilson sample criteria.

## 4. Phạm vi hoàn thành

Deliverables: verifier runtime+tests, freeze lock/config/trace, calibrationOOF/
fit-all calibrators+profiles, savedIID decisions+paired report, ghi kết quả và
gate override trung thực. Giữ mọi old model/data/artifact và active profile.
Binding chưa được chứng minh đúng; geometry evidence hữu ích hoặc vô ích được
quyết định bằng đối chứng. Thành công metric không tự chứng minh semantic
binding hoàn chỉnh hoặc phân rãaleatoric/epistemic.
