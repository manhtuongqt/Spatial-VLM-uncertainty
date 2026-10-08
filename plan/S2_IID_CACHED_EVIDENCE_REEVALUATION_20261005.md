# S2 — IID reevaluation bằng cached baseline evidence đã khóa

Runr3 fit đầy đủcalibration và khóa5calibrators/profiles trướcIID. Live-parity
guard dừng ởIID`v211iid_family_000052__relation_counterfactual`, Adapter
probability drift8,153915405273438e-5>tolerance2e-5 đã khóa. Target/source
khớp. **Numerical gateFAIL giữ nguyên; không nới tolerance bằngIID.**

Tiếp tục nhiệm vụ reevaluation offline trongroot riêng, dùng**đúng cached33
baseline evidence** vốn đã làinput calibration/runtime contract của thí nghiệm,
và freshverifier evidence từpredicted anchors. Không fit/chọn lại model, features,
calibrator hoặcthreshold sauIID. Copy profiles/checkpoints immutable từr3;
calibration vàprimaryP1_G44 không đổi. Residual/model weights không đổi.

Observable verifier inputs không phụ thuộc Adapter logits; targetprobability
grid vàsource phải exact soartifact, Adapter argmax giữ nguyên. Drift Adapter
được đo/báo trênmọiIID sample, không thay cachedprobability/evidence để cứu
guard. Risk sử dụng cachedbasefields đã hash, không sử dụng probabilityrecompute.

Kết quả chỉ chứng minh utility trênfrozen cached baseline outputs + predicted
geometry. **Chưa chứng minh parity của đườnglive mới**, không tựdeploy/promote.
Hai ngoại lệ được ghi tách: người dùngcho phép tiếp tục dùbinding gateFAIL;
IID numerical gateFAIL là hạn chế mới, không tựgắn PASS. OldIID đã được quan
sát, vẫn làreevaluation. Không robot/MC/MAP refinement hoặc neuraltrain.
