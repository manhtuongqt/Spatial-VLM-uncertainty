# S5 — Ước lượng MC cho năm nhãn nguồn, contract trước chạy

Ngày07/10/2026. Người dùng cần cơ chế thật để giải thích năm nguồn với thầy,
không đổi tên kết quả cũ cho khớp lời đã nói.

## Định nghĩa được giữ

Năm tên semantic/relation/spatial/depth/occlusion là năm nhãn đầu ra source
head đã học bằng multi-label weighted BCE và family ranking. Spatial vẫn là
weak label từ AMBIGUOUS. MC không tự tìm hoặc chứng minh năm nguyên nhân.

Nhánh mới xuất hai nhóm đại lượng, cho từng nguồn k:

1. Model-averaged source score `s_k = mean_t sigmoid(z_k^(t))`.
2. Source-label predictive uncertainty: predictive entropy, expected entropy,
   standard deviation và mutual information `MI_k = H(s_k)-mean_t H(p_k^(t))`.

H là entropy Bernoulli; entropy/MI chuẩn hóa bằng ln2. MI gọi là **MC model
disagreement proxy về nhãn nguồn k**, điều kiện trên representation cố định.
Không gọi MI là mức che khuất/noise thật, hoặc posterior của toàn VLM.
Không gọi expected entropy là aleatoric đã nhận dạng và kiểm chứng.

## Thiết kế đã khóa

- Backbone, Sidecar, Adapter, P1 và profile live r2 giữ nguyên.
- Chỉ sample bản sao của source_head đã train; không sửa source head gốc.
  Head hiện có Dropout p=0,1; không thêm dropout hậu nghiệm hay thay p.
- Capture input thật của source_head bằng pre-hook trong một deterministic
  forward, rồi chạy bản sao head20lượt với20mask dropout độc lập theo RNG.
  Không bật `.train()` cho baseline/wrapper; không sampling Transformer/fusion.
- Các weights đều requires_grad=False, không optimizer hoặc cập nhật stats.
  Chỉ Dropout của bản sao hoạt động; LayerNorm/Linear giữ mode eval.
- CUDA BF16 như producer, FP32 sigmoid, Float64 MC moments/entropy. RNG có
  fork/restore; seed24082026+batch_index, batch20 theo manifest train rồi dev.
- Primary T=20, không chọn T/p theo kết quả. Diagnostic batch20đầu dev:
  cùngseed phải replay exact; seedkhác phải có masks khác; T=100reference chỉ
  để báo sampling error, không dùng chọn estimator hoặc cải thiện số.
- Train1600/dev400; không mở calibration/IID, train thêm hoặc fit ngưỡng/risk.
  Lưu source context, raw samples và observable MC rows trước evaluator join.
- Hậu kiểm source labels chỉ sau export. Báo per-source Brier/NLL/ECE, F1 tại
  ngưỡng0,5 để mô tả, entropy/MI và cả confident-wrong cases. Scores chưa có
  per-source calibration; weighted BCE có thể làm sigmoid chưa calibrated.
- Risk/action/MAP deterministic hiện hành bảo toàn; không đưa MC mean/MI
  vào calibrator44 vốn chưa được fit với producer mới.
- Output riêng `new/outputs/pcrau_source_mc_train_dev_20261007/`; bảo toàn
  artifact cũ, có hash/source snapshot, báo cáo cơ chế và lời giải thích cho thầy.

## Nguồn phương pháp

- Gal & Ghahramani2016: https://proceedings.mlr.press/v48/gal16.html
- Gal, Islam & Ghahramani2017, BALD/MI:
  https://proceedings.mlr.press/v70/gal17a/gal17a.pdf

Công thức áp dụng riêng cho từng Bernoulli source label; đó là adaptation cho
head của đồ án, không tái hiện toàn bộ Bayesian architecture của paper.
