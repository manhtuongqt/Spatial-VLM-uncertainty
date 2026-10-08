# S6 — uncertainty của nhiệm vụ, protocol trước optimizer

Ngày 07/10/2026. Người dùng yêu cầu triển khai nhanh, tích hợp và đánh giá,
không bắt buộc thắng baseline. Bản pasted text được đọc đầy đủ; hướng chọn là
semantic/spatial/relation/depth và reference completion dưới occlusion, không
đổi tên source scores thành năm nguyên nhân uncertainty.

## Phạm vi và dữ liệu

- Giữ nguyên baseline Adapter, P1 anchor, bundle live r2, active profile và
  archive. Tạo biến thể riêng. Không LaTeX, robot motion hoặc Test-OOD.
- Train/dev giữ 320/80 family, 1.600/400 presentations. Không capture mới.
- Catalog 22 lớp gồm background và 21 semantic classes của archive. Supervise
  semantic identity bằng instance-label images, không đưa ID/mask vào inference.
  Identity tại predicted target/anchor là closed-set classification; chưa đảm
  bảo chọn đúng instance khi nhiều vật cùng lớp. Matching score lấy probability
  của noun class đã parse; semantic class entropy khác entropy vị trí.
- Depth học mean/variance tại grid centers từ metric-depth reference đã lưu;
  input vẫn là frozen RGB/relative-depth features. Reference là phép đo mô phỏng,
  không ground truth noise vật lý robot thật. Chỉ depth hữu hạn trong (0.1,5)m.
- Supplement completion dùng cặp clean/physical-occluder: clean FOUND một target,
  camera/TF bằng nhau, chỉ requested pose của occluder đổi, observed target subset
  reference dilation 2px. Mask erosion 2px dùng hậu kiểm hidden core tránh biên.
  124 train/32 dev pairs; hidden core >=2% reference: 9 train/6 dev. Đây là
  reference-visible support completion, không full amodal shape. Nhãn mới chỉ
  từ ảnh đã capture; 6 cặp có RGB và overlay để người dùng xem. Không oracle input.

## Model và phép đo

1. RGB projection frozen 128D -> LayerNorm/MLP128/Dropout0.1 -> semantic22;
   unweighted categorical CE ở grid centers. Không coi lớp background là vật.
2. RGB+depth projections frozen 256D -> LayerNorm/MLP128/Dropout0.1 -> depth
   mean softplus+0.05m và variance softplus+1e-4m²; Gaussian NLL. MC variance
   of means tách khỏi expected modeled conditional variance, chưa nhận dạng
   aleatoric noise thật hoặc uncertainty toàn VLM.
3. Fused128 + target noun pooled text128 + original target logit + xy coordinates
   -> MLP128/Dropout0.1 -> residual completion logits; output categorical density
   cho một điểm trong reference support. Train categorical CE với reference mask
   area distribution; ordinary nonempty target masks và override 124 pairs bằng
   clean reference. Không dùng Gaussian regression formula cho BCE classification.
4. Spatial MC: copy read-only của trained fusion residual blocks, dropout p=0.1;
   original target/anchor heads và P1 queries frozen. Text/context/backbone cố định.
   T=20, sequential fresh masks, seed24082026+batch_index. Original baseline eval.
5. Relation: mỗi draw lấy target/anchor categorical distributions, tính left/right
   compatibility bằng cùng kernel12px. Bernoulli entropy/MI dưới product-of-marginals
   approximation. Không xác nhận binding/presence hoặc độc lập thật giữa maps.
6. Semantic PE/MI trên class distribution ở deterministic predicted node locations;
   completion PE/MI trên reference-point distribution; spatial PE/MI trên maps.
   Entropy chuẩn hóa log(number of outcomes). Giữ mean predictions/std và raw samples.

## Huấn luyện, freeze, calibration và evaluation

- Chỉ ba auxiliary heads được optimizer; baseline/P1 luôn đóng băng. Seed24082026,
  AdamW lr3e-4, weight decay1e-3; max15epoch, batch20, clip5, patience3.
- Joint loss = semantic CE + depth Gaussian NLL + completion categorical CE.
  Dev joint NLL chọn best checkpoint, ghi mọi epoch và reload best. Không gate thắng.
- Preflight: finite losses/gradients, chỉ head params trainable, forbidden annotation
  inputs rejected, modes/RNG/weights preserved, MC mean/entropy algebra checked.
- Neural/source/schema freeze trước calibration. Base44 evidence lấy forward live
  nguyên producer; thêm scalar MC tasks và missing flags. Calibrator logistic L2=.01,
  family crossfit5fold, threshold hard_found từ OOF theo budget7.5% và min60family.
  Không ép threshold nếu không có candidate hợp lệ; báo no-execute profile nếu cần.
- Chạy calibration1000/200family rồi reevaluate IID cũ1000/200family với bundle khóa;
  không chọn checkpoint/features/threshold bằng IID. IID đã quan sát, không test mới.
- Báo grounding/answerability/coverage/error risk, Brier/NLL/ECE/AURC; paired delta và
  CI theo family. Task metrics: semantic identity/matching, spatial/anchor hit và
  entropy/MI vs errors; relation diagnostic; depth error/NLL/interval coverage;
  completion reference mass/PIT/hidden-core mass trên eligible pairs, n ghi rõ.
- Không cộng entropy, MI hay năm nhóm thành tổng uncertainty vật lý. Risk fit event
  truth non-FOUND OR deterministic target MAP outside target; MAP không tự đổi.

## Paper và phần được áp dụng

- [Kendall & Gal2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf): task likelihood, depth mean/variance + MC.
- [Gal, Islam & Ghahramani2017](https://proceedings.mlr.press/v70/gal17a/gal17a.pdf): MC mean probabilities, predictive entropy và MI.
- [PSGP, WACV Workshops2026](https://openaccess.thecvf.com/content/WACV2026W/SG4SI/papers/Li_Probabilistic_Scene_Graph_Prompting_Uncertainty-Aware_Structured_Reasoning_in_Multimodal_LLMs_WACVW_2026_paper.pdf): uncertainty propagation qua object/relation predictions; không tái hiện soft prompting hoặc full graph.
- GCA tiếp tục là tham khảo tách parse/geometry. FUSE là related work, không gọi
  logistic+MC là Bayesian fusion/FUSE reproduction.

Các giả thiết và ngưỡng dữ liệu trên khóa trước optimizer; lỗi kỹ thuật sẽ ghi
revision minh bạch, không sửa số/nhãn/ảnh hoặc gate để hợp thức hóa kết quả.
