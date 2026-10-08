# Bàn giao ngữ cảnh đồ án Spatial-VLM / P-CRA-U

Ngày cập nhật: 02/10/2026, theo giờ Việt Nam.

Workspace: `/home/dhcn/ur_ws/src/myproject`.

Đây là bản bàn giao để session mới đọc hiểu đồ án và tiếp tục công việc. Các số liệu dưới đây là bản tóm tắt có đường dẫn nguồn, không thay thế artifact gốc. Khi người dùng yêu cầu công việc mới, kiểm tra file hiện tại trước khi sửa; không tự thực hiện mọi việc được liệt kê trong phần còn thiếu.

## 1. Trạng thái mới nhất cần nắm ngay

- Đồ án tập trung vào spatial vision-language grounding, answerability, source uncertainty, calibration và selective prediction; không phải xây foundation model mới hay agent thao tác tổng quát.
- Mô hình chính được gọi là **P-CRA-U**. Đây là checkpoint lịch sử **best V2, epoch 13, global step 1120** trên backbone RoboRefer-2B-SFT đóng băng.
- Chương 1–6 đã có nội dung LaTeX. Chương 5 có bảng kết quả từ artifact thật; Chương 6 tổng kết đóng góp, hạn chế và hướng phát triển.
- Các hình khoa học dùng dữ liệu thật hoặc sơ đồ phương pháp khớp triển khai, được dựng bằng Python/Matplotlib. Không dùng ảnh sinh AI thay kết quả thực nghiệm.
- Năm hình bổ sung gần đây gồm confusion matrix answerability, đường cong huấn luyện, paired source delta, uncertainty–localization error và coverage–area.
- Đường dẫn ảnh trong LaTeX hiện dùng dạng `hinhanh/<tên ảnh>.png` theo yêu cầu của người dùng.
- **Người dùng xác nhận đã biên dịch được ảnh trên Prism trong tin nhắn mới nhất. Không tiếp tục báo rằng ảnh trên Prism chưa biên dịch được.** Đây là xác nhận từ người dùng, không phải phép biên dịch local của trợ lý.
- Chưa có bằng chứng pick-and-place khép kín do P-CRA-U điều khiển. Audit Gazebo có điểm đúng nhưng policy vẫn ASK_USER, không nối MoveIt và không phát chuyển động.
- Yêu cầu hiện tại chỉ là tạo tài liệu bàn giao và prompt cho session mới; không tự train lại, chạy demo hay chỉnh nội dung các chương.

## 2. Cách làm việc và các yêu cầu của người dùng

Trao đổi bằng tiếng Việt, ngắn gọn, đi thẳng vào kết quả. Khi làm việc lâu cần cập nhật tiến độ, nhưng tránh giải thích công cụ quá dài hoặc làm cầu kỳ ngoài yêu cầu.

Các quy tắc quan trọng:

1. Chỉ sửa phần người dùng yêu cầu. Nếu họ yêu cầu đổi nguồn ảnh thì chỉ đổi đối số đường dẫn của `\includegraphics`; giữ nguyên caption, label, kích thước, tùy chọn, số liệu và toàn bộ nội dung khác.
2. Người dùng đã từng phản đối việc tạo thêm `.py`/`.json` khi đang yêu cầu viết chương. Không tự tạo script/phân tích phụ không cần thiết. Khi họ yêu cầu dựng hình bằng Python thì dùng Python là đúng phạm vi.
3. Viết khóa luận chi tiết, nhưng không bịa kết quả để khớp kế hoạch. Số liệu phải có artifact tương ứng; phần chưa đánh giá ghi rõ chưa đánh giá, không điền 0 thay dữ liệu thiếu.
4. Bảng theo phong cách paper: `booktabs`, không kẻ dọc, đơn vị rõ; tỷ lệ có `%` dùng phần trăm, F1/AUROC/AP thường dùng [0,1], chênh lệch tỷ lệ dùng điểm phần trăm (pp).
5. Hình chưa có có thể giữ khung và tiêu đề “bổ sung sau”. Không tạo minh họa giả rồi gọi là prediction hoặc trial thật.
6. Dùng tên P-CRA-U cho nhãn mô hình hiển thị. Tên run, sample ID, khóa dữ liệu, hash và tên file chứa `v2` giữ nguyên để truy vết.
7. Không tự đổi tên toàn bộ file/checkpoint hoặc thay tất cả chữ V2 trong văn bản nếu nhiệm vụ chỉ yêu cầu đổi nhãn trên hình. Trong các chương vẫn còn cách gọi V2 lịch sử; chưa thực hiện một đợt đổi tên toàn văn riêng.
8. Không sử dụng Test-OOD trong phạm vi khóa luận này. Không tự mở rộng thí nghiệm sang OOD hoặc robot vật lý.
9. Không chỉnh checkpoint, calibrator, risk threshold, region mass hoặc prompt bằng Test-IID đã quan sát.
10. Không tự chạy robot, ép EXECUTE, hạ ngưỡng để demo chuyển động, thay world/URDF hoặc bỏ qua cổng hình học.
11. Bảo toàn worktree bẩn; không reset, checkout, xóa, commit/push nếu người dùng chưa yêu cầu.

## 3. Hiểu đúng bài toán và kiến trúc thực tế

Tên phương pháp: **Probabilistic Cross-modal Relation-Aware Uncertainty (P-CRA-U)**.

Đầu vào nhận thức gồm RGB, relative depth và câu lệnh. Metric depth, CameraInfo và TF phục vụ tầng hình học/kiểm tra 3D; không đồng nhất relative depth đầu vào backbone với độ sâu tính theo mét.

Luồng triển khai chính:

```text
RGB + relative depth + câu lệnh
    → tower RGB/depth của RoboRefer-2B-SFT đóng băng
    → đặc trưng pre-projector R0/D0 trên lưới 24×32×1152
    → sidecar P-CRA-U trainable
       ├─ hashed-token Transformer cho câu lệnh
       ├─ target / relation / anchor slots học được
       ├─ relation-conditioned RGB–depth fusion
       ├─ heatmap target, interior, anchor
       ├─ relation-edge evidence / predicted graph embedding
       ├─ answerability 4 lớp
       └─ source scores 5 nhóm
    → MAP, modes, entropy, peak margin, covariance cục bộ
    → calibrator logistic độc lập trên 18 evidence
    → risk, spatial confidence region
    → EXECUTE / REOBSERVE / ASK_USER / ABSTAIN
```

Code chính: `new/src/pcrau/model.py`, lớp lịch sử `PCRAUTargetV2`; các module liên quan gồm `dataset.py`, `losses.py`, `engine.py`, `postprocess.py`, `calibration.py`, `policy.py`, `metrics.py`, `text.py`.

Ranh giới cần giữ:

- Sidecar lấy feature trước projector từ các tower, không phải hidden state của các lớp ngôn ngữ cuối RoboRefer.
- Các slot là biểu diễn học được có điều kiện theo truy vấn, không phải parser ký hiệu đầy đủ đã kiểm chứng noun/node/edge exact match.
- Có khả năng cấu hình tối đa ba relation/anchor slot; không lấy khả năng cấu hình làm bằng chứng đã giải quyết mọi câu nhiều mệnh đề.
- Spatial head xuất phân bố 24×32. Một mode không tự động tương đương một instance được nhận dạng đúng.
- Source scores là evidence theo supervision, không phải phép phân rã nhân quả mọi lỗi.
- Spatial source là nhãn yếu suy từ AMBIGUOUS. Relation-edge supervision là proxy, không phải ground-truth quan hệ hình học độc lập đầy đủ.
- Mask/annotation evaluator không được đưa vào model input hay dùng để chỉnh prediction sau suy luận.
- Feature disagreement với RoboRefer là optional; prediction Test-IID chính không chứa tín hiệu disagreement thực, không được nói nó đã tạo lợi ích trong kết quả chính.
- Chấp nhận grounding trên ảnh khác với grasp pose hợp lệ, chuyển động an toàn hay task success.

## 4. Nguồn tham chiếu và thứ tự đọc

Session mới nên đọc theo nhu cầu, không chạy train/evaluation chỉ để tìm hiểu:

1. Bản bàn giao này và yêu cầu mới nhất của người dùng.
2. `plan/formkhoaluan.md`: cấu trúc và định hướng trình bày khóa luận.
3. `plan/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md`: kế hoạch nghiên cứu; phân biệt thiết kế mục tiêu với phần đã thực nghiệm.
4. Bản kế hoạch đang mở trong IDE: `/home/dhcn/Downloads/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md`. Có bản trong `plan/`; nếu cần chỉnh kế hoạch phải kiểm tra hai bản, không mặc định đồng bộ.
5. `latex/Chapter1/chapter1.tex` đến `latex/Chapter6/chapter6.tex`, đọc đầy đủ khi được yêu cầu rà soát toàn khóa luận.
6. `new/README.md`, `new/docs/ARCHITECTURE.md`, `new/docs/DATA_CONTRACT.md`, cùng code liên quan để đối chiếu triển khai.
7. `new/test_iid/README.md`, protocol/lock, metrics/prediction/comparison chính thức.
8. `new/hinhanh/README.md` và các builder nếu làm hình.
9. `new/demo_gazebo/README.md` và đúng run audit nếu làm demo.

Tài liệu cũ có thể phản ánh thời điểm development. Ví dụ `new/docs/SCIENTIFIC_BOUNDARY.md` còn câu chưa có Test-IID path, trong khi Test-IID đã được đánh giá ở `new/test_iid/`. Không dùng câu lịch sử đó để phủ nhận artifact hiện tại. Không tự sửa tài liệu ngoài nhiệm vụ.

`plan/SESSION_HANDOFF_PROMPT_20261001.md` là bàn giao cũ, chưa chứa năm hình mới và cập nhật Prism. Khi mâu thuẫn, kiểm tra artifact/current file và bản bàn giao mới này.

## 5. Dữ liệu, protocol và checkpoint đã khóa

| Tập | Family | Sample | Vai trò |
|---|---:|---:|---|
| Train | 320 | 1600 | Huấn luyện sidecar |
| Dev | 80 | 400 | Chọn checkpoint |
| Calibration | 200 | 1000 | Fit hậu xử lý sau model freeze |
| Test-IID | 200 | 1000 | Đánh giá khóa, không fit lại |

Mỗi family có năm variant: clean, semantic counterfactual, relation counterfactual, depth corruption, occlusion/view counterfactual. Các variant cùng family phụ thuộc nhau; không coi 1000 variant là 1000 quan sát độc lập.

Test-IID được capture mới, không đổi tên dev thành test. Có canary 20 family trước khi mở rộng 200 family, tổng 400 capture; ảnh là render/capture Gazebo.

Checkpoint:

- Run: `new/outputs/pcrau_target_v2_full_seed_24082026/`.
- Model: `checkpoints/best/model.safetensors` bên trong run.
- Metadata: `best.json`; history: `history.jsonl`.
- Epoch 13, zero-based, tức lượt huấn luyện thứ 14; global step 1120.
- Chọn theo dev total loss 1.6971108556, không chọn theo test và không lấy checkpoint cuối run.
- Dev PIT 326/335 = 97.3134%.
- Model SHA-256: `1505152fca7b725d7db701c1341ad1a57049d6b5ac450a9f5ce0c8d78705ba26`.
- Config SHA-256: `e4f21ccc255c2f097fd8b062e21e954719357cb6e00645758bc1cdf36a290b0c`.
- Calibrator: `new/outputs/pcrau_target_v2_full_seed_24082026/evaluation/calibration/calibrator.json`.
- Freeze lock: `new/test_iid/contracts/best_v2_freeze_lock.json`.

V3 regularized đã chạy ba seed 24082026/27/28, max dev PIT lần lượt 91.04%, 88.36%, 89.85%, đều chưa đạt cổng dev 97%. Các run đổi nhiều yếu tố đồng thời, không phải ablation cô lập và không tạo bằng chứng nhiều seed cho P-CRA-U chính. Không dùng file tổng hợp V3 rỗng/không hợp lệ để báo mean/std Test-IID.

Lưu ý tên “kế hoạch V3” không có nghĩa checkpoint experiment V3 là mô hình chính thức. Mô hình chính vẫn là P-CRA-U/best V2.

## 6. Artifact nguồn cho kết quả

Nguồn hoạt động đang được script dựng hình đọc:

```text
new/test_iid/protocol/test_iid_eval_manifest.json
new/test_iid/evaluation/best_v2/full_metrics.json
new/test_iid/evaluation/best_v2/predictions.jsonl
new/test_iid/evaluation/best_v2/decisions.jsonl
new/test_iid/evaluation/comparison_original_vs_best_v2/full_comparison.json
new/test_iid/evaluation/comparison_original_vs_best_v2/paired_samples.csv
new/test_iid/evaluation/comparison_original_vs_best_v2/SUMMARY.md
```

`new/ketqua/` là bộ tổng hợp artifact/tài liệu báo cáo, gồm dữ liệu, huấn luyện, đánh giá, experiment V3 và kết quả lịch sử. Các bản sao trong đó hữu ích để tra cứu nhưng không mặc định mọi bản sao được cập nhật cùng thời điểm. Grounding baseline chính là **RoboRefer gốc trên cùng Test-IID**, không thay bằng legacy B1 dev.

## 7. Số liệu khóa cần dùng đúng mẫu số

### 7.1. Grounding paired

- 1000 sample tổng cộng; 835 có target mask không rỗng: 420 FOUND, 105 AMBIGUOUS, 310 INSUFFICIENT_EVIDENCE.
- Target-present PIT: RoboRefer 724/835 = 86.71%; P-CRA-U 775/835 = 92.81%.
- Chênh lệch +6.11 pp, CI 95% [1.78, 10.49] pp; CI hai mô hình lần lượt [83.23, 89.97]% và [90.25, 95.21]%.
- Interior: 86.59% → 91.50%, chênh lệch +4.91 pp, CI [0.48, 9.37] pp.
- Mean distance tới mask: 27.9150 → 8.4437 pixel; chênh lệch −19.4714, CI [−28.5608, −10.9371]. Khoảng cách là distance tới mask, bằng 0 khi hit; không phải sai số tới tâm vật.
- Paired outcomes: cả hai đúng 668, chỉ RoboRefer đúng 56, chỉ P-CRA-U đúng 107, cả hai sai 4.
- Bootstrap paired: 20000 lượt theo 200 family; sign-flip theo family 100000 lượt, p ≈ 0.00745.
- FOUND-only: 397/420 = 94.52% → 405/420 = 96.43%; delta +1.90 pp, CI [−0.98, 5.01] pp chứa 0. Không khẳng định ưu thế FOUND-only có CI loại trừ 0.
- Hoa quả +16.32 pp; hộp −29.67 pp (96.70% → 67.03%). Không tuyên bố tốt hơn ở mọi nhóm vật/quan hệ.
- Phân tầng `relation=direct`: −1.18 pp; phân tầng `family_category=direct_grounding`: −7.97 pp. Đây là hai nhóm khác nhau, không nhầm tên/mẫu số.

### 7.2. Answerability và source

Answerability accuracy 76.80%, macro-F1 0.77516, balanced accuracy 78.62%. Confusion matrix hàng nhãn thật/cột dự đoán theo thứ tự FOUND, AMBIGUOUS, ABSENT, INSUFFICIENT_EVIDENCE:

```text
345    0   31   44
  0  105    0    0
 17    0   85   30
 53    0   57  233
```

- Support bốn lớp: 420/105/132/343. Không nhầm 132 ABSENT answerability với 165 sample không có target mask.
- Có 70 false FOUND trên 580 non-FOUND; chưa phải 70 robot execution sai.
- Answerability multiclass Brier 0.32800, NLL 0.57250, confidence ECE 0.08566; khác event với risk nhị phân.
- Relation active-edge accuracy 505/650 = 77.69%, nhãn proxy; không thay semantic graph exact match.
- Source macro-F1 năm nhóm 0.67739; micro-F1 0.62140; bỏ spatial nhãn yếu thì macro-F1 còn 0.59792.
- Depth F1 0.70391, AP 0.71316; semantic FP 272, occlusion FP 141.
- Threshold source giữ 0.5 cho mọi nhóm trong artifact chính.
- Paired source cùng tên variant tăng trung bình: semantic +0.2202, relation +0.0723, depth +0.4304, occlusion +0.1860; tăng ở 162/200, 130/200, 194/200, 176/200 cặp.
- Semantic CF không mặc định semantic uncertainty dương; toàn bộ semantic CF Test-IID hiện có nhãn FOUND. Không dùng tên variant thay annotation source thật.

### 7.3. Risk, calibration và selective policy

Biến cố lỗi: nhãn thật khác FOUND **hoặc** MAP ngoài target mask. Có 595 lỗi: 580 non-FOUND + 15 localization failure trong FOUND.

| Score | AUROC | AP | Brier | NLL | ECE10 | AURC |
|---|---:|---:|---:|---:|---:|---:|
| Raw 1−p(FOUND), chưa fit | 0.94416 | 0.96223 | 0.10355 | 0.33439 | 0.07203 | 0.26256 |
| Logistic 18 evidence | 0.94546 | 0.96202 | 0.09615 | 0.29848 | 0.04647 | 0.26045 |

Không có scalar-only calibrator đã khóa để cô lập lợi ích thêm source. AP giảm rất nhỏ; không nói mọi metric đều cải thiện.

- Frozen risk threshold 0.0937008824.
- EXECUTE 255/1000: coverage 25.50%; lỗi tổng hợp 6/255 = 2.35%.
- Điểm hit mask trong EXECUTE 250/255 = 98.04%; FOUND và điểm đúng 249/255 = 97.65%. Hai event khác nhau.
- Non-FOUND được execute: 4, gồm 2 ABSENT và 2 INSUFFICIENT_EVIDENCE. Đây là quyết định ảnh, chưa phải wrong-object execution vật lý.
- Policy counts: EXECUTE 255, REOBSERVE 375, ASK_USER 197, ABSTAIN 173.
- Expected-action accuracy 703/1000 = 70.30%, không phải task success/recovery.
- 195 family có ít nhất một sample nhận được không có nghĩa sample coverage là 97.50%.

### 7.4. Vùng tin cậy: phải giữ riêng hai cách chấm

Frozen mass 0.6041831970, alpha 0.1, nominal coverage 90%. Diện tích trung bình 0.5194% lưới 768 ô; trung vị 4/768 ô.

1. **Score coverage chính thức:** `target_required_hd_mass <= frozen mass`, 725/835 = 86.8263%.
2. **Direct region–mask intersection bổ sung:** vùng rời rạc theo policy giao mask, 800/835 = 95.8084%.

Không thay số liệu 86.83% trong artifact gốc bằng 95.81%. Vùng policy nhận cả ô khiến mass đạt/vượt ngưỡng, còn score event yêu cầu cumulative mass đến ô target đầu tiên không vượt ngưỡng. Sáu sample còn có khác biệt thứ tự ô đồng xác suất giữa torch.argsort của evaluator và NumPy argsort của policy.

Hình coverage–area giữ score đã lưu, tái dựng vùng theo policy, đối chiếu tập ô tại mass khóa với decisions. Quét 201 mass chỉ để mô tả test, không chọn lại mass. Giao ít nhất một pixel mask không bảo đảm chứa toàn bộ vật, an toàn 3D hay bảo đảm conformal 90% mọi miền.

### 7.5. Tương quan uncertainty–localization vừa bổ sung

Chỉ xét 420 FOUND thật, giữ 405 khoảng cách 0 và 15 khoảng cách dương; không bỏ outlier/jitter. Pearson tính trên pixel distance gốc, Spearman dùng average ranks cho ties. Symlog chỉ là cách hiển thị trục tung.

| Evidence | Pearson r | Spearman rho |
|---|---:|---:|
| Entropy không gian chuẩn hóa | 0.014807 | 0.011660 |
| 1−peak margin | 0.014595 | −0.011101 |
| Calibrated risk | 0.264108 | 0.225600 |

Đây là thống kê mô tả chưa có CI/p-value theo family. Không biến tương quan dương nhẹ thành tuyên bố phát hiện mọi lỗi localization.

### 7.6. Loss lớn hơn 1 không phải lỗi

Người dùng đã hỏi về train loss >1. Total loss là tổng có trọng số nhiều mục tiêu, không bị giới hạn [0,1]. Epoch 0 train/dev 4.83731/3.65689; epoch 13 1.17249/1.69711; epoch 19 0.94789/1.79097. Train tiếp tục giảm nhưng dev tăng ở cuối run, phù hợp dấu hiệu overfit theo dev criterion; chọn epoch 13 là hợp lý dù train loss còn >1.

## 8. Trạng thái sáu chương LaTeX

Snapshot kiểm tra ngày 02/10/2026; số dòng có thể đổi sau chỉnh sửa:

| Chương | File | Nội dung/trạng thái |
|---|---|---|
| 1 | `latex/Chapter1/chapter1.tex` | Tổng quan; 171 dòng, chưa có figure |
| 2 | `latex/Chapter2/chapter2.tex` | Lý thuyết và liên quan; 739 dòng, 1 bảng, chưa có figure |
| 3 | `latex/Chapter3/chapter3.tex` | Phương pháp; 1473 dòng, 17 section, 9 figure |
| 4 | `latex/Chapter4/chapter4.tex` | Thiết lập; 1412 dòng, 10 section, 18 bảng, 3 figure |
| 5 | `latex/Chapter5/chapter5.tex` | Kết quả; 1925 dòng, 13 section, 30 bảng, 19 figure (17 có ảnh, 2 placeholder) |
| 6 | `latex/Chapter6/chapter6.tex` | Kết luận; 747 dòng, 3 section, chưa có figure |

Chương 1 có ba trục câu hỏi nghiên cứu, Chương 5 cụ thể hóa thành RQ1–RQ5: grounding, answerability/spatial uncertainty, source uncertainty, calibration, selective policy. Không coi đây là hai đề tài khác nhau.

Nhãn chapter quan trọng: Chương 4 `sec:experimental-setup`, Chương 5 `sec:experiments`, Chương 6 `sec:conclusion`. Chương 6 tham chiếu bảng `tab:results-rq-summary` ở Chương 5.

Các label figure còn hậu tố `-pending` có thể đã có hình thật, ví dụ training/confusion/source delta/correlation/coverage area. Đọc nội dung figure thay vì suy trạng thái từ tên label; không đổi label tùy tiện làm hỏng ref.

Preamble cần hỗ trợ tiếng Việt và các package `amsmath, amssymb, graphicx, booktabs, tabularx`; bibliography ở `latex/references.bib`. Workspace hiện chưa có master document/trình biên dịch LaTeX local được xác nhận. **Điều này không phủ nhận việc người dùng đã biên dịch được ảnh trên Prism.**

### Quy ước nguồn ảnh Prism đã được yêu cầu

```latex
\includegraphics[width=\textwidth,height=0.62\textheight,keepaspectratio]{hinhanh/fig10_reliability_test_iid.png}
```

Chỉ đổi đường dẫn thành `hinhanh/<basename>.png`; không thay mọi tùy chọn bằng ví dụ trên. Figure minipage phải giữ `width=\linewidth` nếu đó là tùy chọn gốc. Caption/label/footnote giữ nguyên khi nhiệm vụ chỉ đổi nguồn.

Trong lần kiểm tra local, các tham chiếu đã ở dạng phẳng như trên, nhưng không có `latex/hinhanh` chứa ảnh. Người dùng quản lý bản trên Prism và đã xác nhận biên dịch ảnh thành công; không tự copy/move ảnh hoặc sửa path ngược về local nếu chưa được yêu cầu.

## 9. Hình và script hiện có

Thư mục nguồn chính: `new/hinhanh/`, mỗi hình trong subfolder riêng có PNG và PDF. Không dùng thư mục tổng hợp phụ `new/hinhanh/fig/` làm nguồn chuẩn mặc định; nó có thể không đủ hoặc chứa bản cũ.

| Mục | Subfolder | Basename |
|---|---|---|
| Query graph | `04_language_query_graph` | `fig04_language_query_graph_v2` |
| Fusion | `05_relation_conditioned_fusion` | `fig05_relation_conditioned_fusion_v2` |
| Target heatmap | `06_target_heatmap` | `fig06_target_heatmap_real_rgb` |
| Multimodal distribution | `07_multimodal_distribution` | `fig07_multimodal_spatial_distribution` |
| Source cases | `08_source_uncertainty` | `fig08_source_uncertainty_cases` |
| Calibrator | `09_independent_calibrator` | `fig09_independent_calibrator_v2` |
| Reliability | `10_reliability` | `fig10_reliability_test_iid` |
| Risk–coverage | `11_risk_coverage` | `fig11_risk_coverage_test_iid` |
| Confidence region | `12_confidence_region` | `fig12_confidence_region_true_rgb` |
| Four decisions | `14_decision_cases` | `fig14_four_policy_decisions` |
| Paired failures | `15_failure_cases` | `fig15_paired_success_and_failure` |
| Object strata | `16_stratified_object` | `fig16_grounding_by_object` |
| Relation strata | `17_stratified_relation` | `fig17_grounding_by_relation` |
| Variant strata | `18_stratified_variant` | `fig18_grounding_by_variant` |
| Confusion matrix | `19_answerability_confusion` | `fig19_answerability_confusion_test_iid` |
| Training curves | `20_training_curves` | `fig20_pcrau_training_curves` |
| Paired source delta | `21_source_delta` | `fig21_pcrau_paired_source_delta` |
| Uncertainty–localization | `22_uncertainty_localization` | `fig22_pcrau_uncertainty_localization` |
| Coverage–area | `23_coverage_area` | `fig23_pcrau_coverage_area` |

Builder:

- `new/hinhanh/build_figures.py`: 16 hình kết quả, đọc prediction/history đã lưu.
- `new/hinhanh/build_method_figures.py`: ba sơ đồ 04/05/09.
- `.conda-roborefer/bin/python` có Matplotlib/NumPy/PIL; system Python không mặc định có các package này. Môi trường từng không có SciPy; builder dùng NumPy, gồm tính rank có ties và neighborhood max thay dependency SciPy.

Khi thực sự được yêu cầu tái dựng toàn bộ:

```bash
cd /home/dhcn/ur_ws/src/myproject
.conda-roborefer/bin/python new/hinhanh/build_figures.py
.conda-roborefer/bin/python new/hinhanh/build_method_figures.py
```

Không cần chạy các lệnh này chỉ để bàn giao. Builder ghi lại hình, không train/infer lại. Tên hiển thị các hình đã đổi V2 → P-CRA-U; basename lịch sử vẫn giữ v2.

Các ảnh ngoài builder:

- Pipeline: `plan/pipeline.png`, tham chiếu Prism là `hinhanh/pipeline.png`.
- Bốn panel capture/annotation Chương 4: `new/test_iid/dataset/report_assets/real_capture_qc/canary_000/v211iid_family_000141/`, gồm `rgb.png`, `depth_metric_visualization.png`, `semantic_instance_labels.png`, `target_anchor_overlay.png`. Không lấy nhầm family vì basename trùng nhiều thư mục.
- Dashboard: `new/demo_gazebo/runs/gazebo_v2_20261001T074056Z_46219/live/last_dashboard.png`.
- Montage audit: `new/demo_gazebo/runs/iid_pose_audit_20261001T080100Z_49405/audit_montage.png`.

Hai placeholder còn lại trong Chương 5:

1. Anchor heatmap / relation-edge evidence trên RGB-D: prediction lưu hiện thiếu anchor grid, cần export evidence thật nếu được yêu cầu; không thay bằng sơ đồ kiến trúc rồi gọi là kết quả.
2. Pick-and-place khép kín: thiếu thí nghiệm thao tác, không chỉ thiếu ảnh. Không dựng chuỗi grasp/lift/place giả.

Chương 1 overview và Chương 2 sơ đồ RoboRefer có thể là hình bổ sung sau; chưa tự tạo trong bàn giao.

## 10. Demo Gazebo và dashboard 2×3 trong kế hoạch

Mục 16 bản kế hoạch đề xuất dashboard sáu ô:

```text
RGB + Depth         | Language graph      | Predicted graph
Spatial heatmap     | Source uncertainty  | Calibrated action
```

Đây là thiết kế đề xuất, **không mặc định dashboard 2×3 đã triển khai**. Demo hiện có dashboard bốn panel và audit năm frame. Nếu session mới được yêu cầu xây 2×3, kiểm tra code/prediction thật, thể hiện rõ slot evidence thay vì bịa graph parser, và đánh dấu oracle overlay là evaluator-only.

Run audit dùng pose camera Test-IID trên cảnh demo riêng:

`new/demo_gazebo/runs/iid_pose_audit_20261001T080100Z_49405/`.

- Năm frame có MAP (150,210), target/interior hit 5/5; maximum jitter 0 pixel trên lưới lượng tử hóa, không phải chứng minh sai số 3D bằng 0.
- Checker RGB-D độc lập 5/5, nhưng chỉ kiểm tra quả táo trong cảnh cụ thể (đỏ/tròn/depth), chưa phải object-identity gate tổng quát.
- Risk khoảng 0.29459–0.32682, cao hơn frozen threshold 0.09370.
- Decision ASK_USER 5/5; pre-handoff không đạt; `moveit_connected=false`, `robot_motion_commanded=false`.
- Prediction/checker được khóa trước semantic-label hậu kiểm. Không đưa oracle target vào inference/control.
- Năm frame cùng cảnh không phải năm trial robot độc lập.
- UR3/ROS2 Humble/Gazebo Fortress/MoveIt2/D435i là bối cảnh hạ tầng; chưa có kết quả robot vật lý hoặc khép kín do model chính điều khiển.

Các scene preview generated riêng không phải lý do sửa world/URDF nguồn. Giữ đúng RGB-D/timestamp dùng inference; không đặt điểm của frame cũ lên frame mới để cấp target chuyển động.

## 11. Hạn chế khoa học không được xóa để làm đẹp báo cáo

- P-CRA-U chính có một seed; chưa có ablation retrain cô lập depth/relation/anchor/source/ranking cùng protocol.
- V3 nhiều thay đổi đồng thời không thay ablation hoặc chứng minh nhiều seed cho V2.
- Anchor localization/semantic graph exact match/oracle graph upper bound chưa có evaluator đầy đủ.
- Spatial weak label và relation proxy phải được công bố.
- Subgroup grounding cải thiện không đều, box hồi quy mạnh; FOUND-only CI chứa 0.
- Source semantic/occlusion còn nhiều false positive; không coi mọi score là attribution đúng.
- Calibration/source/selective metrics chưa có CI theo family đầy đủ; subgroup CI chưa hiệu chỉnh multiple comparisons.
- Hai event coverage vùng không đồng nhất; chưa có cơ sở bảo đảm phân phối bất kỳ/robot thật.
- Reobserve recovery, hiệu quả hỏi người dùng và budget exhaustion chưa đo qua chuỗi tương tác thật.
- Cache-only sidecar latency không được so trực tiếp với RoboRefer sinh token rồi gọi là end-to-end realtime.
- Nếu dùng lỗi Test-IID để chỉnh model vòng sau, cần bộ đánh giá độc lập mới; bộ test đã xem không còn là test chưa sử dụng.

## 12. Trạng thái file và cách tiếp tục an toàn

Branch kiểm tra hiện tại: `results/best-v2-test-iid-20260930`. LaTeX, hình và một số plan đang local/untracked; có thay đổi/xóa file khác của người dùng. Không coi mọi thứ đã nằm trên GitHub và không dọn worktree.

`latex/Chapter5/derive_statistics.py` và `derived_statistics.json` là phân tích phụ đã tạo trước đây. Người dùng từng phản đối việc tạo chúng khi yêu cầu viết chương; chúng vẫn tồn tại. Không tự xóa, tạo thêm hay chạy lại nếu không liên quan nhiệm vụ.

Yêu cầu đổi đường dẫn ảnh xuất hiện trước các lần ngắt turn. **Kiểm tra source hiện tại cho thấy đường dẫn đã ở dạng Prism và người dùng xác nhận ảnh đã biên dịch được. Không tự lặp lại việc sửa hoặc quy lỗi còn thiếu ảnh local thành lỗi Prism.**

Session mới nên bắt đầu bằng đọc file này, đối chiếu các file liên quan, tóm tắt ngắn đã nắm phạm vi và hỏi/đợi nhiệm vụ tiếp theo nếu chưa được giao. Không tự thực hiện roadmap, retrain, dashboard 2×3 hay robot handoff chỉ vì chúng xuất hiện trong kế hoạch.
