# RefCOCO+ UNC val — 100 ảnh / 300 câu

Ngày 07/10/2026. Pilot theo ảnh, không toàn bộ val.

## Protocol

- Source `jxu124/refcoco-benchmark`, revision `2f2f892835bbf44b4db81f1c4ad6d46a3e9e4359`, file `data/refcoco_plus_unc_val-00000-of-00001-2c68b2dca7cc4d2b.parquet`, SHA256 `c7949870f569f09cba78286fd29d979e8e29fe5af1e69b30d1a709d88125738b` đã đối chiếu LFS.
- Full val 1500 ảnh, 3805 references, 10758 câu; chỉ chạy100ảnh/300câu/194references.
- Sampling khóa trước forward: seed24082026, SHA256 rank100ảnh trong 1487ảnh có≥3câu; rank3câu/ảnh. Loại 13ảnh thiếu3câu theo eligibility, không theo model score.
- Giữ raw phrase, chỉ thêm Locate/mạo từ và dấu chấm như pilot RefCOCO. Không dùng category/bbox để viết câu hoặc chọn prediction.
- Primary frozen RoboRefer-2B-SFT RGB-D640×480, DepthAnythingV2 Large infer từ RGB gốc, greedy max_new_tokens128. Native RGB-only là đối chứng riêng. Không so hai nhánh rồi chọn theo nhãn.
- Sidecar Tasks60 chạy forward + MC20 trên cùng features/câu; primary adapter giữ điểm RoboRefer. Primary risk/threshold null; decision REVIEW_UNCALIBRATED khi valid point.
- Metric point-in-GT-bbox xywh trong ảnh gốc; không IoU hoặc point-in-mask. GT chỉ đọc trong report sau lưu predictions. CI bootstrap10000 theo100image groups.
- 96/100ảnh trùng imageID với pilot RefCOCO cũ; câu/annotation là RefCOCO+. Không xem hai pilot là hai bộ cảnh độc lập hoặc cộng mẫu như600cảnh.
- Không train/fit/tune threshold; không đổi active profile/bundle cũ hoặc robot motion.

## Kết quả

| Pipeline / nguồn target | Đúng | Point-in-box | 95% CI theo ảnh |
|---|---:|---:|---:|
| RoboRefer RGB-D primary | 273/300 | 91.00% | [87.33; 94.33]% |
| Sidecar Tasks60, MAP riêng | 92/300 | 30.67% | [25.00; 36.67]% |
| RoboRefer RGB-only, ảnh gốc | 265/300 | 88.33% | [83.67; 92.67]% |

Primary so Sidecar: sửa187, làm sai6; delta +60.33pp, CI [54.0, 66.66666666666666]pp.
Official decoder crosscheck 300/300. Status primary `{'VALID': 300}`.
Scope geometry `{'DIRECT_BYPASS': 205, 'PARSE_UNSUPPORTED': 95}`. Semantic matching missing diagnostic 300.0/300, không là accuracy semantic.

## Case thật và lỗi

![Case thật](../new/outputs/refcoco_plus_val_pilot300_20261007/cases.png)

Xanh lá: GT bbox chỉ dùng hậu kiểm; cyan: primary; đỏ: Sidecar. Hình gồm các lỗi primary và ca được sửa/làm sai khi đổi nguồn target.
Lưu toàn bộ 27 lỗi primary trong `primary_errors.json`; không chỉ chọn ảnh đẹp.

## Đánh giá và giới hạn

- Điểm primary là năng lực grounding kế thừa từ RoboRefer. Sidecar không thay/veto điểm này; chưa chứng minh uncertainty cải thiện grounding.
- Pilot RefCOCO cũ primary 281/300 (93.67%). RefCOCO+ mới 273/300 (91.00%). Hai kết quả khác câu/targets; không diễn giải chênh lệch như paired model improvement.
- Risk/coverage primary chưa được hiệu chuẩn. Không lấy risk Sidecar đánh giá như risk của RoboRefer, không báo answerability accuracy vì benchmark không có nhãn4states.
- Scope verifier hạn chế trái/phải với cú pháp đã hỗ trợ; không gọi mô tả unsupported là đã kiểm chứng. Benchmark này chưa xác nhận năm task uncertainty, depth mét, occlusion amodal hoặc an toàn robot.
- Không phải tái hiện bảng paper theo full split hoặc leaderboard. Model/prompt/RGB-D preprocessing được ghi rõ; pilot không dùng để train/fit/tune.

## Artifacts

`new/outputs/refcoco_plus_val_pilot300_20261007/`: source, protocol, runtime/evaluator manifests, inputs/features, backbone_predictions, predictions Sidecar, primary_predictions, paired_evaluator, summary, primary_errors, cases và final status.
Runner `new/scripts/evaluate_refcoco_plus_pilot.py`; worker `new/scripts/refcoco_plus_pilot_worker.py`.

Nguồn: [REFER tác giả](https://github.com/lichengunc/refer), [mirror dataset](https://huggingface.co/datasets/jxu124/refcoco-benchmark), [RoboRefer point-in-box protocol](https://arxiv.org/html/2506.04308v2#S4.SS3).

## Audit kết quả sau chấm

- Có27câu sai trên23ảnh; 77/100ảnh đúng cả3câu, 99/100ảnh đúng ít nhất1câu. Đây là diagnostic theo ảnh; metric chính vẫn là273/300câu.
- 2 lỗi nằm cách bbox≤2pixel ảnh gốc: 86146 (0.25px), 14966 (1.43px). Giữ nguyên27lỗi trong score chính, không tăng tolerance hoặc sửa annotation. Khoảng cách gần bbox không tự chứng minh nằm đúng mask.
- Case46312 `sitting w green umb.`: điểm primary nằm trên ô xanh, trong khi GT bbox là người ngồi bên dưới. Đây là lỗi chọn vật được mô tả; đã xem trực tiếp hình.
- Case84801 `Cusions matching couch`: primary chọn đệm phía bên kia trong cùng ảnh, ngoài bbox GT. Ảnh có các vật cùng họa tiết, nên ghi nhận sai reference theo annotation; chưa kết luận mọi lỗi là không nhận ra loại vật.
- Case86146 `fountain drink`: điểm [95,0], bbox bắt đầu y=0.25; sai theo metric chính dù nằm sát cốc trong hình. Đây là ví dụ cần phân biệt lỗi biên với chọn nhầm instance.
- Điểm giữa ảnh diagnostic đạt104/300 (34.67%), không dùng để chọn prediction. Primary91.00% cho thấy grounding cao hơn heuristic này.
- Primary RGB-D hơn RGB-only native2.67pp, CI[0.00,5.33]pp; hai nhánh khác cả depth và độ phân giải/preprocessing, không quy riêng lợi ích cho depth.
- Đối chiếu theo300câu: primary hơn Sidecar60.33pp, CI[54.00,66.67]pp; sửa187/làm sai6. Điều này tiếp tục ủng hộ dùng grounding backbone cho biến thể hiện tại, không chứng minh verifier hoặc MC tăng accuracy.
- Đã xem figure trực tiếp;300/300tọa độ khớp parser điểm official. Hash source code, bundle, active profile và các pilot cũ không đổi.
