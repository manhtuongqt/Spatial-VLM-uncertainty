# RefCOCO UNC val — pilot 300 câu

Ngày 07/10/2026. **Subset pilot, không phải toàn bộ val hoặc submission leaderboard.**

## Dữ liệu và protocol

- Nguồn mirror `jxu124/refcoco-benchmark`, revision `2f2f892835bbf44b4db81f1c4ad6d46a3e9e4359`; parquet SHA256 `9f2f0057259be527249e0d260511ace5869b9055c23b65eb951d01e6e6920424`, đã đối chiếu LFS.
- Full UNC val: 1.500 ảnh, 3.811 references, 10.834 câu. Pilot 100 ảnh × 3 câu = 300 câu, 202 references.
- Sampling khóa trước forward: rank SHA256(seed:image:id) chọn 100 ảnh trong 1459 ảnh có ≥3 câu; rank SHA256(seed:sentence:id) chọn 3 câu mỗi ảnh. 41 ảnh có <3 câu không đủ điều kiện pilot. Không lọc theo model score. Đây là sampling cân bằng ảnh, không random uniform 300 câu trong toàn val.
- Giữ nguyên nội dung raw phrase; thêm lệnh Locate và mạo từ the khi cần. Không rút gọn quan hệ/dùng class GT để viết prompt.
- P-CRA-U dùng RGB và pseudo-depth 640×480. RoboRefer có đối chứng RGB-D cùng đầu vào và RGB-only ở ảnh gốc. Greedy max_new_tokens=128.
- MAP được chuyển về hệ tọa độ ảnh gốc, chấm point-in-box từ bbox xywh GT. Annotation được tách thành evaluator manifest khi chuẩn bị dữ liệu; inference không đọc manifest này, và bước chấm chỉ chạy sau khi lưu predictions; không IoU, không PIT theo mask.
- Frozen Tasks60 bundle, threshold 0.37876498603729764; T=20; seed 24082026. Không train/fit/chọn ngưỡng bằng pilot.

## Kết quả

| Pipeline | Point-in-box | Coverage | Selective grounding risk |
|---|---:|---:|---:|
| P-CRA-U Tasks60 MC | 101/300 (33.67%) | 0.67% | 100.00% (2/2) |
| RoboRefer RGB-D, matched640 | 281/300 (93.67%) | — | — |
| RoboRefer RGB-only, ảnh gốc | 277/300 (92.33%) | — | — |

Risk metrics (event MAP ngoài bbox, khác event composite đã fit):

```json
{
  "accepted": 2,
  "coverage": 0.006666666666666667,
  "errors": 2,
  "correct_acceptances": 0,
  "selective_grounding_risk": 1.0,
  "Brier": 0.34153048871784275,
  "NLL": 1.8722838562634392,
  "ECE10": 0.34256882634128244,
  "AURC": 0.6962569952267134,
  "error_AUROC": 0.4898253644459924,
  "error_AUPRC": 0.671501549590224
}
```

Scope: `{'PARSE_UNSUPPORTED': 193, 'DIRECT_BYPASS': 107}`. Semantic phrase matching missing: 299.0/300, không đồng nghĩa semantic classification accuracy=0%.

Paired so matched640: fixed 7, broken 187; delta -60.00 pp; CI theo ảnh [-66.33333333333333, -53.333333333333336] pp. PIT P-CRA-U CI [27.666666666666668, 39.666666666666664]%.

## Case thật

![Cases](../new/outputs/refcoco_val_pilot300_20261007/cases.png)

Xanh lá: bbox GT hậu kiểm; đỏ: P-CRA-U; cyan: RoboRefer RGB-D matched640.

![Risk coverage](../new/outputs/refcoco_val_pilot300_20261007/risk_coverage.png)

## Diễn giải và giới hạn

- Pilot đo grounding 2D trên ảnh thực. Verifier chỉ hoạt động nếu parser hỗ trợ; không coi unsupported/direct là đã kiểm chứng quan hệ.
- Box bao gồm cả nền/vật khác nên point-in-box có thể đạt khi điểm ngoài mask của target. Không so trực tiếp với 6/100 point-in-mask trên RefSpatial.
- Ngay cả nếu point-in-box cao, chưa xác nhận depth theo mét, anchor binding, amodal completion hoặc cả năm nhóm uncertainty.
- Không có nhãn answerability bốn trạng thái để chấm accuracy/macro-F1 hoặc event composite gốc. Risk metrics là kiểm tra transfer sang event grounding box.
- Đây không phải tái hiện chính xác điểm bảng paper: subset cân bằng ảnh, prompt/decoding/RGB-D được ghi rõ. Không lấy số của 300 câu như điểm toàn val; không dùng benchmark đã xem để tune rồi gọi test độc lập.
- Checkpoint, calibrator, threshold và active profile giữ nguyên; không phát lệnh robot.

## Artifacts

`new/outputs/refcoco_val_pilot300_20261007/`: protocol, runtime/evaluator manifests tách biệt, predictions, paired_evaluator.json, summary.json, figures, logs.

Nguồn: https://github.com/lichengunc/refer ; https://huggingface.co/datasets/jxu124/refcoco-benchmark ; https://arxiv.org/html/2506.04308v2#S4.SS3 .

## Kết luận từ pilot

- P-CRA-U Tasks60 đạt 101/300 (33,67%), thấp hơn RoboRefer RGB-D matched640 281/300 (93,67%) và RGB-only native 277/300 (92,33%). Khoảng cách matched là -60,00 pp; CI bootstrap theo ảnh [-66,33; -53,33] pp. Pilot này xác nhận giới hạn chuyển miền của target head Sidecar, không chứng minh backbone RoboRefer kém.
- Parser nhận 107 direct và 193 unsupported. Không câu nào kích hoạt verifier hình học trái/phải; vì vậy không gọi kết quả này là kiểm chứng verifier. Semantic phrase matching missing 299/300, gồm cả ca parse thất bại và phrase không khớp catalog. Đây là hạn chế giao diện/catalô, không phải một phép đo accuracy của semantic head.
- Tasks60 và Live44 đều nhận 2 câu; cả hai MAP đều ngoài bbox đúng. Hai câu này cùng một ảnh, không phải 2 cảnh độc lập. Không có correct acceptance trong pilot. Mức risk 100% là 2/2 quan sát, không là ước lượng chính xác của risk trên toàn RefCOCO.
- Tasks60 risk error-AUROC 0,4898, gần ngẫu nhiên; Brier 0,3415, NLL 1,8723, ECE10 0,3426. Các giá trị này không ủng hộ khả năng transfer calibration. AURC 0,6963 được diễn giải theo prevalence lỗi 66,33%, không theo chuẩn phổ quát AURC<0,1. Không có risk ties chính xác trong 300 câu.
- Thêm MC không tạo lợi ích selective trên pilot này: cùng 2 lượt nhận sai với Live44. Đây vẫn là so sánh bundle, chưa ablation để quy riêng nguyên nhân cho MC hoặc auxiliary heads.
- Case thực tế đáng chú ý: câu “top banana” dự đoán vào banana phía dưới, risk vẫn dưới ngưỡng nên nhận. Nhận ra loại đồ vật chưa đủ để đáp ứng thuộc tính/vị trí được yêu cầu.
- Giữ nguyên các số IID và RefSpatial trước đây. 33,67% ở đây không phải cải thiện trực tiếp so 6% RefSpatial, vì ảnh, câu và event bbox/mask khác nhau. Không đưa điểm pilot vào cột full-val của bảng paper.

## Diagnostic bổ sung

- Heuristic điểm chính giữa ảnh: 106/300 (35.33%). Điểm ngẫu nhiên đều có expected point-in-box 20.01% từ diện tích bbox. Các reference này giúp đánh giá mức khó của event box; chúng không dùng nhãn để sinh prediction P-CRA-U.
- Direct: {'n': 107, 'pcrau_hits': 35, 'backbone_matched_hits': 101}; unsupported: {'n': 193, 'pcrau_hits': 66, 'backbone_matched_hits': 180}. Chỉ là phân tầng sau chấm, không phải chọn lại subset chính.
- Constant risk bằng prevalence của chính pilot có Brier 0.223322, NLL 0.638800; chỉ diagnostic hậu kiểm, không baseline triển khai hoặc calibrator fit.
- Crosscheck serialization `jxu124/refcoco` revision `9fba8200c5326e996f789191f095bd464ef1d09e`: 300/300 bbox, raw phrase, image_id, split khớp. Đây là hai cách đóng gói cùng nhãn gốc, không hai nguồn gán nhãn độc lập.

## Hai câu nhận sai

```json
[
  {
    "id": 141432,
    "image_id": 3518,
    "ref_id": 49729,
    "ann_id": 1042682,
    "bbox_xywh": [
      141.7,
      20.68,
      359.02,
      144.57
    ],
    "category_id": 52,
    "phrase": "top bananna",
    "image_size": [
      640,
      427
    ],
    "pcrau_point": [
      270,
      329
    ],
    "pcrau_hit": false,
    "backbone_matched_points": [
      [
        255,
        97
      ]
    ],
    "backbone_matched_hit": true,
    "backbone_native_rgb_points": [
      [
        250,
        98
      ]
    ],
    "backbone_native_rgb_hit": true,
    "risk": 0.24058952699926792,
    "action": "EXECUTE",
    "live44_risk": 0.10964483558266126,
    "live44_action": "EXECUTE",
    "scope": "DIRECT_BYPASS",
    "semantic_phrase_missing": 1.0
  },
  {
    "id": 141430,
    "image_id": 3518,
    "ref_id": 49729,
    "ann_id": 1042682,
    "bbox_xywh": [
      141.7,
      20.68,
      359.02,
      144.57
    ],
    "category_id": 52,
    "phrase": "top banana",
    "image_size": [
      640,
      427
    ],
    "pcrau_point": [
      330,
      329
    ],
    "pcrau_hit": false,
    "backbone_matched_points": [
      [
        254,
        97
      ]
    ],
    "backbone_matched_hit": true,
    "backbone_native_rgb_points": [
      [
        237,
        97
      ]
    ],
    "backbone_native_rgb_hit": true,
    "risk": 0.08185174477153938,
    "action": "EXECUTE",
    "live44_risk": 0.0346794219468438,
    "live44_action": "EXECUTE",
    "scope": "DIRECT_BYPASS",
    "semantic_phrase_missing": 1.0
  }
]
```

![Two accepted errors](../new/outputs/refcoco_val_pilot300_20261007/accepted_errors.png)

Hai câu cùng ảnh/target reference, một câu giữ nguyên typo “bananna” của annotation. Không sửa câu để làm đẹp số; cả hai dự đoán banana phía dưới thay vì phía trên.
