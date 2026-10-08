# RefSpatial-Bench Location — đánh giá bundle đóng băng

Ngày 07/10/2026. Chấm đủ 100 mẫu; không train, không fit calibration, không chọn lại ngưỡng.

## Protocol

- Dataset `BAAI/RefSpatial-Bench`, revision `5cb4c34a36c09962442fc5b76e2e462b75008329`.
- Primary: nguyên văn `prompt`. RoboRefer thêm suffix định dạng chính thức; Sidecar xuất heatmap nên không thêm suffix.
- RGB và pseudo-depth Depth Anything V2 Large resize 640×480 cho cả hai. Điểm được đưa về kích thước mask gốc; masks chỉ dùng ở evaluator.
- Secondary: chỉ thay tiền tố `Please point out/to` bằng `Locate`; không rút gọn các quan hệ/cụm danh từ. Không dùng để chọn primary sau khi thấy kết quả.
- Risk hậu kiểm là MAP ngoài mask, khác event composite đã fit trên dữ liệu đồ án. Không có GT bốn trạng thái answerability.

## Kết quả

| Pipeline | PIT | Coverage | Selective grounding risk |
|---|---:|---:|---:|
| RoboRefer-2B-SFT greedy | 47.00% | — | — |
| RoboRefer-2B-SFT, ảnh nguyên kích thước | 50.00% | — | — |
| P-CRA-U Tasks60 MC, primary | 6/100 | 1.00% | 100% |
| Live44, cùng MAP | 6/100 | 4.00% | 100% |
| Prefix-only diagnostic | 6/100 | 1.00% | 100% |

Primary scope: `{'PARSE_UNSUPPORTED': 100}`; secondary scope: `{'DIRECT_BYPASS': 10, 'PARSE_UNSUPPORTED': 90}`.
Primary semantic phrase missing: 100.0/100; secondary: 100.0/100.

Calibration/ranking (event grounding error):
```json
{
  "accepted": 1,
  "coverage": 0.01,
  "correct_acceptances": 0,
  "errors": 1,
  "selective_risk": 1.0,
  "risk_wilson95": [
    0.20654329147389294,
    1.0
  ],
  "Brier": 0.07164432760866786,
  "NLL": 0.4210070137457855,
  "ECE10": 0.07924370165894931,
  "AURC": 0.9666998570385616,
  "error_AUROC": 0.36879432624113473,
  "error_AUPRC": 0.9270308612209992
}
```

Paired cases tốt hơn RoboRefer: [20, 25]; kém hơn: [1, 2, 3, 7, 8, 11, 12, 13, 15, 18, 28, 33, 34, 36, 37, 38, 40, 41, 42, 43, 44, 47, 48, 49, 50, 53, 56, 59, 61, 63, 64, 65, 66, 68, 74, 77, 81, 87, 92, 93, 94, 95, 96].
Delta: -41.00 pp; bootstrap descriptive CI: [-51.0, -31.0].

## Case thật

![Actual cases](../new/outputs/refspatial_location_frozen_20261007/cases.png)
Xanh lá: mask GT hậu kiểm; đỏ: P-CRA-U; cyan: RoboRefer ở điều kiện chung 640×480.
![Risk coverage](../new/outputs/refspatial_location_frozen_20261007/risk_coverage.png)

## Giới hạn và cách diễn giải

- Đây là đánh giá chuyển miền sang ảnh thực/câu lệnh công khai, không phải train lại trên benchmark. Backbone đã được huấn luyện RefSpatial; không tuyên bố hoàn toàn unseen đối với cả VLM.
- Không chấm anchor hit, depth MAE theo mét hoặc amodal completion vì benchmark không cung cấp các nhãn này. Không chứng minh cả năm nhóm uncertainty chỉ từ PIT.
- Verifier chỉ hoạt động khi parser hỗ trợ. Unsupported vẫn có MAP/risk từ pipeline, nhưng không được diễn giải là đã kiểm chứng quan hệ.
- Nếu coverage=0 thì selective risk không xác định, không phải risk=0 hoặc bảo đảm an toàn.
- Nếu điểm thấp hơn IID, đó là bằng chứng giới hạn chuyển miền. Không dùng kết quả này để sửa ngưỡng rồi báo lại như test độc lập.
- Public benchmark score tương thích point-in-mask; chưa là submission leaderboard. Resize/greedy/cap được ghi rõ; không đồng nhất điểm với cấu hình RFT/8B trong paper.

## Artifacts

- `new/outputs/refspatial_location_frozen_20261007/summary.json`, `paired_evaluator.json`, predictions, protocol, source hashes.
- Sources: https://huggingface.co/datasets/BAAI/RefSpatial-Bench ; https://github.com/Zhoues/RoboRefer/tree/main/Evaluation .

## Kết luận thực nghiệm

- Pipeline đã chạy đủ 100 câu, nhưng khả năng grounding chuyển miền chưa đạt: P-CRA-U 6/100, RoboRefer cùng đầu vào 47/100 và RoboRefer ảnh gốc 50/100. Hai ca P-CRA-U tốt hơn đối chứng (#20, #25), 43 ca kém hơn. Đây là lỗi/hạn chế của toàn cấu hình đóng băng trên miền mới, chưa là ablation để quy cho riêng loss, parser hoặc MC.
- Nguyên câu gốc: parser unsupported 100/100. Đổi tiền tố: chỉ 10 direct, 90 unsupported; không câu nào chạy verifier hình học. PIT vẫn 6/100, nên tiền tố không giải thích được toàn bộ thất bại.
- Semantic phrase matching missing 100/100 kể cả diagnostic. Với câu nguyên gốc, missing còn do parse thất bại; với 10 câu direct đã parse, phrase cũng không khớp chính xác catalog 22 lớp. Không diễn giải trường missing này là semantic classification accuracy=0%.
- Tasks60 giảm lượt nhận từ 4 xuống 1 so Live44, nhưng lượt duy nhất vẫn sai. Ba lượt nhận sai bị chặn thêm không đủ để kết luận MC giải quyết chuyển miền. Không đạt ngân sách thực nghiệm 7,5% ở phần grounding; chưa có nhãn để chấm event composite gốc.
- Risk error-AUROC 0,3688, AURC 0,9667: ranking không hữu ích trên tập này. Chỉ có 6 ca MAP đúng nên các ước lượng phân biệt có độ bất ổn cao.
- ECE 0,0792 và Brier 0,0716 không đủ để kết luận hiệu chuẩn tốt: prevalence lỗi grounding là 94%. Reference hằng lấy prevalence trực tiếp từ test có Brier 0,0564 (chỉ là diagnostic hậu kiểm, không phải baseline triển khai). Risk cao gần khắp tập dễ có sai lệch tổng thể nhỏ nhưng vẫn không chọn được ca đúng.
- Các kết quả IID cũ không bị sửa; kết quả benchmark này cho thấy giới hạn của Sidecar được học trên dữ liệu đồ án khi sang cảnh thực và câu lệnh mới. Kiến trúc có cơ chế trong phạm vi đã triển khai; chưa có bằng chứng tổng quát hóa sang spatial referring công khai.

## Kiểm tra độc lập và khoảng tin cậy theo ảnh

Evaluator chính thức `RoboRefer/Evaluation/summarize_acc.py` tái hiện đúng bốn điểm số: `{'RoboRefer_matched640': {'score': 0.47, 'evaluated': 100}, 'RoboRefer_native': {'score': 0.5, 'evaluated': 100}, 'P_CRA_U_primary': {'score': 0.06, 'evaluated': 100}, 'P_CRA_U_prefix': {'score': 0.06, 'evaluated': 100}}`.

100 câu có 61 RGB SHA256 khác nhau. Bootstrap 10.000 lượt theo nhóm ảnh: CI PIT P-CRA-U [2.0202020202020203, 10.526315789473683]%; paired delta so RoboRefer cùng kích thước [-51.64835164835166, -30.43478260869565] pp. Đây là CI bổ sung thay cho giả định 100 câu độc lập; ảnh khác SHA256 vẫn có thể thuộc cùng cảnh vật lý. Quy tắc grouping được lưu trước khi mở masks chấm kết quả trong `bootstrap_cluster_addendum.json`.

## Tái tạo

```bash
# Stage prepare đã hoàn tất; không chạy lại để ghi đè protocol.
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 -s -B new/scripts/refspatial_location_worker.py
.conda-roborefer/bin/python3.10 -B new/scripts/evaluate_refspatial_location.py infer
.conda-roborefer/bin/python3.10 -B new/scripts/evaluate_refspatial_location.py report
```

Các stage inference từ chối ghi đè predictions đã có. Reproduction cần checkout cùng source và một output root mới; script hiện khóa đường dẫn run này. Source script/worker, bundle và input hashes nằm trong protocol/provenance.

## Lượt nhận sai duy nhất

![Accepted error](../new/outputs/refspatial_location_frozen_20261007/accepted_error.png)

Câu #35: “Please point out the silver box closest to the gray bottle.” Điểm P-CRA-U ở [530,70] ngoài mask target; risk 0,265540 thấp hơn ngưỡng 0,378765, nên policy nhận ở mức nhận thức. Parser báo unsupported, không có verifier hình học cho quan hệ closest-to. Đây là evidence-only pipeline, chưa có hard veto cho unsupported. Không phát lệnh robot.

### Quyết định sau đánh giá

Giữ nguyên bundle và số liệu này như kết quả chuyển miền. Nếu tiếp tục, nên audit binding/target map trên các ca direct và đối chiếu điểm RoboRefer trước; đổi tiền tố không sửa được chất lượng MAP. Chưa nên tune calibrator/threshold bằng chính 100 câu này. Muốn cải thiện chuyển miền cần train/dev ngoài benchmark và tập calibration riêng, không biến benchmark đã xem thành tập chọn model.
