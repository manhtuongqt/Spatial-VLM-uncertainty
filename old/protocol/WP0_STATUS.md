# WP0 — Khóa điểm xuất phát

## Kết luận

**PASS — Giai đoạn 0 đạt toàn bộ gate đã yêu cầu trong phạm vi build/test/API smoke.**

Đề tài đang thực hiện là: **“Nghiên cứu phương pháp LLM/VLM Agent lập kế hoạch vòng kín với hiệu chuẩn độ bất định không gian và can thiệp thích ứng cho thao tác gắp đặt bằng tay máy UR3.”**

Thời điểm kiểm tra: 18/08/2026 (Asia/Ho_Chi_Minh). Pilot khóa tại `results/roborefer_pilot_v0_20260813_173305` chỉ được đọc để kiểm tra và demo hậu nghiệm; không file nào trong artifact này bị sửa.

## Trạng thái gate

| Gate | Kết quả | Bằng chứng |
|---|---:|---|
| Build sạch | PASS | Clean-configure dependency closure tới `ur3_perception`: 9 package hoàn tất trong 5,79 s. |
| Đồng bộ `ur3_perception_interfaces` | PASS | Build/install lại và import được `ObjectObservation` từ install space. |
| Unit test bằng một lệnh tái lập | PASS | `./protocol/run_wp0_checks.sh`: 72/72 test pass. |
| 54 test perception/pilot cũ | PASS | 54/54 pass riêng trong 0,92 s. |
| MoveIt pose generation và test mở rộng | PASS | 18/18 pass: projection, reason code, pose generation và D435i description. |
| Sáu structured reason code | PASS | Demo deterministic quan sát đúng 6/6 mã mong đợi. |
| RoboRefer API RGB | PASS | HTTP 200, `[(0.787, 0.477)]`, 1.419 s. |
| RoboRefer API RGB-D | PASS | HTTP 200, `[(0.787, 0.481)]`, 1.814 s. |
| Pilot cũ không đổi | PASS | Digest trước/sau cùng là `df8ec334c82b14447463101cccb5e54836fb27e579a760322256f84d157a2b9c`. |
| Inventory/hash source và config | PASS | 97/97 file modified/untracked đã hash; 13/13 source/config trọng yếu đã hash. |

Build có cảnh báo không gây lỗi từ `ur_controllers` và `ur_moveit_config`; cảnh báo chính là biến CMake `PYTHON_EXECUTABLE` không được một package sử dụng. CMake vẫn chọn đúng `/usr/bin/python3` 3.10 cho các package cần Python. Năm cảnh báo test là `PendingDeprecationWarning` từ ROS `image_geometry` dùng `numpy.matrix`.

## Phần đã triển khai

### Structured depth-component Gate v2

Gate v2 nằm tách khỏi hàm legacy để bảo toàn khả năng replay pilot đã khóa. Node grounder chạy hiện tại dùng Gate v2 và phát `reason_code` cùng `geometry_diagnostics` dạng máy đọc được.

| Reason code | Điều kiện phát |
|---|---|
| `AREA_TOO_SMALL` | Component có seed nhỏ hơn `min_area_px`. |
| `AREA_TOO_LARGE` | Component có seed vượt diện tích tối đa nhưng vẫn nằm gọn trong ROI. |
| `NO_SEED_SUPPORT` | Seed ngoài ảnh, quanh seed không có metric depth hợp lệ, hoặc không component nào hỗ trợ seed. |
| `PLANE_MERGE` | Component quá lớn và chạm biên ROI; đây là dấu hiệu quan sát được của vật dính vào mặt phẳng/vùng đồng độ sâu. |
| `BORDER_TRUNCATED` | Component chạm biên ảnh khi chế độ reject fail-closed được bật. |
| `UNSTABLE_3D` | Cửa sổ temporal đã đủ mẫu nhưng độ lệch chuẩn vị trí 3D vượt ngưỡng. |

`BORDER_TRUNCATED` mặc định tắt để chưa làm thay đổi policy vận hành trước khi có calibration. `PLANE_MERGE` là heuristic hình học, không được diễn giải như nhãn semantic ground truth.

Chạy Gate v2 hậu nghiệm trên input đã khóa của `pilot_scene_0007` cho `PLANE_MERGE`, thay cho thông báo legacy mơ hồ. Kết quả này được ghi `POST_HOC_DIAGNOSTIC_ONLY`: không sửa prediction, summary hoặc điểm pilot.

### Môi trường test tái lập

Lệnh chuẩn:

```bash
cd /home/dhcn/ur_ws/src/myproject
./protocol/run_wp0_checks.sh
```

Script cố định `/usr/bin/python3`, đặt `PYTHONNOUSERSITE=1` và `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, nên không còn phụ thuộc plugin `pytest` lỗi trong user site. Script tự clean-configure/build, source install, kiểm tra interface, chạy test, chạy demo reason code và đối chiếu digest pilot trước/sau.

## Demo và artifact

- Demo trực quan: [`WP0_DEMO.html`](WP0_DEMO.html)
- Chi tiết từng reason code: [`reason_code_demo.json`](reason_code_demo.json)
- Inventory đầy đủ: [`source_inventory.json`](source_inventory.json)
- Báo cáo test máy đọc được: [`test_report.json`](test_report.json)
- Kết quả API thật: [`api_smoke_result.json`](api_smoke_result.json)
- Log build/test: [`logs/`](logs/)

API smoke dùng checkpoint thật `RoboRefer-2B-SFT`, greedy decoding, seed `8132026` và checkpoint inventory SHA-256 `5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa`. Nhánh RGB-D nhận registered depth view có sẵn từ D435i; Depth Anything không được load trùng.

## Giới hạn và việc chưa được khẳng định

- Demo sáu reason code xác minh hợp đồng phần mềm và đường phát lỗi, chưa hiệu chuẩn threshold trên tập lỗi RGB-D thực lớn.
- API smoke xác minh model thật, giao thức, greedy policy và output contract; không được tính như một thí nghiệm accuracy mới.
- WP0 chưa chạy pick-and-place vật lý hoặc mô phỏng vòng kín. Đây là điểm xuất phát đã khóa cho các work package tiếp theo.

## Quyết định bước kế tiếp

Mở WP1 theo thứ tự an toàn: thu failure set RGB-D có nhãn reason code, hiệu chuẩn các threshold Gate v2, rồi nối reason code vào policy `EXECUTE/REOBSERVE/REPROMPT/ASK_USER/ABSTAIN`. Chỉ sau khi pass replay và shadow evaluation mới cho phép reason-code policy tác động tới MoveIt execution.
