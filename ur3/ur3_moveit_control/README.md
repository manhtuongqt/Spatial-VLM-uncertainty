# UR3 MoveIt Control (uet_ur3)

Package này cung cấp node điều khiển và cấu hình quỹ đạo chuyển động cho cánh tay robot UR3 của lab, sử dụng MoveIt 2 trên ROS 2 Humble. Hệ thống hỗ trợ mô phỏng Gazebo và robot vật lý.

---

## Baseline V0 tái lập: UR3 + SusGrip pick-and-place

Baseline khởi chạy Gazebo Fortress, `ros2_control`, MoveIt 2 và tự động chạy nhiều episode trong cùng một world: reset cube → mở kẹp → pregrasp → hạ tuyến tính → kẹp/attach → nâng → chuyển tới khay → thả → retreat → home. Mặc định dùng đúng biến thể lab `ur3` và chạy 3 episode.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to ur3_moveit_control
source install/setup.bash
ros2 launch ur3_moveit_control baseline_fixed_pick_place.launch.py
```

Chạy headless để kiểm thử nhanh:

```bash
ros2 launch ur3_moveit_control baseline_fixed_pick_place.launch.py \
  launch_rviz:=false gazebo_gui:=false \
  episode_count:=10 \
  metrics_output:=/tmp/ur3_baseline_10ep.json \
  shutdown_on_completion:=true
```

Mỗi lần chạy ghi JSON chi tiết từng state và CSV tổng hợp từng episode. JSON chứa success rate, tổng thời gian, planning/execution time và `failed_step`/`failure_reason`. Nếu không truyền `metrics_output`, kết quả nằm tại `~/.ros/ur3_baseline_metrics/`.

Kết quả thành công kết thúc bằng log:

```text
DONE: repeatable fixed pick-and-place benchmark completed
```

Smoke/launch test tự động (một episode end-to-end, có hậu kiểm metrics):

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --packages-select ur3_moveit_control --return-code-on-test-failure
colcon test-result --verbose
```

Các tùy chọn chính:

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `ur_type` | `ur3` | Biến thể robot của lab. |
| `run_script` | `true` | Đặt `false` để chỉ khởi chạy simulation/MoveIt. |
| `launch_rviz` | `true` | Bật/tắt RViz. |
| `gazebo_gui` | `true` | Bật/tắt Gazebo GUI. |
| `episode_count` | `3` | Số episode chạy liên tiếp trong cùng world. |
| `metrics_output` | tự động | Đường dẫn file JSON; CSV được ghi cùng tên. |
| `stop_on_failure` | `false` | Tiếp tục thu thập các episode còn lại khi có lỗi. |
| `shutdown_on_completion` | `false` | Tự thu gom toàn bộ launch sau benchmark/test. |
| `world_file` | `ur3_pick_place.sdf` | World bàn, cube và khay cố định. |

Các pose TCP và kích thước Planning Scene nằm trong `config/fixed_pick_place.yaml`. Cơ chế giữ vật dùng đồng thời MoveIt AttachedCollisionObject và Gazebo DetachableJoint. Reset giữa episode gọi `/world/ur3_pick_place/set_pose`, nên không cần restart world.

---

## Perception RGB-D cổ tay

Camera mô phỏng dùng cùng interface logic dự kiến cho RealSense thật, nên perception node không cần biết nguồn dữ liệu là simulation hay hardware:

```text
/wrist_camera/color/image_raw
/wrist_camera/depth/image_raw       # 32FC1, mét
/wrist_camera/color/camera_info
/wrist_camera/points
```

Camera đã được mô hình hóa theo Intel RealSense D435i đang kết nối. Pipeline
điều khiển dùng giao diện `align_depth_to_color`: RGB, registered depth và
`CameraInfo` đều ở `camera_color_optical_frame`, 640×480 @ 30 Hz. Intrinsics
được chụp từ thiết bị là `fx=606.082`, `fy=605.797`, `cx=325.544`,
`cy=249.996`; raw depth intrinsics và factory depth→color extrinsics được lưu
trong `ur3_perception/config/d435i_640x480_30.yaml`. Cây TF:

```text
tool0 → camera_link
             ├── camera_color_frame → camera_color_optical_frame
             └── camera_depth_frame → camera_depth_optical_frame
```

`tool0 → camera_link` hiện vẫn là mount mô phỏng giữ nguyên view top-down đã
kiểm thử. Nó không được coi là hand–eye calibration của gá camera thật. Sau khi
camera được bắt cứng lên robot lab, phải đo lại transform này.

Chạy riêng D435i thật bằng đúng interface logic của simulation:

```bash
ros2 launch ur3_perception d435i_camera.launch.py publish_tf:=true
```

Khi camera đã nằm trong URDF của robot thật và `robot_state_publisher` đang
chạy, dùng `publish_tf:=false` để tránh hai nguồn cùng publish internal TF.

Chạy demo perception với Gazebo, RViz và ảnh debug:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ur3_perception perception_demo.launch.py
```

Node đồng bộ gần đúng RGB, registered depth và `CameraInfo`; segment cube đỏ, dùng median depth, back-project pixel bằng pinhole model rồi TF sang `base_link`. Các output chính:

```text
/ur3_perception/object_pose
/ur3_perception/object_pose_stable
/ur3_perception/object_observation
/ur3_perception/object_point_camera
/ur3_perception/debug_image
/ur3_perception/oracle_pose
/ur3_perception/position_error_m
/ur3_perception/status
```

Interface pixel → 3D dùng được độc lập với detector:

```bash
ros2 service call /ur3_perception/project_pixel \
  ur3_perception_interfaces/srv/ProjectPixel \
  "{u: 320, v: 240, window_radius: 3}"
```

Gazebo `PosePublisher` cung cấp oracle cho cube. Oracle chỉ dùng để kiểm thử từng tầng và báo sai số, không được dùng làm đầu vào pose ước lượng. Automated test gồm unit test projection/depth và launch test camera → sync → TF → oracle:

```bash
colcon test --packages-select ur3_perception --return-code-on-test-failure
colcon test-result --test-result-base build/ur3_perception/test_results --verbose
```

### Perception-driven pick-and-place

Runner hiện hữu có tham số `object_pose_source: fixed|perception`. Nhánh `fixed`
giữ nguyên baseline hồi quy; nhánh `perception` đưa robot tới view pose, gom 7
frame hợp lệ, kiểm tra freshness/depth/workspace/position variance và đóng băng
pose trước khi sinh quỹ đạo gắp. `ObjectObservation` chứa `PoseStamped` trong
`base_link`, kích thước quan sát, yaw tùy chọn, confidence, status, timestamp,
số frame hợp lệ và độ lệch chuẩn vị trí.

Chạy demo đầy đủ với cube được đặt ngẫu nhiên trong workspace:

```bash
ros2 launch ur3_perception perception_pick_place.launch.py \
  randomize_object:=true episode_count:=1
```

### Pick-and-place với RoboRefer RGB-D

Pipeline hiện tại dùng RoboRefer chọn point theo mô tả không gian. Registered
depth tạo vùng vật, bbox, điểm gắp và hình học 3D; target đã kiểm tra mới được
TF sang `base_link`, kiểm tra ổn định rồi đưa cho MoveIt 2:

```bash
cd /home/dhcn/ur_ws
source /opt/ros/humble/setup.bash
source install_ur3_demo_sys/setup.bash
ros2 launch ur3_perception multimodal_pick_place.launch.py
```

Dashboard gồm camera RGB, depth mask, point + bbox và ô so sánh/suy luận 3D.
Quyết định điều khiển nằm ở `/ur3_perception/target_selection`; trạng thái
RoboRefer được publish dưới namespace `/ur3_perception/roborefer`. Metrics
JSON/CSV được ghi theo timestamp vào `/home/dhcn/ur_ws/src/myproject/metrics`.
Có thể tự thu gom toàn bộ tiến trình bằng `shutdown_on_completion:=true`.

Chuỗi perception thêm vào state machine là: pose ổn định → đóng băng → thêm
cube quan sát vào Planning Scene và chờ xác nhận → sinh grasp/pregrasp/lift từ
pose cube → gắp → attach đồng bộ Gazebo/MoveIt → đặt vào bin cấu hình → detach
và chờ scene xác nhận. Oracle Gazebo chỉ cấp `/position_error_m` cho báo cáo;
runner không subscribe oracle pose/TF và không dùng nó để sinh lệnh robot.

Các tham số TCP, khoảng approach/lift, kích thước, độ mở kẹp, grasp offset,
place offset, bin pose và stability gate nằm trong
`ur3_moveit_control/config/fixed_pick_place.yaml` và
`ur3_perception/config/rgbd_perception.yaml`.

Thí nghiệm robustness độc lập theo từng trục dùng bốn profile có sẵn. Mỗi
profile có 7 mức nhiễu; x/y/z dùng mét, yaw dùng radian:

```bash
ros2 launch ur3_perception perception_pick_place.launch.py \
  launch_rviz:=false gazebo_gui:=false image_view:=false \
  episode_count:=7 stop_on_failure:=false \
  robustness_profile:=robustness_x.yaml \
  metrics_output:=/tmp/robustness_x.json \
  shutdown_on_completion:=true
```

Đổi profile lần lượt thành `robustness_y.yaml`, `robustness_z.yaml` và
`robustness_yaw.yaml`. JSON/CSV schema v2 báo perception/planning/grasp/place và
end-to-end success rate, sai số perception, ngưỡng nhiễu lớn nhất đã thành
công, thời gian, cùng failure stage/reason của mọi trial thất bại.

---

## Cài đặt và Biên dịch

Yêu cầu thực hiện biên dịch package trong không gian làm việc (workspace) trước khi vận hành:

```bash
cd ~/ros2_ws
colcon build --packages-select ur3_moveit_control
source install/setup.bash
```

---

## 1. Vận hành trong môi trường Mô phỏng (Simulation)

Chế độ này sử dụng môi trường mô phỏng Gazebo (Ignition) phối hợp với cơ chế đồng bộ thời gian mô phỏng (`use_sim_time:=true`).

### Terminal 1: Khởi chạy môi trường Gazebo
Khởi tạo mô phỏng vật lý của robot UR3 trên Gazebo:
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur3
```

### Terminal 2: Khởi chạy MoveIt Server và Node điều khiển
Tiến trình này sẽ tích hợp khởi động MoveIt Planning Server, giao diện trực quan hóa RViz 2, và thực thi tuần tự quỹ đạo di chuyển đã lập trình sẵn sau thời gian trễ 5 giây:
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur3_moveit_control ur3_demo.launch.py \
  ur_type:=ur3 \
  use_sim_time:=true
```

---

## 2. Vận hành với Robot vật lý (Real Robot)

Chế độ này thực thi các lệnh điều khiển trực tiếp tới phần cứng robot UR3 của lab qua Ethernet (`use_sim_time:=false`).

### Terminal 1: Khởi chạy trình điều khiển phần cứng (Driver)
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3 \
  robot_ip:=192.168.1.10 \
  launch_rviz:=false
```

> [!IMPORTANT]
> **Quy trình bắt buộc trên thiết bị Teach Pendant của Robot:**
> 1. Thiết lập trạng thái robot sang chế độ Remote Control (Điều khiển từ xa).
> 2. Mở chương trình điều khiển có chứa node External Control (đảm bảo cấu hình đúng địa chỉ IP của máy tính gửi lệnh).
> 3. Nhấn nút Play trên Teach Pendant để bắt đầu thực thi chương trình kết nối.
> 4. Xác nhận kết nối thành công tại Terminal Driver thông qua log:
>    `[UR_Client_Library:]: Robot connected to reverse interface. Ready to receive control commands.`

### Terminal 2: Khởi chạy MoveIt Server và Node điều khiển
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur3_moveit_control ur3_demo.launch.py \
  ur_type:=ur3 \
  use_sim_time:=false
```

---

## Các tham số cấu hình trong `ur3_demo.launch.py`

Người dùng có thể tùy biến hành vi hệ thống thông qua việc truyền các đối số (arguments) khi thực thi file launch:

| Tham số | Giá trị mặc định | Mô tả chi tiết |
| :--- | :--- | :--- |
| `ur_type` | `ur3` | Dòng robot Universal Robots; cấu hình lab là `ur3`. |
| `use_sim_time` | `true` | Xác định nguồn thời gian sử dụng (đặt `false` đối với robot vật lý). |
| `launch_rviz` | `true` | Tùy chọn hiển thị công cụ trực quan hóa RViz 2. |
| `launch_moveit` | `true` | Tùy chọn tự động gọi MoveIt Planning Server (`ur_moveit.launch.py`). |

*Ví dụ cấu hình vận hành thực tế không sử dụng giao diện đồ họa RViz 2:*
```bash
ros2 launch ur3_moveit_control ur3_demo.launch.py ur_type:=ur3 use_sim_time:=false launch_rviz:=false
```

---

## Xử lý lỗi hệ thống (Troubleshooting)

### 1. Lỗi từ chối nhận lệnh điều khiển: `Can't accept new action goals. Controller is not running.`
* **Nguyên nhân:** Bộ điều khiển `scaled_joint_trajectory_controller` trên driver phần cứng chưa được kích hoạt. Lỗi này thường do kết nối giữa máy tính điều khiển và UR Controller Box bị gián đoạn (chương trình External Control trên Teach Pendant chưa được chạy).
* **Khắc phục:** Thực hiện kiểm tra danh sách bộ điều khiển đang hoạt động bằng lệnh `ros2 control list_controllers`. Đảm bảo chương trình External Control đã chạy và hiển thị trạng thái `active`.

### 2. Tiến trình bị treo khi nhận tín hiệu kết thúc (Ctrl+C)
* **Khắc phục:** Node điều khiển `ur3_demo_node` đã được tái cấu trúc luồng xử lý. Tác vụ di chuyển và lập kế hoạch quỹ đạo được đẩy xuống chạy bất đồng bộ ở luồng phụ (background thread), trong khi luồng chính đảm nhiệm việc lắng nghe sự kiện spin của ROS 2. Khi nhận tín hiệu SIGINT (`Ctrl+C`), toàn bộ hệ thống sẽ thoát lập tức mà không gặp hiện tượng treo tiến trình chờ giải phóng tài nguyên.

---

## 3. Hiệu chuẩn Camera Eye-in-Hand (Hand-Eye Calibration)

Tính năng này tự động tính toán **ma trận chuyển vị 4×4** (Homogeneous Transformation Matrix) từ end-effector (`tool0`) tới camera (`camera_link`) bằng thuật toán Hand-Eye Calibration của OpenCV. Script hỗ trợ chạy đồng thời **5 thuật toán** (Tsai, Park, Horaud, Andreff, Daniilidis) và tự động chọn kết quả tốt nhất.

### Yêu cầu trước khi chạy

- Camera RealSense đã cắm và hoạt động
- ArUco Marker đã in và đặt cố định trên mặt bàn (ID mặc định: `26`, kích thước: `0.1m`)
- Robot UR3 đã kết nối và TF `base_link` → `tool0` đang publish
- Package `aruco_ros` đã cài đặt

### Terminal 1: Khởi chạy Camera RealSense
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```
*Đợi đến khi log báo `RealSense Node Is Up!`*

> [!CAUTION]
> **KHÔNG bật PointCloud2 color** khi calibrate. PointCloud2 tiêu tốn rất nhiều tài nguyên (~9 triệu điểm 3D/giây) và có thể gây **freeze toàn bộ hệ thống**. Chỉ cần image RGB là đủ cho ArUco detection.

### Terminal 2: Khởi chạy Robot Driver + MoveIt
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3 \
  robot_ip:=192.168.1.10 \
  launch_rviz:=false
```
*(Nhớ bật External Control trên Teach Pendant)*

### Terminal 3: Khởi chạy ArUco Marker Detection
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur3_moveit_control eye_in_hand_calib.launch.py
```

### Terminal 4: Chạy Script Calibration
```bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/ur3_uet/uet_ur3-master/ur3_moveit_control/scripts/realsense_calib_eye_in_hand.py
```

### Quy trình thao tác

| Phím | Chức năng |
| :--- | :--- |
| `Enter` | Lấy 1 mẫu calibration (TF robot + pose marker) |
| `c` + `Enter` | Tính toán calibration và in kết quả |
| `q` + `Enter` | Thoát chương trình |

> [!IMPORTANT]
> **Quy trình lấy mẫu chuẩn xác:**
> 1. Cần **ít nhất 5 mẫu** ở các tư thế (pose) tay máy **khác nhau rõ rệt** — thay đổi cả vị trí lẫn góc xoay.
> 2. Tại mỗi tư thế, đảm bảo camera **nhìn thấy rõ toàn bộ ArUco Marker** trước khi nhấn Enter.
> 3. Tránh các tư thế quá gần nhau hoặc chỉ thay đổi 1 trục — điều này gây ra kết quả không chính xác.
> 4. Sau khi lấy đủ mẫu, gõ `c` để tính toán. Script sẽ in ra:
>    - **Bảng so sánh** kết quả 5 thuật toán (X, Y, Z, Roll, Pitch, Yaw)
>    - **Ma trận chuyển vị 4×4** của thuật toán tốt nhất
>    - **Lệnh `static_transform_publisher`** sẵn sàng copy-paste (cả Quaternion và Euler)

### Output mẫu

```
═══════════════════════════════════════════════════
  🏆 KẾT QUẢ TỐT NHẤT: PARK
═══════════════════════════════════════════════════
--- MA TRẬN CHUYỂN VỊ 4×4 (tool0 → camera_link) ---
  [ +0.999123  -0.012345  +0.034567  +0.045678 ]
  [ +0.011234  +0.998765  +0.023456  -0.023456 ]
  [ -0.035678  -0.022345  +0.999234  +0.067890 ]
  [ +0.000000  +0.000000  +0.000000  +1.000000 ]
```

### Lưu và tái sử dụng kết quả

Hệ thống sẽ tự động lưu lại **2 file** cho mỗi lần calib tại thư mục `~/ros2_ws/calib_results/`:
1. **File `.txt` (Báo cáo tổng hợp):** Chứa kết quả ma trận, nhận xét, và lệnh publish TF có thể đọc và copy trực tiếp (phù hợp để làm báo cáo).
2. **File `.npz` (Dữ liệu gốc):** Chứa toàn bộ ma trận và dữ liệu mẫu. Để load lại bằng code:
```python
import numpy as np
data = np.load('~/ros2_ws/calib_results/hand_eye_calib_20260625_170000.npz', allow_pickle=True)
T_best = data['HORAUD_T']  # Ma trận 4x4 của thuật toán HORAUD
print(T_best)
```

### Áp dụng kết quả vào hệ thống

Sau khi calibration xong, copy lệnh `static_transform_publisher` từ output và thêm vào launch file hoặc chạy trực tiếp:
```bash
# Ví dụ (thay số thực tế từ output):
ros2 run tf2_ros static_transform_publisher \
  --x 0.045678 --y -0.023456 --z 0.067890 \
  --qx 0.012345 --qy -0.017890 --qz 0.005678 --qw 0.999750 \
  --frame-id tool0 --child-frame-id camera_link
```
