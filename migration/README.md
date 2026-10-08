# Danh sách sao lưu và chuyển máy đồ án P-CRA-U

Ngày kiểm kê: **08/10/2026**, máy lab hiện tại. Project gốc:
`/home/dhcn/ur_ws/src/myproject`.

**Cập nhật phạm vi theo yêu cầu mới:** người dùng chọn repo GitHub riêng,
private và chỉ giữ dataset 4.000 mẫu (train/dev 2.000, calibration 1.000,
IID 1.000), cùng model/artifacts cần chạy. Xem `GITHUB_DRIVE_GUIDE.md`,
`DATA_REQUIRED_PATHS.txt` và `LAPTOP_REPRODUCTION.md` cho phương án hiện hành.
Danh mục toàn máy bên dưới và `BACKUP_ROOTS.txt` là kiểm kê/phương án bảo toàn
rộng hơn trước khi thu hẹp, không phải danh sách upload Drive đang được chọn.

**Khuyến nghị để tránh mất dữ liệu: chép nguyên `/home/dhcn/ur_ws/`, sau đó chép
các mục ngoài workspace trong phần 2.** Cách này giữ cả project hiện tại,
source robot bên cạnh và các overlay ROS đã có. Không chỉ clone Git hoặc chép
`new/src`: dữ liệu, weights, môi trường và nhiều kết quả không nằm trong Git.

Đây là **kiểm kê và hướng dẫn**, chưa phải bản sao lưu ở một thiết bị khác.
Không có dữ liệu gốc nào được di chuyển/xóa, không train, fit, inference,
capture hoặc chạy robot. Chỉ thêm thư mục kiểm kê này vào `plan/`.

## 1. Project chính: phải giữ nguyên cả folder

Dung lượng đo bằng `du` khoảng **53,8 GiB**; toàn `/home/dhcn/ur_ws` khoảng
**66,7 GiB**. Số đo gồm file đang có trên đĩa, không dereference symlink;
hardlink được đếm một lần trong từng lần đo. Dung lượng các hàng con không
nhất thiết cộng đúng bằng folder cha vì có hardlink giữa các cây.

Trong bảng dưới, đường dẫn tương đối tính từ `myproject/`.

| Folder/file | Khoảng dung lượng | Nội dung và lý do giữ |
|---|---:|---|
| `new/` | 11,1 GiB | Triển khai hiện tại, scripts, configs, tests, tài liệu, bundles, kết quả IID và benchmark, hình khoa học, demo Gazebo. **Giữ toàn bộ.** |
| `old/` | 14,5 GiB | Archive lịch sử **đồng thời là đầu vào đang được code hiện tại sử dụng**: train/dev, calibration, feature cache, manifests. Không bỏ vì tên `old`. |
| `datasets/` | 5,3 GiB | RefSpatial gốc, Gazebo data, bản clean/human evaluation và supplement. Giữ RGB, depth, masks, labels, manifests cùng nhau. |
| `RoboRefer/` | 6,0 GiB | Source backbone có sửa local, API/extractor, nested Git và **model weights thực nằm tại `models/`**. |
| `sam2/` | 0,26 GiB | Source segmentation, cấu hình, nested Git và checkpoint SAM2.1. |
| `ur3/` | 0,06 GiB | Source ROS/Gazebo/MoveIt, URDF/Xacro, meshes robot/gripper/YCB, world và camera mount. Chi tiết ở phần 3. |
| `.conda-roborefer/` | 6,3 GiB | Môi trường Python local. Chép để bảo toàn; không mặc định chạy được khi đổi đường dẫn/OS. |
| `.git/` và `.gitignore` | 1,1 GiB cho `.git` | Lịch sử root repo, trạng thái các file đã commit. Worktree hiện có rất nhiều file untracked và các xóa từ trước; **Git không thay thế bản copy toàn cây**. |
| `plan/` | nhỏ so với dữ liệu | Handoff, protocol, S0–S6, kế hoạch GCA/FUSE, kết quả public benchmark, pipeline PNG và bản kiểm kê này. |
| `protocol/` | 9 MiB | Audit/capture/feature/evaluation tooling của các giai đoạn trước; giữ để đọc lại provenance và tái lập. |
| `results/` | 4,7 GiB | Dữ liệu capture, kết quả và báo cáo giai đoạn trước; không chỉ giữ `SUMMARY.md`. |
| `ketqua1/` | 3,6 GiB | Bộ artifact/report đã tập hợp theo các khối phương pháp, có bản dữ liệu/feature và bằng chứng lịch sử. |
| `ketquangay/` | 0,96 GiB | Báo cáo và artifact theo ngày. |
| `ketqua/` | 2 MiB | Bộ tổng hợp kết quả lịch sử. |
| `answerability and uncertainty/` | 90 MiB | `ocid_ref_occlusion_200/`: dữ liệu/QA/ảnh/nhãn occlusion đang mở trong IDE. **Giữ cả folder, không chỉ `qa.jsonl`.** |
| `workspace/` | 28 MiB | `mh_pcrau_v3/`, generated worlds/layout/capture plan và helper của các giai đoạn triển khai. |
| `latex/` | 28 MiB | Nguồn khóa luận, bibliography, hình/template và bản build. |
| `baocao/`, `baocao1/` | khoảng 22 MiB | Các báo cáo LaTeX/PDF và ảnh tương đối. `baocao1` gồm báo cáo tiến độ người dùng sửa và báo cáo benchmark mới. |
| `.tools/` | 57 MiB | Công cụ local hỗ trợ tài liệu/biên dịch. |
| `output/`, `tmp/` | khoảng 14 MiB | Output/ảnh/traces nhỏ còn trong workspace; giữ khi copy cả cây. Không nhầm `tmp/` này với `/tmp` của hệ điều hành. |
| `build/`, `install/`, `log/` | khoảng 0,28 GiB | Sản phẩm colcon và logs bên trong project. Giữ để đối chiếu, nhưng sẽ phải rebuild khi đổi máy/đường dẫn. |
| `README.md`, `DEPENDENCIES.md`, các file ẩn còn lại | nhỏ | Thông tin dependency/setup. README root còn mô tả giai đoạn cũ; không dùng nó để quyết định bỏ `new/` hoặc `old/`. |

### Các phần bên trong `new/` cần nhận diện khi kiểm tra bản sao

- `new/src/pcrau/`, `new/scripts/`, `new/configs/`, `new/tests/`, `new/docs/`:
  code, runner và contracts hiện tại.
- `new/outputs/`: **giữ toàn bộ**, bao gồm checkpoint, config, freeze locks,
  calibrator, profile, raw predictions, feature inputs/cache, metrics, manifests,
  source benchmark parquet/ảnh và báo cáo; không chỉ copy checkpoint tốt nhất.
- `new/test_iid/`: `dataset/`, `feature_cache/`, `protocol/`, `contracts/`,
  `scripts/`, `evaluation/`; toàn bộ khoảng 2,9 GiB. Đây là IID cũ cần giữ
  nguyên, không thay bằng dataset khác khi chuyển máy.
- `new/demo_gazebo/`: code, README và `runs/` có RGB-D capture, world đã sinh,
  traces, audit, dashboard/video và logs.
- `new/hinhanh/` và `new/ketqua/`: hình, tài liệu nguồn hình và bộ kết quả
  đóng gói. Dù có bản trùng artifact, chưa loại bỏ bản nào trong lần chuyển này.

### Những bundle/kết quả quan trọng hiện có

Các đường sau đều phải nằm trong bản sao của `new/outputs/`:

| Đường dẫn | Vai trò |
|---|---|
| `active_experimental_profile.json` | Registry baseline Adapter đã chọn; chứa cả đường dẫn tuyệt đối và hashes. |
| `pcrau_target_v2_full_seed_24082026/` | V2 nền đóng băng và artifact lịch sử. |
| `pcrau_answerability_language_dev_20261003/` | Adapter `detail_cost`, config/metadata, logistic33, profiles và IID cũ. |
| `pcrau_s1_anchor_peak_pilot_v2_20261005/` | C1/P1/P2; checkpoint anchor residual và đối chứng. |
| `pcrau_unified_spatial_20261005_r2/` | Bundle live44: `bundle.json`, `frozen/anchor_mlp.safetensors`, freeze lock, calibrator/profile. |
| `pcrau_unified_spatial_iid_20261007/` | Kết quả IID của chính đường live44. |
| `pcrau_source_mc_train_dev_20261007/` | MC của source head trên train/dev. |
| `pcrau_task_uncertainty_20261007/` | Task heads, cache, supplement/ảnh nhãn occlusion, training lock và raw outputs. Bản release/calibration cuối nằm tại **`evaluation_r3/`**, báo cáo tại `evaluation_r3/report_r2/`. Không chỉ giữ r1/r2. |
| `pcrau_roborefer_primary_pilot300_20261007/` | Biến thể giữ điểm RoboRefer, replay RefCOCO và kiểm tra case live. |
| `refspatial_location_frozen_20261007/` | Source Location, inputs, predictions/metrics và provenance. |
| `refcoco_val_frozen_20261007/` | Source parquet RefCOCO; không bỏ vì thấy tên giống pilot. |
| `refcoco_val_pilot300_20261007/` | 100 ảnh/300 câu RefCOCO, raw generation, Sidecar, features và evaluator. |
| `refcoco_plus_val_pilot300_20261007/` | 100 ảnh/300 câu RefCOCO+, source parquet/ảnh, predictions, error audit, hình case. |

Các run thử nghiệm khác cũng được giữ trong bản copy toàn folder: chưa dọn,
promote, reset hoặc thay tên bundle để chuyển máy.

### `old/` là dependency trực tiếp, đã đối chiếu với config và dataset loader

`new/configs/pcrau_target_v2.json` và `new/src/pcrau/dataset.py` đọc:

```text
old/roborefer_dataset_v2_1_1_development_400_20260824/
old/datasets/roborefer_dataset_v2_1_calibration_200_20260824/
old/results/pcra_u_feature_cache/pcra_u_development_v1/
old/results/pcra_u_feature_cache/pcra_u_calibration_v1/
old/protocol/pcra_u_development_train_manifest.json
old/protocol/pcra_u_calibration_eval_manifest.json
old/results/pcra_u_runs/pcra_u_development_failure_audit_20260824/
```

Các feature indexes/manifests có path tới RGB/depth/mask/feature files.
Copy manifest mà thiếu các file payload tương ứng sẽ không tái lập được.

## 2. Ngoài `myproject`: phải kiểm tra và sao lưu thêm

| Đường dẫn tuyệt đối | Dung lượng | Mức ưu tiên và lý do |
|---|---:|---|
| `/home/dhcn/ur_ws/src/uet_ur3/` | 0,25 GiB | **Giữ.** `ur_ws/install` hiện có symlink thật tới Driver/MoveIt/SusGrip ở clone này, bên cạnh package ở `myproject/ur3`. Không mặc định hai clone giống nhau. Đã bao gồm nếu chép cả `ur_ws`. |
| `/home/dhcn/ur_ws/build*`, `install*`, `log*` | nằm trong 66,7 GiB | Giữ khi chép cả `ur_ws`; có nhiều overlay robot/Gazebo. Các bản compiled install không thay thế source. |
| `/home/dhcn/.local/lib/python3.10/` | **6,75 GiB** | **Giữ.** Sidecar hiện dùng packages user site, đặc biệt Torch 2.13/OpenCV 5, khác Conda sạch. |
| `/home/dhcn/workspace/ros_ur_driver/` | 2,78 GiB | Giữ để bảo toàn source driver/description/control/librealsense bên ngoài. `src/` khoảng 0,88 GiB; build/install có thể dựng lại. Không khẳng định mọi package ở đây thuộc live pipeline. |
| `/home/dhcn/Downloads/` | 15,90 GiB toàn folder | Giữ toàn folder nếu ưu tiên chống mất dữ liệu; có PPTX gốc, kế hoạch ngoài repo, paper, zip data và ảnh. `gazefollow_extended`/installer cũng chiếm dung lượng nhưng chưa xác nhận là dependency của pipeline P-CRA-U. |
| `/home/dhcn/.ignition/` | 0,11 GiB | **Giữ.** Fuel model cache, Gazebo Fortress config và logs; tránh mất model/texture chỉ tải về ở máy lab. |
| `/home/dhcn/.gazebo/`, `.rviz2/`, `.sdformat/` | rất nhỏ | Giữ cấu hình simulator/RViz nếu muốn khôi phục giao diện và lịch sử setup. |
| `/opt/ros/humble/share/ur_description/` | 72 MiB | **Giữ bản snapshot assets/config.** World đã xuất dùng URI tuyệt đối tới meshes UR3 ở đây. Package chuẩn có thể cài lại đúng phiên bản. |
| `/opt/ros/humble/share/realsense2_description/` | 92 MiB | **Giữ bản snapshot.** World dùng `meshes/d435.dae`, URDF camera nằm ngoài project. |
| `/home/dhcn/.codex/attachments/` | 276 KiB | Giữ các text/tài liệu người dùng đã đính kèm. Đây không phải toàn bộ lịch sử chat; `plan/` là handoff chính. Không thu thập auth/token tài khoản. |
| `/home/dhcn/.99-realsense-libusb.rules`, `.realsense-config.json` | nhỏ | Giữ file rules/preferences camera hiện có; không tự cài rules lên máy mới khi chưa xem lại. |
| `/home/dhcn/.cache/tectonic/`, `.cache/Tectonic/` | tổng 126 MiB | Có ích khi cần biên dịch báo cáo offline; được thêm vào danh sách copy nếu tồn tại. |
| `/home/dhcn/.local/bin/` | nhỏ | Giữ các entry scripts môi trường user; khi dựng lại Python cần cài lại entry points đúng prefix. |

Các file đặc biệt trong `Downloads/` cần nhận diện:

```text
báo cáo kết quả đồ án.pptx
ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md
markdown.md
Uncertainty-Aware RoboRefer for Reliable Spatial Referring and Robotic Manipulation.md
formkhoaluan.md
pipeline hien tai.png
P-CRA-U.png
paper/
arXiv-2411.16537v5.tar.gz
arXiv-2506.04308v4.tar.gz
arXiv-2606.06891v1.tar.gz
RefSpatial-*.zip  (5 archive nguồn hiện có)
plan_spatial_vlm_refspatial_gazebo_ur3_exact.md
research_plan_data_centric_ur3_full.md
phan_tich_ke_hoach_16_tuan_ur3_cho_codex.md
roborefer_five_step_dashboard_panel_20260812_113605.png
stomp_planning.yaml
externalcontrol-1.0.5.urcap
```

Bản kế hoạch Downloads và bản `plan/` được giữ riêng, không đồng bộ/ghi đè
trong lần kiểm kê này.

### Mục bổ sung tùy nhu cầu, chưa xác nhận là đầu vào của pipeline hiện tại

- `/home/dhcn/.ros/` khoảng 1,34 GiB, chủ yếu logs: nên giữ nếu cần lịch sử
  debug robot hoặc phân tích lỗi ROS cũ.
- `/home/dhcn/miniforge3/` khoảng 24,67 GiB: giữ nếu muốn bảo toàn cả Conda
  manager, cache và các env khác; riêng env P-CRA-U đã ở `.conda-roborefer`.
  Các env `GestureTarget`/`intentmotion` chưa xác nhận phục vụ đồ án hiện tại.
- `/home/dhcn/shawn_ws/` khoảng 5,71 GiB: TinyVLA/dataset/robot workspace khác;
  chưa tìm thấy reference trực tiếp từ code/config P-CRA-U hiện tại.
- `ur_ws/src/lerobot`, `IROS2025-IntentMotion`, `TransGesture`,
  `ur_action_intent`, `/home/dhcn/workspace/lerobot`: dự án bên cạnh. Copy cả
  `ur_ws` sẽ giữ các folder ở trong đó; không đưa số liệu của chúng vào P-CRA-U.
- `/home/dhcn/Pictures/Screenshots`, `/home/dhcn/Videos/Screencasts`: nên xem
  lại nếu có ảnh/video demo chưa đưa vào repo.
- `~/.cache/pip` khoảng 9 GiB là cache cài đặt, không phải model dataset hiện
  hành. Không bắt buộc giữ để bảo toàn kết quả. Không thấy `~/.cache/huggingface`
  tại thời điểm kiểm kê; weights backbone đang nằm trong `RoboRefer/models`.

## 3. Robot/Gazebo/URDF: chép các nhóm này cùng nhau

Trong `/home/dhcn/ur_ws/src/myproject/ur3/`:

| Nhóm | Thành phần cần giữ |
|---|---|
| UR3 + camera mount | `ur3_moveit_control/urdf/ur3_with_susgrip*.xacro`, `d435i_wrist_camera.xacro` |
| UR simulation | `ur_simulation_gz/urdf/`, `launch/`, `config/`, `worlds/` |
| Các world đang có | `ur3_pick_place_uq_occlusion_v2.sdf`, `ur3_pick_place_uq_occlusion.sdf`, `ur3_pick_place.sdf`, inventory/gallery worlds |
| Vật YCB | **Cả** `ur_simulation_gz/models/`, bao gồm `.obj`, `.mtl`, textures và metadata; không chỉ model mesh |
| SusGrip | **Cả** `susgrip_2f/`: description, meshes STL/OBJ, control, hardware, gazebo và configs |
| MoveIt/planner | `ur3_moveit_control/config/`, `launch/`, `rviz/`, scripts/controllers |
| Perception + dữ liệu | `ur3_perception/`, `ur3_perception_interfaces/`, `ur3_spatial_dataset/` |
| Driver UR | `Universal_Robots_ROS2_Driver/` và bản bên `ur_ws/src/uet_ur3/` |
| Camera/UR meshes ngoài source | Hai package description tại `/opt/ros/humble/share/` nêu ở phần 2 |
| World/poses đã sinh | `workspace/mh_pcrau_v3/`, `new/demo_gazebo/runs/`, capture/results/protocol archive |

Đã kiểm tra các symlink và URI trong source/world:

- Trong **`myproject` có 1.538 symlink, không có symlink đứt** tại thời điểm scan.
  Bốn symlink ra ngoài project là environment hooks của ROS trong `install/`,
  đích tại `/opt/ros/humble/share/ament_cmake_core/`.
- Tuy nhiên các `.sdf` đã sinh còn có **URI tuyệt đối**, không phải symlink,
  trỏ tới UR3/camera meshes ở `/opt/ros/humble/share/...`; copy symlink đủ
  không có nghĩa đã copy đủ mesh dependencies.
- Trong phạm vi rộng hơn có **782 symlink đã đứt từ trước**: 769 tại
  `ur_ws/install_kitchen`, 11 tại `ur_ws/install_ur3_demo_sys`, một README ở
  dự án `ur_action_intent`, một link `/ursim/...` trong test fixture UR Client
  Library. Danh sách đầy đủ trong `SYMLINK_INVENTORY.json`.
- Việc backup giữ lại hiện trạng này; **không khôi phục được các file đã mất
  từ trước chỉ bằng chép symlink**. Nếu sau này cần demo kitchen cũ, phải tìm
  source trong Git/archive hoặc tái tạo có kiểm chứng. Không sửa/khôi phục
  source cũ trong nhiệm vụ này.

## 4. Môi trường chạy: đã xuất thông tin để dựng lại

Snapshot máy hiện tại: Ubuntu **22.04.5**, ROS2 **Humble**, Gazebo Fortress
**6.18.0**, GPU RTX 2000 Ada **16.380 MiB**, NVIDIA driver **580.173.02**.

Hai đường Python hiện khác nhau:

| Đường chạy | Package quan trọng được kiểm kê |
|---|---|
| `.conda-roborefer/bin/python3.10` với `PYTHONNOUSERSITE=1` | Torch 2.5.1, torchvision 0.20.1, NumPy 1.26.4, OpenCV-headless 4.11, Transformers 4.49.0, safetensors 0.5.2 |
| Python3.10 có user site `~/.local/lib/python3.10/site-packages` | Có Torch 2.13.0 và OpenCV 5.0.0.93; runtime bundle Sidecar đã khóa Torch `2.13.0+cu130` |

Backbone worker chạy Conda sạch, Sidecar chạy runtime riêng có user site.
**Chép riêng `.conda-roborefer` sẽ không giữ toàn bộ môi trường Sidecar.**
`pip_user_python310.txt` chỉ ghi version metadata; exact build CUDA cần xem
thêm freeze lock/runtime manifests và snapshot package thực đã copy.

Trong folder này đã có:

- `conda_environment.yml`, `conda_explicit.txt`: export env hiện tại.
- `pip_conda_clean.txt`: pip freeze từ Conda với user site tắt.
- `pip_user_python310.txt`: inventory distributions trong user site Python3.10.
- `dpkg_installed.tsv`: packages/version/status hệ thống, để tra ROS, MoveIt,
  Ignition/Gazebo, RealSense và libraries khi cài lại. Không tự cài mù tất cả
  packages trong file này lên OS khác.
- `gpu.txt`, `gazebo_versions.txt`, `os_release.txt`, `storage.txt`.
- `shell_dependency_lines.txt`: các dòng setup ROS/Conda liên quan trong shell.
- `EXPORT_STATUS.json`: tất cả tám lệnh export đã hoàn tất với exit code 0.

Không cần chép toàn `/usr` để bảo toàn đồ án. ROS/Gazebo/CUDA hệ thống có thể
cài lại theo OS/phần cứng máy mới. Các model, source sửa local, dữ liệu và
results riêng của đồ án mới là phần không thể thay bằng cài package.

## 5. Cách copy khi có ổ đích

Danh sách **`BACKUP_ROOTS.txt`** dùng phương án giữ toàn `ur_ws`, whole Downloads,
workspace driver bên ngoài, thư viện user Python, các model/config bên ngoài.
Không bao gồm các mục tùy chọn ở trên như toàn Miniforge/shawn_ws/ROS logs.

Dung lượng nhóm chính theo snapshot khoảng **92–93 GiB** trước phần dự phòng
và sự thay đổi dữ liệu. Nên chuẩn bị **ít nhất 120 GiB trống** cho phương án
này; cần thêm dung lượng nếu giữ các workspace/env tùy chọn hoặc có bản sao
thứ hai. Máy lab còn khoảng 30 GiB trống trên root filesystem, không đủ để
tạo thêm một bản copy toàn dữ liệu trên chính phân vùng này.

Đã kiểm tra chính danh sách 16 roots bằng **rsync dry run**, exit code 0:
548.754 entries, tổng logical size 98.420.521.305 bytes (khoảng 91,66 GiB).
Kết quả lưu tại `RSYNC_DRY_RUN.txt`. Đây là kiểm tra danh sách copy đọc được,
**chưa truyền dữ liệu hoặc xác nhận một bản backup thật**.

### Đích Linux filesystem, hỗ trợ symlink/hardlink

Thay `/media/dhcn/TEN_O_DICH` bằng mountpoint thật đã kiểm tra. Lệnh dưới
**chưa được chạy** trong nhiệm vụ kiểm kê:

```bash
TASK_BACKUP_DIR='/media/dhcn/TEN_O_DICH/PCrau_backup_20261008'
TASK_BACKUP_LIST='/home/dhcn/ur_ws/src/myproject/plan/MIGRATION_BACKUP_20261008/BACKUP_ROOTS.txt'
mkdir -p "$TASK_BACKUP_DIR"
rsync -aHr --partial --info=progress2 --relative \
  --files-from="$TASK_BACKUP_LIST" / "$TASK_BACKUP_DIR/"
```

`-r` được ghi tường minh vì `--files-from` thay cách xử lý recursion của
`-a`; `-H` giữ hardlink. Layout bản sao sẽ là
`PCrau_backup_20261008/home/dhcn/ur_ws/...`, `.../opt/ros/...`.
Danh sách có folder `ur_ws`, nên bao gồm `.git`, `.conda-roborefer` và folder
kiểm kê này. Không dùng glob `myproject/*` vì sẽ bỏ file/folder ẩn.

Lệnh copy không dùng `--delete` và không dereference toàn bộ symlink. Sau
copy, kiểm tra nội dung bằng dry run checksum; có thể mất thời gian vì phải
đọc cả dữ liệu gốc và bản sao:

```bash
rsync -aHrcn --itemize-changes --relative \
  --files-from="$TASK_BACKUP_LIST" / "$TASK_BACKUP_DIR/"
```

Exit code 0, không còn file cần update và không có lỗi đọc/copy là điều kiện
kiểm tra thực tế; inventory tạo trước copy **không thay thế** bước này.
Dừng các tiến trình đang ghi dữ liệu của chính đồ án trước bản copy cuối để
tránh một bộ file lấy từ nhiều thời điểm. Không tự tắt robot/job của người khác.

### Đích exFAT/NTFS không giữ Linux links/mode đúng

Nên dùng archive tar để giữ cấu trúc Linux bên trong file, thay vì kéo thả
folder và để symlink thành shortcut hoặc mất link. Ví dụ, có đủ dung lượng
ở ổ đích rồi mới chạy:

```bash
tar -cpf "$TASK_BACKUP_DIR/pcrau_lab_backup.tar" -C / \
  --verbatim-files-from -T "$TASK_BACKUP_LIST"
sha256sum "$TASK_BACKUP_DIR/pcrau_lab_backup.tar" \
  > "$TASK_BACKUP_DIR/pcrau_lab_backup.tar.sha256"
tar -tf "$TASK_BACKUP_DIR/pcrau_lab_backup.tar" > /dev/null
```

Tar lưu hardlink/symlink mặc định, không thêm `--dereference`. GNU tar sẽ bỏ
dấu `/` đầu đường dẫn trong archive; khi restore phải xem layout trước.
`tar -tf` kiểm tra archive đọc được; checksum cần đối chiếu lại sau khi chuyển
file. Đây vẫn là bản archive của hiện trạng, không tự sửa broken links.

## 6. Khôi phục ở máy mới

1. Giữ nguyên bản backup gốc ở một nơi khác; khôi phục vào cây làm việc mới.
2. Nếu có thể, dùng cùng đường dẫn `/home/dhcn/ur_ws/src/myproject` để giảm
   lệch absolute paths. Nhiều active profiles/freeze locks/generated worlds
   chứa đường dẫn này; một số còn hash chính file source/manifests.
3. Nếu username/path mới khác, **không search–replace cả archive/bundle khóa**.
   Trước hết dùng resolver/runtime config riêng hoặc tạo bundle relocation
   có mapping/provenance rõ ràng, giữ bản khóa gốc nguyên byte.
4. Cài ROS Humble/Fortress/MoveIt và các dependencies phù hợp từ version
   inventory. Khôi phục/cài đúng các package `ur_description`,
   `realsense2_description`; kiểm tra meshes/texture URIs mở được.
5. Dựng lại Python từ exports nếu env copy không chạy. File YAML có `prefix`
   cũ và editable paths: kiểm tra chúng khi tạo env mới. Xác nhận cả hai
   đường Conda sạch và Sidecar trước khi inference; không chỉ kiểm tra một
   `import torch` mặc định.
6. Rebuild ROS vào `build/install/log` mới. Chọn một cây source robot rõ ràng:
   **không `colcon build` quét cả `ur_ws/src` một cách mặc định**, vì có thể
   gặp package trùng giữa `uet_ur3` và `myproject/ur3`. Các overlays cũ dùng
   cây source khác nhau; dùng `--base-paths`/package selection có chủ ý.
7. Kiểm tra checksum model/checkpoint/calibrator, dataset/index paths và
   mở vài RGB/depth/mask/case figures. Không train/capture lại để bù một file
   missing mà chưa kiểm tra backup gốc.
8. Sau khi runtime đúng, mới chạy các kiểm tra kỹ thuật/inference cần thiết.
   Không coi chuyển máy là lý do fit lại bằng IID hoặc đổi threshold.

`CRITICAL_FILES.sha256` có paths gốc tuyệt đối. Nếu project được restore ở
path khác, có thể kiểm tra các hash này theo đường tương đối từ root project:

```bash
cd /DUONG_DAN_PROJECT_DA_RESTORE
sed 's#  /home/dhcn/ur_ws/src/myproject/#  #' \
  plan/MIGRATION_BACKUP_20261008/CRITICAL_FILES.sha256 | sha256sum -c -
```

Tất cả 37 file trong danh sách hash này nằm trong project, gồm toàn payload
model RoboRefer/Depth Anything/SAM2 và các file bundle chính. Các file khác
vẫn phải được copy và đối chiếu bằng checksum/dry run ở phần 5.

## 7. Phạm vi đã rà soát và bằng chứng kiểm kê

Đã đọc handoff 05/10 đầy đủ, README/dependency maps, các docs/bundle hiện
hành liên quan, data-path config và loader, contracts/live/primary, các bảng
kết quả mới để xác định artifact; kiểm tra source robot, generated worlds,
reference paths, hidden folders và symlinks. Không tuyên bố đã đọc và hiểu
từng dòng của mọi archive script hay từng binary dataset/model.

Scan reference paths đọc **72.779 file text** trong project (có giới hạn
2 MB/file, bỏ environment/Git/build/install/log); kết quả regex có cả path
template/upstream/log lịch sử, **không dùng mọi path chưa tồn tại làm bằng
chứng mất dataset**. Sau đó inventory filesystem quét các roots nêu trong
`SNAPSHOT_SUMMARY.json`, không bỏ hidden files, không follow symlink:

- 454.220 file thường, 85.070 directory, 8.249 symlink; không có lỗi đọc khi
  tạo inventory trong phạm vi này. Folder kiểm kê đang sinh được loại khỏi
  chính file inventory để tránh tự tham chiếu.
- `FILE_INVENTORY.tsv.gz`: paths, type, bytes, mtime, mode, inode/hardlink và
  symlink target; **metadata inventory, không hash toàn bộ 454.220 file**.
- `DIRECTORY_SIZES.json`: số đo `du` từng cây, không cộng mù các hàng lồng nhau.
- `SYMLINK_INVENTORY.json`, `BROKEN_SYMLINK_SUMMARY.json`,
  `INVENTORY_ERRORS.json`: links và lỗi truy cập.
- `EXTERNAL_TEXT_REFERENCES.json`: toàn bộ matches regex để tra lại source
  reference; nhiều match không phải runtime dependency.
- `CRITICAL_FILES.json` và `CRITICAL_FILES.sha256`: hashes thật của **37 file**,
  checkpoint/config/calibrator/profiles selected Adapter khớp registry.
- `GIT_REPOSITORIES.json`, `git_*`: commit/status/diff-stat root, RoboRefer,
  SAM2, ur3, uet_ur3 và các Git repo bên workspace driver. Không commit/reset
  hoặc dọn worktree. Root status hiện có hơn 110.000 file untracked.
- `capture_inventory.py`: helper đọc inventory/exports, chỉ ghi cạnh script;
  không chạy mô hình hoặc kết nối robot. Có thể chạy lại trước copy cuối nếu
  dữ liệu thay đổi, nhưng đây là một snapshot tại thời điểm ghi.

Không kiểm kê ổ khác chưa mount, lịch sử chat nằm trên dịch vụ, hay file đã
xóa khỏi filesystem. `lsblk` có thiết bị `sda` khoảng 1,8 TiB nhưng không có
mountpoint tại thời điểm kiểm kê; chưa xác định đó là ổ đích của người dùng
và không format/mount/ghi lên thiết bị này.

**Thứ tự ưu tiên khi sắp hết thời gian:** toàn `myproject` → `uet_ur3` →
packages user Python3.10 → PPTX/Markdown/paper ngoài repo → robot description
assets/Fuel cache → các overlays/workspace/log tùy nhu cầu. Phương án copy
nguyên `ur_ws` và danh sách roots tránh phải chọn thủ công từng artifact.
