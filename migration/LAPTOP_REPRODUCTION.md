# Khôi phục P-CRA-U trên laptop từ GitHub + Drive

Repository: `manhtuongqt/Spatial-VLM-uncertainty`, private.
Snapshot source/URDF/world/assets ngày 08/10/2026; không phải kết quả kiểm thử
thực tế trên laptop của người dùng. GPU/OS/RAM laptop chưa được cung cấp.

## 1. Chuẩn bị đường dẫn và data

Đường dẫn ít thay đổi nhất so với lab:

```bash
mkdir -p /home/dhcn/ur_ws/src
git clone git@github.com:manhtuongqt/Spatial-VLM-uncertainty.git \
  /home/dhcn/ur_ws/src/myproject
cd /home/dhcn/ur_ws/src/myproject
```

SSH chỉ dùng nếu laptop đã đăng ký SSH key; có thể clone HTTPS với GitHub CLI
hoặc credential manager. Vì repo private, phải đăng nhập tài khoản được cấp quyền.

Nếu laptop có username khác, clone vào vị trí sở hữu được. Nhiều freeze locks
vẫn chứa `/home/dhcn/ur_ws/src/myproject`; cần tạo path alias phù hợp hoặc một
resolver/relocation bundle riêng có kiểm chứng. **Không sửa trực tiếp các
locks/calibrator/source bằng search–replace toàn cây.** Chưa kiểm chứng phương
án relocation trên laptop, nên giữ bản gốc và hashes để đối chiếu.

Giải nén các archive Drive từ root project, giữ các prefixes `old/`, `new/`,
`RoboRefer/`, `sam2/`. Danh sách cần có nằm ở
`migration/DATA_REQUIRED_PATHS.txt`; dataset là 1.600 train + 400 dev +
1.000 calibration + 1.000 Test-IID. Không đổi family/split hoặc capture lại IID.

Metadata nhỏ đã được GitHub giữ tại `reproducibility/locked/`; có thể phục hồi
vào root bằng:

```bash
rsync -a reproducibility/locked/ ./
```

Weights/checkpoints từ Drive phải khớp với metadata này, không chọn checkpoint
có tên tương tự từ run khác. `active_experimental_profile.json` vẫn chọn
Adapter/logistic33; live44, Tasks60 và RoboRefer primary là các nhánh riêng.

## 2. Python: dựng lại cả hai runtime

Máy lab: Ubuntu22.04.5, RTX2000Ada16GiB, driver580.173.02. Backbone worker dùng
Python3.10/Conda sạch, Torch2.5.1/torchvision0.20.1; Sidecar bundle đã chạy
Torch2.13.0+cu130 từ user site. Xem exports tại `migration/`.

- `conda_environment.yml` chứa prefix/editable paths của lab; kiểm tra chúng
  khi tạo env mới, không coi env YAML là một installer portable đã test.
- `conda_explicit.txt` là package lock platform của lab.
- `pip_conda_clean.txt` và `pip_user_python310.txt` phân biệt Conda với user site.
  Đừng cài hai version Torch đè lên một env duy nhất rồi coi như tương đương.
- Source RoboRefer/SAM2 đã nằm trong repo; cài dependencies/editable packages
  từ chính snapshot này, giữ source bytes của bundle.
- Script hiện tìm `.conda-roborefer/bin/python3.10`/`bin/python` trong project.
  Nếu đặt env nơi khác, cần cấu hình/alias đúng và kiểm tra source/hash guards.

Ví dụ kiểm tra hai đường sau khi đã dựng env, không chạy train:

```bash
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 -B -c \
  'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
.conda-roborefer/bin/python3.10 -B -c \
  'import torch; print(torch.__version__, torch.__file__, torch.cuda.is_available())'
```

Full RGB-D generation/AMP inference hiện đã được kiểm chứng trên CUDA của
lab, chưa kiểm chứng trên CPU/Apple Silicon hoặc GPU laptop ít VRAM. Không
tuyên bố clone repo bảo đảm realtime hoặc tái hiện y hệt mọi số trên phần cứng khác.

## 3. ROS/Gazebo/URDF

Hệ tương ứng là ROS2Humble/GazeboFortress6.18/MoveIt2 trên Ubuntu22.04.
Danh sách versions trong `migration/dpkg_installed.tsv`, `gazebo_versions.txt`.
Windows native không phải runtime Gazebo/ROS đã đánh giá; nên chuẩn bị môi
trường Linux phù hợp trước khi mô phỏng.

`ur3/` có đầy đủ source robot hiện tại, YCB meshes/textures, SusGrip, camera
mount, URDF/Xacro/SDF và MoveIt/controller configuration. Clone này không
có các package trùng từ `uet_ur3` bên cạnh workspace lab. Build vào overlay mới:

```bash
source /opt/ros/humble/setup.bash
colcon --log-base log_laptop build --base-paths ur3 --build-base build_laptop \
  --install-base install_laptop --symlink-install
```

Dependencies robot/control và compiler phải được cài trước; nếu rosdep báo
thiếu, tra package.xml và inventory, không chọn ngẫu nhiên ROS distribution khác.

Các world exported có URI tới `/opt/ros/humble/share/ur_description` và
`realsense2_description`. Cài hai package description đúng version hoặc khôi
phục assets snapshot tại `robot_assets/system_description/` vào paths mà
runtime thực sự resolve. Các assets này không thay thế libs ROS/Gazebo.

Demo mặc định/four-panel/audit đang là perception V2 lịch sử; nó không tự
chuyển thành Adapter/Tasks60/primary hoặc robot pick-and-place success.
Đọc `new/demo_gazebo/README.md` trước khi launch; không chạy motion chỉ để
kiểm tra migration. Trước hết kiểm tra `ros2 pkg prefix`, xacro/meshes/world
và observation-only preview.

## 4. Kiểm tra nội dung trước inference

Ngay sau clone có thể kiểm tra riêng code/assets:

```bash
python3 -B migration/verify_restore.py --code-only
```

Sau khi đã khôi phục payloads Drive:

```bash
python3 -B migration/verify_restore.py
```

Verifier chỉ dùng standard library: kiểm tra source-copy SHA256, required
data/model/artifact paths và các file critical. Không fit/evaluate model hay
điều khiển robot. Tùy chọn `--hash-critical` đọc weights thật và kiểm tra
checksum; mất thêm thời gian nhưng không dùng GPU.

Sau khi data/runtime đúng, dùng entry point theo nhánh mong muốn:

- Adapter baseline: scripts/config/profile hiện hành trong `new/docs/MODEL_SELECTION.md`.
- Live44: `new/scripts/infer_spatial_variant.py`, bundle
  `new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json`.
- Tasks60: `new/scripts/infer_task_variant.py`, release bundle
  `new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/bundle_release.json`.
- RoboRefer target primary: `new/scripts/infer_roborefer_primary.py`; README
  `new/docs/ROBOREFER_PRIMARY_INFERENCE.md`. Primary risk vẫn chưa calibrated,
  không gán calibrator Sidecar cho điểm RoboRefer.

Ảnh đầu vào phải theo đúng contract RGB/relative-depth640x480; masks/labels
chỉ vào evaluator. Có thể kiểm tra một sample train/dev đã giữ, rồi so output
với saved case phù hợp cùng bundle. Không train lại hoặc chọn threshold bằng
IID khi chỉ đang khôi phục môi trường.

## 5. Những phần chưa hoàn tất

Google Drive folder/link/upload chưa được xác nhận trong lần chuẩn bị code.
Repo có manifest chính xác, nhưng tensor payloads/data không ở GitHub.
Tái lập trên laptop chỉ có thể xác nhận sau khi restore các gói và kiểm tra
phần cứng/runtime. Hai helpers `capture_inventory.py`/`prepare_code_snapshot.py`
được dùng ở lab trong `plan/MIGRATION_BACKUP_20261008`; không chạy nguyên
chúng trên laptop mà chưa chỉnh phạm vi paths. Dùng `verify_restore.py` để
kiểm tra snapshot đã clone.
