# Phạm vi GitHub và Google Drive đã chốt — 08/10/2026

Repository đích do người dùng tạo:
`https://github.com/manhtuongqt/Spatial-VLM-uncertainty`.
Repository đã chuyển sang private theo lựa chọn của người dùng.

## GitHub

Snapshot riêng giữ source hiện tại và các sửa đổi chưa commit của lab:

- `new/src`, scripts/configs/tests/docs, Gazebo demo source, hình khoa học và
  scripts/contracts/protocol Test-IID.
- `ur3/`: source perception, MoveIt, interfaces, driver, SusGrip; **URDF/Xacro,
  SDF/worlds, YCB meshes và textures**. Không bỏ folder `models` của Gazebo.
- Source RoboRefer/SAM2 local, gồm file untracked cần thiết; giữ license gốc,
  không chỉ ghi link tới upstream vì có sửa đổi local.
- `workspace/`, `protocol/`, `old/protocol/`, `plan/`, LaTeX/báo cáo/hình.
- `robot_assets/system_description/`: snapshot UR và RealSense descriptions
  từ `/opt/ros/humble/share`, để giữ assets dùng bởi world đã xuất.
- `migration/`: môi trường/version inventory, manifest data, hashes, hướng
  dẫn laptop. `reproducibility/locked/`: metadata/config/calibrator/profile
  và lock files nhỏ của các bundle đã chọn, giữ nguyên byte.

Không push tensor checkpoints, RGB-D datasets/caches, Python env, compiled
ROS install/build, Git history của repo lab hoặc toàn output/archive.
Snapshot có `.gitignore` riêng; không đổi Git/worktree của repo gốc.
File lớn hơn 20 MiB trong phần source thông thường được ghi nhận ở
`migration/OMITTED_ITEMS.json`; system description assets được giữ đầy đủ.

## Data Drive: đúng 4.000 mẫu, không lấy mọi dataset lịch sử

| Split | Mẫu | Family | Dataset path trong project |
|---|---:|---:|---|
| train | 1.600 | 320 | `old/roborefer_dataset_v2_1_1_development_400_20260824/` |
| dev | 400 | 80 | Cùng root development, tách theo manifest |
| calibration | 1.000 | 200 | `old/datasets/roborefer_dataset_v2_1_calibration_200_20260824/` |
| Test-IID cũ | 1.000 | 200 | `new/test_iid/dataset/` |

Ba roots dataset khoảng **3,21 GiB**; feature caches tương ứng khoảng
**8,20 GiB**, tổng data + cached features khoảng **11,41 GiB** theo `du`.
Cached features không phải mẫu mới. Các raw captures/metric depth/instance
labels trong đúng ba roots cũng cần giữ cho task heads và supervision; không
chỉ lấy ảnh RGB hoặc `qa.jsonl`.

Các manifests đã kiểm tra đủ 2.000/1.000/1.000 entries. Tất cả 25.439 file
reference được kiểm tra theo các profile (RGB, depth, record, masks, features)
đều tồn tại; đây là existence check, không phải chạy model/đánh giá mới.
Thông tin chi tiết/hashes manifests nằm ở `SCOPED_DATASET_MANIFEST.json`.

`DATA_REQUIRED_PATHS.txt` liệt kê relative paths cần đóng gói từ project root.
Ngoài data, giữ **model/artifacts runtime riêng**:

- `RoboRefer/models/`, checkpoint SAM2 nếu còn dùng segmentation.
- Baseline V2, selected Adapter, P1 anchor pilot, live44 và Tasks60 release r3;
  neural weights + config + freeze lock + calibrator + profile phải đi cùng nhau.
- Kết quả IID của bundle hiện tại và derived task caches/occlusion supplement
  từ cùng train/dev. Không thêm bộ OCID/public benchmark/historical prototypes
  vào gói dataset 4.000 mẫu này.

Các dataset khác, source parquet/ảnh benchmark, old experiment archives và
Python binary env không thuộc gói data thu hẹp. Báo cáo/hình benchmark đã có
trong snapshot; muốn chạy lại public benchmark phải tải source pinned theo
protocol tương ứng. Muốn tái lập một run lịch sử khác có thể cần thêm archive.

## Đóng gói và upload Drive sau khi xác định folder đích

Ưu tiên archive tar theo từng nhóm để giữ Linux symlinks/hardlinks và paths:
`01_train_dev`, `02_calibration`, `03_test_iid`, `04_features`,
`05_models`, `06_selected_artifacts`. Không chép hàng trăm nghìn file package
Python lên Drive; dựng môi trường mới từ exports.

Máy lab còn khoảng 30 GiB trống: tạo/upload/kiểm tra từng archive thay vì tạo
bản sao toàn workspace 90 GiB trên phân vùng này. Giữ nguyên dữ liệu nguồn.
Ví dụ cho dataset development, từ root project:

```bash
mkdir -p /home/dhcn/pcrau_transfer_staging
tar -cpf /home/dhcn/pcrau_transfer_staging/01_train_dev.tar \
  old/roborefer_dataset_v2_1_1_development_400_20260824
sha256sum /home/dhcn/pcrau_transfer_staging/01_train_dev.tar \
  > /home/dhcn/pcrau_transfer_staging/01_train_dev.tar.sha256
```

Các lệnh đóng gói/upload trên **chưa được thực hiện**. Drive folder link vẫn
chưa được cung cấp/xác thực; không ghi rằng data đã upload.

Rclone hỗ trợ Google Drive và `copy`/`check`; cần cấu hình OAuth qua trình
duyệt của chủ tài khoản, không gửi mật khẩu/token trong chat. Nếu dùng rclone,
upload archive và checksum, rồi đối chiếu trước khi xóa file staging tự tạo.
Không dùng `sync` để có nguy cơ xóa file đích không thuộc gói này.

Nguồn hướng dẫn công cụ:
- GitHub file limits: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Google Drive backend: https://rclone.org/drive/
- Copy: https://rclone.org/commands/rclone_copy/
- Check: https://rclone.org/commands/rclone_check/

## Tái lập

Xem `LAPTOP_REPRODUCTION.md`. Clone GitHub **chưa đủ** để chạy inference:
còn cần các gói Drive ở đúng relative paths, neural weights và môi trường.
Runtime cũ có absolute paths/hash guards; không thay hàng loạt paths hoặc
fit lại calibrator để làm cho bundle chạy mà mất provenance.
