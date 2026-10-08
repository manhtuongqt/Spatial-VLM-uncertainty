# P-CRA-U — snapshot code ngày 08/10/2026

Snapshot code riêng của đồ án: RGB-D spatial referring, phrase-conditioned
anchor, verifier trái/phải và các nhánh uncertainty/calibration.

Đường RoboRefer primary là biến thể opt-in. Registry baseline Adapter và các
bundle live44/tasks60 giữ riêng trong backup data; xem `new/docs/` và `plan/`
để phân biệt scope/kết quả của từng bundle.

## Khôi phục

Repository này chỉ chứa code, configs, tài liệu và assets robot nhỏ. **Clone
riêng repo chưa đủ để inference/train.** Khôi phục data/model/artifacts từ
Google Drive vào đúng relative paths trước; xem `migration/README.md` và
`migration/GITHUB_DRIVE_GUIDE.md`. Drive folder link sẽ bổ sung sau khi upload
được kiểm chứng; hiện chưa có upload hoặc policy promotion.

Giữ các dataset/cache được chỉ rõ ở `migration/SCOPED_DATASET_MANIFEST.json`:
1.600 train, 400 dev, 1.000 calibration và 1.000 Test-IID; không cần copy mọi
dataset lịch sử chỉ để chạy model hiện tại. Các paths này còn nằm dưới `old/`.
`RoboRefer/`, `sam2/`, `ur3/` chứa snapshot source local, giữ licenses, gồm các
sửa đổi và file source untracked đã copy. Nested Git history không nằm trong
snapshot này; source trên máy lab giữ nguyên. Licenses gốc được giữ trong cây.

URDF/Xacro, YCB/gripper meshes, textures/worlds nằm trong `ur3/`; camera/UR
description assets bên hệ thống được lưu tại `robot_assets/system_description/`.
Metadata/calibration khóa nằm ở `reproducibility/locked/`, để phục hồi vào
paths gốc cùng neural weights từ Drive. Binaries ROS và Python env không
đưa lên GitHub; exports/version inventory nằm trong `migration/`.

Exports môi trường nằm ở `migration/`; backbone dùng Conda sạch, Sidecar còn
dùng user packages Python3.10. Các paths máy lab phải được xử lý có provenance
khi chuyển máy, không sửa hàng loạt bundles khóa.

Danh sách file đã copy và SHA256 nằm ở `migration/CODE_COPY_MANIFEST.json`.
Items bỏ khỏi snapshot nằm ở `migration/OMITTED_ITEMS.json` và vẫn cần giữ
trong full backup. Đây không phải bản thay thế toàn workspace.
