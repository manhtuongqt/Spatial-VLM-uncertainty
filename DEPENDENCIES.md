# External dependencies

Các source tree dưới đây nằm trong workspace nhưng được quản lý như repository
riêng và không được vendor vào Spatial-VLM để tránh đưa model/baseline lớn hoặc
lịch sử không liên quan vào commit này.

| Thành phần | Remote | Commit quan sát | Vai trò |
|---|---|---|---|
| RoboRefer | `https://github.com/Zhoues/RoboRefer.git` | `d97a995ad28376720a4c8beb64915c58ed16c844` | Frozen RGB-D VLM baseline và feature towers |
| SAM 2 | `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Segmentation dependency |
| UR3 workspace | `https://github.com/manhtuongqt/uet_ur3-master.git` | `e94d30cf12b53b7c1ba314252f6fdc1423f07738` | ROS 2 / Gazebo robot integration |

Môi trường đã dùng cho lần chạy 2026-08-24:

- ROS 2 Humble;
- Python environment `.conda-roborefer`;
- PyTorch `2.13.0+cu130`;
- OpenCV `5.0.0`;
- NumPy `1.26.4`;
- NVIDIA RTX 2000 Ada Generation.

Các local modification trong external source tree không được ngầm coi là một
phần của commit root này. Protocol/report ghi hash các file baseline được khóa
để phát hiện drift.
