# Upload data và model P-CRA-U — 08/10/2026

Gói này giữ đúng 2.000 mẫu train/dev, 1.000 calibration, 1.000 IID cũ;
feature caches cùng dữ liệu; model RoboRefer; checkpoints, calibrators,
profiles và kết quả của các bundle đã chọn. Không thêm dataset lịch sử khác.

## Tạo archive local

```bash
cd /home/dhcn/ur_ws/src/myproject
python3 plan/MIGRATION_BACKUP_20261008/pack_drive_payloads.py --create
```

Script kiểm tra paths/dung lượng, tạo tar giữ relative paths, kiểm tra archive
đọc được và tạo SHA256. Không sửa/xóa nguồn, không upload. Nếu archive đã có
và checksum đúng thì bỏ qua; archive chưa xác minh sẽ không bị ghi đè.
Payload trước tar headers khoảng 22.19 GB (20.67 GiB). Có thể dùng
`--group 07_models` để đóng một nhóm nếu không đủ chỗ cho toàn bộ.

Folder đích: `/home/dhcn/pcrau_transfer_staging/`.

| Archive | Nội dung |
|---|---|
| `01_train_dev.tar` | 2.000 mẫu train/dev và manifest |
| `02_calibration.tar` | 1.000 mẫu calibration và manifest |
| `03_test_iid.tar` | 1.000 mẫu IID cũ và manifest |
| `04_features_train_dev.tar` | Cache train/dev |
| `05_features_calibration.tar` | Cache calibration |
| `06_features_test_iid.tar` | Cache IID |
| `07_models.tar` | RoboRefer/models; không gồm SAM2 |
| `08_selected_artifacts.tar` | V2, Adapter, anchor P1, live/Tasks60 và kết quả |

Khi script báo `COMPLETE`, folder có tám archive, checksum từng file,
`SHA256SUMS.txt`, `ARCHIVE_CONTENTS.json` và tài liệu tái lập. Có archive local
không có nghĩa đã backup lên Drive. Chưa có tài khoản/folder Drive được xác thực.

## Upload bằng trình duyệt

1. Trên máy lab, đăng nhập tài khoản của bạn tại https://drive.google.com/.
2. Tạo folder `P-CRA-U_backup_20261008`.
3. Chọn **Mới → Tải thư mục lên**, chọn `/home/dhcn/pcrau_transfer_staging/`.
   Hoặc mở folder đích rồi **Tải tệp lên**, chọn toàn bộ nội dung staging.
4. Chờ mọi file báo hoàn tất. Kiểm tra đủ tám `.tar` và `SHA256SUMS.txt`.
5. Tải các file về laptop, đặt trong cùng một folder rồi chạy:

```bash
sha256sum -c SHA256SUMS.txt
```

Phải có tám dòng `OK`. Sau đó giải nén vào **clone mới** của repo, theo
`LAPTOP_REPRODUCTION.md`; ví dụ từ folder chứa các archive:

```bash
for archive in 0*.tar; do
  tar -xpf "$archive" -C /home/dhcn/ur_ws/src/myproject
done
```

Không giải nén đè vào project có chỉnh sửa riêng. Giữ data nguồn trên lab cho
tới khi bản tải từ Drive đã kiểm tra checksum và tái lập. Không cần gửi mật
khẩu/token trong chat; bạn đăng nhập trực tiếp trên trình duyệt.

## Nếu cần upload bằng dòng lệnh

Máy lab chưa cài/cấu hình rclone tại lúc kiểm tra. Sau khi cài và tự xác thực
Google Drive qua OAuth, đặt tên remote `gdrive`, có thể chạy:

```bash
rclone copy /home/dhcn/pcrau_transfer_staging gdrive:P-CRA-U_backup_20261008 -P --transfers 2
rclone check /home/dhcn/pcrau_transfer_staging gdrive:P-CRA-U_backup_20261008 --one-way
```

`copy` không xóa các file khác ở đích. `check` đối chiếu dung lượng/hash mà
backend hỗ trợ; SHA256 sau khi tải về vẫn là kiểm chứng riêng của gói này.
Tài liệu rclone hiện khuyến nghị OAuth client ID riêng vì shared client đang
được thay đổi trong 2026; xem hướng dẫn chính thức khi cấu hình.

Nguồn:
- [Google: upload files/folders](https://support.google.com/drive/answer/2424368?hl=en)
- [Rclone Drive và OAuth](https://rclone.org/drive/)
- [Rclone copy](https://rclone.org/commands/rclone_copy/)
- [Rclone check](https://rclone.org/commands/rclone_check/)
