# Mã nguồn MH-PCRA-U-v3 trong workspace

Trạng thái: mã phát triển mới đặt trong workspace/mh_pcrau_v3/. Những file
G0 đã khóa ở đường dẫn gốc được giữ lại để kiểm tra SHA-256 và tái lập quyết
định G0; bảng dưới ghi bản sao byte-identical, không phải một G0 run mới.

| Source G0 đã khóa | Bản sao trong workspace | SHA-256 |
|---|---|---|
| RoboRefer/llava/model/spatial/__init__.py | __init__.py | 623115ce5b606c6b2d325e72a8895b8fbe9d41d8a9448b5eda25e2f1443088b9 |
| RoboRefer/llava/model/spatial/hidden_state_adapter.py | hidden_state_adapter.py | ccab70c15875b7c40a2dce6cf078d124abcbd5a2e524a70e3332aebd57bf6cbb |
| RoboRefer/tests/mh_pcrau_v3/test_h_spatial_adapter.py | g0_locked/test_h_spatial_adapter.py | 1862f2d5b32c9a0cf028498cbfafc024913406b900fe943540e4c8dda88d6465 |
| RoboRefer/scripts/mh_pcrau_v3/run_g0_feasibility.py | g0_locked/run_g0_feasibility.py | 6cda74394b0539d1fbb46c601f1b5368b24cf2f4284c8c6f17d2d8ce232760f2 |
| RoboRefer/scripts/mh_pcrau_v3/finalize_g0.py | g0_locked/finalize_g0.py | 23a17cb905cb79fd97ce17293fe4c458d2f66312add4dde6a7e6943f696ece72 |
| ketqua1/03_backbone_h_spatial/ngay_02/attempt_01_source/run_g0_feasibility.py | g0_locked/attempt_01_source/run_g0_feasibility.py | 2ff349897df5167eeac97c2214103e3423d9d55311396c038b1d226a6d466e96 |
| ketqua1/03_backbone_h_spatial/ngay_02/attempt_02_bf16/run_g0_feasibility.py | g0_locked/attempt_02_bf16/run_g0_feasibility.py | 67cbb03246ee42481763e1eca161f877171e01201956f306c16c677510037b90 |

Workspace test file tests/test_h_spatial_adapter.py thay duy nhất import để
kiểm tra adapter tại vị trí mới; g0_locked/test_h_spatial_adapter.py giữ bytes
nguyên gốc. G0 runner/finalizer dùng đường dẫn phụ thuộc source location,
được lưu ở đây để tra cứu, không chạy lại trong thư mục mới.

Code Ngày 3 bắt đầu tại g1/audit_development.py và test tại
tests/test_g1_audit.py. Kết quả chỉ đọc development:
ketqua1/01_dau_vao_tien_xu_ly/ngay_03/INITIAL_SCHEMA_INVENTORY.json.
