# Prompt khởi động session mới

Sao chép phần trong khung dưới đây vào session mới. Nếu session mới không truy cập được workspace, gửi kèm file `SESSION_CONTEXT_20261002.md` và các tài liệu/chương liên quan.

```text
Hãy tiếp tục hỗ trợ đồ án/khóa luận của tôi bằng tiếng Việt trên workspace:
/home/dhcn/ur_ws/src/myproject

Trước khi thực hiện công việc, đọc đầy đủ:
plan/SESSION_CONTEXT_20261002.md

Để hiểu toàn bộ định hướng đồ án, bắt buộc đọc cả pipeline và plan:
1. Mở và xem trực tiếp ảnh plan/pipeline.png, không chỉ đọc tên file.
   Đọc các khối, nhánh, mũi tên, đầu vào/đầu ra và luồng evidence;
   giải thích được đường RGB/depth/ngôn ngữ → backbone → sidecar
   → calibration → policy → kiểm chứng robot trong sơ đồ.
2. Đọc đầy đủ plan/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md,
   gồm kiến trúc, dữ liệu, loss, protocol, work packages, tiêu chí nghiệm thu
   và mục 16 dashboard 2×3. Không chỉ đọc phần đầu hay bản tóm tắt.
3. Đọc đầy đủ plan/formkhoaluan.md để hiểu cấu trúc và yêu cầu sáu chương.
4. Liệt kê các tài liệu trong plan/ và đọc các kế hoạch/bàn giao còn lại
   liên quan, kể cả SESSION_HANDOFF_PROMPT_20261001.md để hiểu lịch sử.
   Phân biệt bản cũ với quyết định mới; không lấy kế hoạch cũ thay artifact.
5. Đọc đầy đủ latex/Chapter1/chapter1.tex đến latex/Chapter6/chapter6.tex.

Bản kế hoạch đang mở của tôi còn nằm ở:
/home/dhcn/Downloads/ke_hoach_v3_spatial_vlm_visual_grounding_uncertainty.md
Đọc bản này và đối chiếu với bản trong plan/; không mặc định chúng đồng bộ.
Nếu file dài hoặc kết quả đọc bị cắt, đọc tiếp cho tới hết, không tuyên bố
đã đọc đầy đủ khi mới xem một phần. Nếu không xem được ảnh pipeline,
yêu cầu tôi gửi ảnh, không suy đoán nội dung từ tên file.

Đối chiếu triển khai với new/README.md, new/docs/ARCHITECTURE.md,
new/docs/DATA_CONTRACT.md và code new/src/pcrau/ khi cần.
Đọc new/hinhanh/README.md trước khi làm hình, và artifact gốc Test-IID
trước khi kết luận số liệu. Tài liệu kế hoạch là mục tiêu, không phải
bằng chứng rằng tất cả thành phần đã được triển khai hay đánh giá.
Đối chiếu từng khối pipeline và từng mục kế hoạch với code/artifact:
phần đã triển khai, phần đã đánh giá, phần chỉ ở mức thiết kế và phần
chưa có bằng chứng. Không tự sửa pipeline hay mở rộng nhiệm vụ theo plan.

Những điểm bắt buộc nhớ:
- Mô hình chính gọi là P-CRA-U, checkpoint lịch sử best V2 epoch 13,
  global step 1120 trên RoboRefer-2B-SFT đóng băng; không thay bằng V3.
- Test-IID 200 family/1000 sample; PIT 92.81% trên 835 mẫu có target,
  so với RoboRefer gốc 86.71%, delta +6.11 pp với CI [1.78,10.49] pp.
- Dev chọn checkpoint, calibration fit sau freeze, test chỉ đánh giá.
  Không train lại, fit threshold bằng test, mở Test-OOD hay chạy robot
  nếu tôi chưa yêu cầu.
- Phân biệt grounding đúng, answerability FOUND, lỗi risk tổng hợp
  và robot task success; spatial source là weak label, relation là proxy.
- Score coverage vùng 86.83% và direct region-mask intersection 95.81%
  là hai event khác nhau; không thay số trong artifact gốc.
- Năm hình mới (19–23) đã dựng bằng Python từ dữ liệu thật, chèn Chương 5.
- Tôi đã biên dịch được ảnh trên Prism. Nguồn LaTeX dùng
  hinhanh/<tên ảnh>.png. Nếu tôi chỉ yêu cầu sửa nguồn ảnh thì chỉ đổi
  đường dẫn; giữ nguyên caption, label, tùy chọn kích thước và nội dung khác.
- Hình hiển thị tên P-CRA-U; tên file/run/sample/hash chứa v2 giữ nguyên.
- Audit Gazebo có điểm đúng 5/5 nhưng ASK_USER 5/5, chưa handoff MoveIt,
  chưa robot motion, chưa phải pick-and-place thành công.
- Dashboard 2×3 ở mục 16 kế hoạch là đề xuất, không tự coi đã triển khai.
- Không bịa số liệu/ảnh, không tạo thêm py/json ngoài nhu cầu công việc,
  không reset/xóa/commit/push hoặc sửa phần ngoài phạm vi tôi giao.

Sau khi đọc, tóm tắt ngắn sự hiểu biết về bài toán, luồng pipeline,
kế hoạch nghiên cứu, kiến trúc thực tế, checkpoint/dữ liệu, trạng thái
khóa luận/hình và giới hạn còn lại. Nêu các khác biệt quan trọng giữa
pipeline/plan mục tiêu và phần thực sự đã triển khai, có kết quả.
Nêu rõ tài liệu nào không truy cập được, không giả vờ đã đọc.
Chưa tự sửa hoặc chạy thí nghiệm; chờ yêu cầu tiếp theo của tôi.
```
