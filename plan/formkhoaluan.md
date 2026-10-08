Được. Với **plan mới + kiến trúc mới trong hình**, cách viết khóa luận phải đổi khá nhiều so với kiến trúc cũ. Trục của luận văn bây giờ không còn là “RoboRefer + vài head mở rộng”, mà là:

\[
\boxed{
\text{RoboRefer point-only grounding}
\rightarrow
\text{Query-centric relation representation}
\rightarrow
\text{P-CRA-U}
\rightarrow
\text{Calibration}
\rightarrow
\text{Selective decision}
}
\]

Đây cũng chính là câu chuyện mà plan V3 đang xác định: RoboRefer giữ vai trò model nền, P-CRA-U tạo spatial distribution và uncertainty theo nguồn, calibrator chuyển các bằng chứng đó thành grounding risk, sau đó policy dùng risk để quyết định có thực thi hay không. 

Tôi đề xuất khóa luận theo **6 chương**, nhưng nội dung bên trong phải bám rất sát pipeline bạn vừa chốt.

---

# 1. Khung tổng thể khóa luận

Cấu trúc phù hợp nhất là:

> **Chương 1. Tổng quan**  
> **Chương 2. Cơ sở lý thuyết và nghiên cứu liên quan**  
> **Chương 3. Phương pháp đề xuất**  
> **Chương 4. Xây dựng hệ thống và thiết lập thực nghiệm**  
> **Chương 5. Kết quả và đánh giá**  
> **Chương 6. Kết luận và hướng phát triển**

Số trang tham khảo:

| Phần | Số trang nên có |
|---|---:|
| Mở đầu / Abstract | 2–4 |
| Chương 1 | 7–10 |
| Chương 2 | 15–22 |
| Chương 3 | 22–30 |
| Chương 4 | 12–18 |
| Chương 5 | 20–30 |
| Chương 6 | 3–5 |
| Phụ lục | tùy |

Nếu làm đầy đủ đúng plan này, phần nội dung chính khoảng **80–110 trang** là hợp lý.

---

# 2. Trước hết phải xác định đúng “câu chuyện” của khóa luận

Plan mới không cố xây một Spatial VLM foundation model mới. Trong tài liệu, “Spatial VLM” chỉ mô tả bài toán VLM hiểu vị trí và quan hệ không gian; cách gọi chính xác hơn là **Spatial Vision-Language Grounding**. fileciteturn2file0L73-L83

Vì vậy storyline của toàn bộ khóa luận nên là:

\[
RGB + Depth + Language
\]

\[
\Downarrow
\]

\[
RoboRefer
\rightarrow
p_{RR}
\]

nhưng:

\[
p_{RR}=(u,v)
\]

chỉ là một điểm.

Một point không thể biểu diễn tốt:

\[
ambiguity,\ multimodality,\ absent\ target,\ relation\ conflict,\ uncertainty
\]

Đây chính là khoảng trống mà plan chỉ ra. fileciteturn2file0L242-L268

Do đó ta xây:

\[
P\text{-}CRA\text{-}U
\]

để tạo:

\[
H_{target}(u,v)
\]

\[
H_{anchor_k}(u,v)
\]

\[
relation\ evidence
\]

\[
answerability
\]

\[
source\ uncertainty
\]

Sau đó:

\[
Independent\ Calibrator
\rightarrow
r_{ground}
\]

rồi:

\[
EXECUTE/
REOBSERVE/
ASK\_USER/
ABSTAIN
\]

Đây là xương sống của luận văn.

---

# 3. CHƯƠNG 1 – TỔNG QUAN

Chương 1 không đi sâu công thức hay implementation.

Nó chỉ cần trả lời:

> Vì sao bài toán này cần nghiên cứu?  
> Hạn chế hiện nay là gì?  
> Khóa luận giải quyết phần nào?  
> Đóng góp dự kiến là gì?

Tôi đề xuất:

# CHƯƠNG 1. TỔNG QUAN

## 1.1. Bối cảnh nghiên cứu

### 1.1.1. Vision-Language Model trong robot

Dẫn từ robot manipulation.

Một robot muốn làm theo câu:

> “Pick up the banana left of the apple and nearer to the camera than the mustard bottle.”

thì không thể chỉ nhận dạng “banana”.

Nó phải đồng thời hiểu:

\[
target = banana
\]

\[
relation_1 = left\_of
\]

\[
anchor_1 = apple
\]

\[
relation_2 = nearer\_to\_camera\_than
\]

\[
anchor_2 = mustard\ bottle
\]

Từ đây dẫn đến VLM và visual grounding.

---

### 1.1.2. Spatial Vision-Language Grounding

Giải thích bài toán:

\[
(I,D,q)
\rightarrow
target
\]

Trong đó:

\[
I:
RGB
\]

\[
D:
Depth
\]

\[
q:
Language\ instruction
\]

Nhấn mạnh đây không chỉ là object detection.

Object detection hỏi:

> “Trong ảnh có những vật gì?”

Visual grounding hỏi:

> “Đối tượng nào mà câu lệnh đang nói tới?”

Spatial grounding còn thêm:

> “Đối tượng nào thỏa các ràng buộc không gian trong câu lệnh?”

---

### 1.1.3. Nhu cầu biểu diễn độ tin cậy

Đây là phần dẫn vào uncertainty.

RoboRefer baseline hiện trả:

\[
p_{RR}=(u,v)
\]

Nhưng point này không nói được:

- có một hay nhiều target hợp lệ;
- target có tồn tại không;
- model có chắc về relation không;
- depth có đáng tin không;
- có nên cho robot hành động không.

Plan mới chính xác tập trung vào vấn đề này. fileciteturn2file0L85-L106

---

# 4. 1.2. Bài toán nghiên cứu

Đây nên là phần formalize.

Input:

\[
\mathcal X=
(I,Z_{rel},Z_{metric},q,K,T)
\]

trong đó:

\[
I
\]

là ảnh RGB,

\[
Z_{rel}
\]

là relative depth,

\[
Z_{metric}
\]

là metric depth,

\[
q
\]

là instruction,

\[
K
\]

là camera intrinsic,

và:

\[
T
\]

là transform camera–robot nếu cần geometry.

Plan target architecture cũng dùng chính tập observable input này. fileciteturn2file0L374-L411

Output mục tiêu gồm:

\[
H_{target}(u,v)
\]

\[
H_{anchor_k}(u,v)
\]

\[
E_{relation}
\]

\[
s_{ans}
\]

\[
s_{src}
\]

sau calibration:

\[
r_{ground}\in[0,1]
\]

Cuối cùng:

\[
a=
\pi(r_{ground},answerability,geometry)
\]

với:

\[
a\in
\{
EXECUTE,
REOBSERVE,
ASK\_USER,
ABSTAIN
\}
\]

---

# 5. 1.3. Những thách thức hiện tại

Không cần liệt kê quá nhiều.

Nên gom thành 4 vấn đề chính.

## 1.3.1. Point-only grounding không biểu diễn đầy đủ không gian

Ví dụ có hai quả chuối giống nhau.

Một point:

\[
p=(u,v)
\]

chỉ chọn một điểm.

Trong khi phân bố:

\[
H(u,v)
\]

có thể giữ được hai mode:

\[
m_1,m_2
\]

Từ đó model nhận ra ambiguity.

---

## 1.3.2. Relation reasoning còn implicit

RoboRefer có thể xử lý relation trong hidden feature nhưng không trực tiếp đưa ra:

\[
target
\]

\[
anchor
\]

\[
relation
\]

\[
relation\ confidence
\]

Đây là lý do cần Language Query Graph.

Plan cũng nêu rõ RoboRefer hiện không xuất target–relation–anchor structure, anchor localization hay relation-consistency explanation. fileciteturn2file0L258-L268

---

## 1.3.3. Nguồn gây bất định bị trộn lẫn

Failure có thể đến từ:

\[
semantic
\]

\[
relation
\]

\[
spatial
\]

\[
depth
\]

\[
occlusion
\]

Plan giới hạn P-CRA-U vào năm nguồn perception này. fileciteturn2file0L270-L285

---

## 1.3.4. Confidence thô chưa đủ cho robot

Entropy hay logit chưa phải calibrated probability.

Do đó:

\[
raw\ evidence
\neq
P(error)
\]

Cần:

\[
Calibrator
\]

để sinh:

\[
r_{ground}
=
P(
grounding\ error
\mid
observable\ evidence
)
\]

Plan cũng phân biệt rõ raw score và calibrated probability. fileciteturn2file0L287-L293

---

# 6. 1.4. Mục tiêu nghiên cứu

## Mục tiêu tổng quát

Có thể viết:

> Xây dựng và đánh giá phương pháp định lượng và hiệu chuẩn độ bất định đa phương thức có nhận biết quan hệ cho bài toán spatial vision-language grounding, trong đó RoboRefer-2B-SFT được sử dụng làm backbone cố định, P-CRA-U tạo biểu diễn phân bố không gian và uncertainty theo nguồn, còn bộ hiệu chuẩn độc lập chuyển các bằng chứng mô hình thành grounding risk phục vụ quyết định chọn lọc cho robot.

Câu này rất sát mục tiêu trong plan. fileciteturn2file0L297-L311

## Mục tiêu cụ thể

Có thể giữ 6 mục:

1. Phân tích câu lệnh thành query-centric target–relation–anchor graph.
2. Dự đoán target và anchor spatial distributions.
3. Dự đoán relation evidence và answerability.
4. Ước lượng source uncertainty.
5. Hiệu chuẩn grounding risk.
6. Đánh giá selective policy trên offline replay và Gazebo/UR3.

---

# 7. 1.5. Đối tượng và phạm vi nghiên cứu

## Đối tượng

- RoboRefer-2B-SFT;
- spatial visual grounding;
- relation reasoning;
- multimodal uncertainty;
- calibration;
- selective robot decision.

## Phạm vi

Đây là chỗ phải khóa scope rất rõ:

- không huấn luyện Spatial VLM foundation model từ đầu;
- RoboRefer ưu tiên frozen;
- không xây full scene graph;
- chỉ dùng query-centric graph;
- robot dùng để validation downstream;
- không lấy robot controller làm contribution chính;
- P-CRA-F là optional, không phải core.

Đây đúng với phạm vi mà plan V3 đặt ra. fileciteturn2file0L39-L49

---

# 8. 1.6. Đóng góp của khóa luận

Đừng liệt kê 8–10 “đóng góp”.

Plan đã giới hạn rất hay: **tối đa hai đóng góp phương pháp chính**. fileciteturn2file0L333-L359

Bạn nên viết:

### Đóng góp 1 – P-CRA-U

Một relation-aware spatial uncertainty sidecar gồm:

\[
target/anchor\ spatial\ distributions
\]

\[
relation\ evidence
\]

\[
answerability
\]

\[
source\ uncertainty
\]

### Đóng góp 2 – Multimodal uncertainty calibration

Xây dựng calibration pipeline từ multimodal evidence:

\[
e
\rightarrow
r_{ground}
\]

và sử dụng:

\[
r_{ground}
\]

cho:

\[
EXECUTE/
REOBSERVE/
ASK\_USER/
ABSTAIN
\]

Robot/Gazebo nên gọi là:

> **downstream validation**

không phải contribution thuật toán chính.

---

# 9. 1.7. Cấu trúc khóa luận

Chỉ mô tả mỗi chương 1 đoạn ngắn.

Không cần subsection riêng cho từng chương.

---

# 10. CHƯƠNG 2 – CƠ SỞ LÝ THUYẾT VÀ NGHIÊN CỨU LIÊN QUAN

Đây sẽ là chương khá quan trọng.

Tôi đề xuất:

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ NGHIÊN CỨU LIÊN QUAN

## 2.1 Vision-Language Model

Trình bày:

\[
Image+Text
\rightarrow
Multimodal\ representation
\]

Các thành phần:

- vision encoder;
- language tokenizer/encoder;
- projector;
- transformer;
- cross-modal representation.

Không cần viết Transformer như giáo trình.

---

# 11. 2.2 Visual Grounding và Referring Expression

Định nghĩa:

\[
(I,q)
\rightarrow
location
\]

Phân biệt:

- detection;
- grounding;
- referring expression comprehension;
- spatial grounding.

Đây là nền tảng trực tiếp cho bài toán của bạn.

---

# 12. 2.3 Spatial relationship reasoning

Đây phải là mục lớn.

Chia quan hệ thành:

### Directional

\[
left/right
\]

\[
front/behind
\]

### Distance

\[
near/far
\]

### Camera-relative

\[
nearer\_to\_camera
\]

\[
farther\_from\_camera
\]

### Compositional relation

Ví dụ:

> “banana left of apple and nearer to camera than mustard bottle”

là:

\[
left(banana,apple)
\land
nearer(banana,bottle)
\]

---

# 13. 2.4 Query-centric relation graph

Đây là phần mới phải có trong Chương 2.

Không phải full scene graph.

Graph chỉ chứa:

\[
target
\]

\[
relation_1...relation_L
\]

\[
anchor_1...anchor_K
\]

\[
reference\ frame
\]

Plan định nghĩa đúng cấu trúc này. fileciteturn2file0L115-L133

Bạn cần giải thích:

\[
Q=
\{
q_{target},
q_{relation_l},
q_{anchor_k}
\}
\]

và tại sao query-centric tốt hơn full scene graph cho scope đồ án.

---

# 14. 2.5 RGB-D perception

Giải thích hai loại depth.

### Relative depth

Dùng như modality cho model:

\[
Z_{rel}
\]

### Metric depth

Dùng cho geometry:

\[
Z_{metric}
\]

Đây là chỗ rất quan trọng.

Plan tách hai mục đích này rõ ràng.

Pixel:

\[
(u,v)
\]

cùng metric depth:

\[
z
\]

back-project:

\[
X=
\frac{(u-c_x)z}{f_x}
\]

\[
Y=
\frac{(v-c_y)z}{f_y}
\]

\[
Z=z
\]

Sau đó:

\[
P_{base}
=
T_{base}^{camera}
P_{camera}
\]

---

# 15. 2.6 RoboRefer

Phải có riêng một mục.

Trình bày:

- RoboRefer là gì;
- RGB/RGB-D input;
- vision/depth feature;
- language processing;
- point output;
- ưu điểm;
- limitation.

Điểm dẫn:

\[
RoboRefer
\rightarrow
p_{RR}
\]

nhưng:

\[
p_{RR}
\]

không phải probability distribution.

---

# 16. 2.7 Spatial distribution

Giải thích vì sao dùng heatmap:

\[
H_{target}(u,v)
\]

thay vì chỉ coordinate.

Một spatial distribution có thể cho:

- MAP;
- entropy;
- mode count;
- peak margin;
- covariance;
- top-k modes.

---

# 17. 2.8 Multimodal uncertainty

Giải thích uncertainty ở bài toán của bạn theo **nguồn**, thay vì chỉ aleatoric/epistemic chung chung.

\[
u_{src}
=
[
u_{sem},
u_{rel},
u_{spa},
u_{depth},
u_{occ}
]
\]

Giải thích từng thành phần.

### Semantic

không chắc object identity.

### Relation

không chắc relation/reference frame.

### Spatial

nhiều location hợp lệ.

### Depth

depth thiếu hoặc corrupt.

### Occlusion

object/anchor bị che.

---

# 18. 2.9 Answerability

Bốn trạng thái:

\[
FOUND
\]

\[
AMBIGUOUS
\]

\[
ABSENT
\]

\[
INSUFFICIENT\_EVIDENCE
\]

Đây phải được giải thích thật rõ.

Ví dụ:

**FOUND**

Có một target phù hợp.

**AMBIGUOUS**

Có nhiều target cùng thỏa.

**ABSENT**

Không tồn tại target.

**INSUFFICIENT_EVIDENCE**

Không đủ dữ liệu để kết luận.

Plan dùng đúng bốn lớp này. fileciteturn2file0L580-L605

---

# 19. 2.10 Calibration

Đây sẽ là nền tảng lý thuyết quan trọng nhất sau uncertainty.

Giải thích:

\[
score\neq probability
\]

Calibration muốn:

\[
P(error|score=s)\approx s
\]

Nếu:

\[
r_{ground}=0.1
\]

thì lý tưởng khoảng 10% các sample có risk tương tự thực sự sai.

Giới thiệu:

- temperature scaling;
- Platt scaling;
- logistic calibration;
- isotonic regression;
- calibration bằng MLP nhẹ.

---

# 20. 2.11 Calibration metrics

Giới thiệu:

### Brier

\[
BS=
\frac1N
\sum_i
(p_i-y_i)^2
\]

### NLL

\[
NLL
=
-\frac1N
\sum_i
[
y_i\log p_i+
(1-y_i)\log(1-p_i)
]
\]

### ECE

\[
ECE=
\sum_m
\frac{|B_m|}{N}
|
acc(B_m)-conf(B_m)
|
\]

### Reliability Diagram

Cần hình minh họa.

---

# 21. 2.12 Selective prediction

Đây là nền tảng cho policy.

Cho:

\[
r_{ground}
\]

robot chỉ execute khi:

\[
r_{ground}<\tau
\]

Coverage:

\[
Coverage=
\frac{N_{accepted}}{N}
\]

Risk:

\[
Risk=
\frac{N_{error,accepted}}{N_{accepted}}
\]

Vẽ risk–coverage curve.

---

# 22. 2.13 Spatial confidence region

Nếu bạn thực hiện optional conformal.

Giải thích set-valued prediction:

\[
C_\alpha
\]

với mục tiêu:

\[
P(Y\in C_\alpha)\geq1-\alpha
\]

Phải báo cùng:

\[
coverage
\]

và:

\[
area
\]

vì region phủ cả ảnh thì coverage cao nhưng vô nghĩa.

Plan yêu cầu đúng trade-off này. fileciteturn2file0L678-L690

---

# 23. 2.14 Các nghiên cứu liên quan

Có thể chia thành:

- spatial VLM / grounding;
- referring-expression grounding;
- relation-aware grounding;
- uncertainty quantification;
- multimodal calibration;
- selective prediction;
- robot manipulation.

Cuối cùng làm bảng:

| Method | RGB-D | Relation | Spatial distribution | Answerability | UQ | Calibration | Robot |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | | | | | | | |
| B | | | | | | | |
| RoboRefer | ✓ | implicit | ✗ | ✗ | ✗ | ✗ | ✓ |
| Proposed | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Bảng phải dựa trên paper thật khi viết bản cuối.

---

# 24. 2.15 Khoảng trống nghiên cứu

Kết luận Chương 2 bằng 4 gap:

\[
G_1:
point-only\ grounding
\]

\[
G_2:
implicit\ relation\ reasoning
\]

\[
G_3:
source\ uncertainty\ chưa\ tách
\]

\[
G_4:
raw\ confidence\ chưa\ calibrated
\]

Sau đó dẫn:

> Chương 3 đề xuất kiến trúc P-CRA-U kết hợp query-centric relation representation, spatial distribution, source-aware uncertainty và independent calibration để giải quyết các khoảng trống trên.

---

# 25. CHƯƠNG 3 – PHƯƠNG PHÁP ĐỀ XUẤT

Đây là **chương quan trọng nhất**.

Hình bạn vừa gửi nên đặt ngay đầu Chương 3:

> **Hình 3.1. Kiến trúc tổng thể hệ thống đề xuất**

---

# 26. 3.1 Tổng quan hệ thống

Mô tả 7 khối trong hình:

\[
1.\ Input
\]

\[
2.\ Multimodal\ Fusion
\]

\[
3.\ RoboRefer
\]

\[
4.\ Language\ Query\ Graph
\]

\[
5.\ P\text{-}CRA\text{-}U
\]

\[
6.\ Selective\ Decision
\]

\[
7.\ Validation
\]

Sau đó tóm tắt flow:

\[
RGB,Depth,Language
\rightarrow
RoboRefer
\]

song song:

\[
Language
\rightarrow
Q
\]

sau đó:

\[
R_0+D_0+Q
\rightarrow
P\text{-}CRA\text{-}U
\]

rồi:

\[
evidence
\rightarrow
calibrator
\rightarrow
r_{ground}
\rightarrow
decision
\]

---

# 27. 3.2 Đầu vào đa phương thức

Mô tả:

\[
I\in\mathbb R^{H\times W\times3}
\]

\[
Z_{rel}\in\mathbb R^{H\times W}
\]

\[
Z_{metric}\in\mathbb R^{H\times W}
\]

\[
q=\{w_1,...,w_N\}
\]

và:

\[
K,T
\]

cho geometry.

Phân biệt:

> Relative depth dành cho model.

> Metric depth dành cho geometry/backprojection.

---

# 28. 3.3 Multimodal feature extraction

RGB:

\[
R_0=
E_{rgb}(I)
\]

Depth:

\[
D_0=
E_{depth}(Z_{rel})
\]

Text:

\[
L=
E_{text}(q)
\]

Plan chỉ rõ với tile 448×448:

\[
R_0,D_0
\in
\mathbb R^{32\times32\times1152}
\]

fileciteturn2file0L513-L533

---

# 29. 3.4 RoboRefer-2B-SFT backbone

Checkpoint frozen:

\[
\theta_{RR}
=
constant
\]

RoboRefer nhận multimodal tokens:

\[
X=
[
P_R(R_0);
P_D(D_0);
L
]
\]

và sinh:

\[
p_{RR}=(u,v)
\]

Điểm quan trọng:

P-CRA-U không thay đổi đường output gốc của RoboRefer.

Điều này cho phép giữ baseline nguyên vẹn.

---

# 30. 3.5 Language Query Graph

Câu:

> “Pick up the banana left of the apple and nearer to the camera than the mustard bottle.”

được parse thành:

\[
q_{target}=banana
\]

\[
q_{relation_1}=left\_of
\]

\[
q_{anchor_1}=apple
\]

\[
q_{relation_2}=nearer\_to\_camera\_than
\]

\[
q_{anchor_2}=mustard\ bottle
\]

Graph:

\[
Q=
(V,E)
\]

Trong prototype:

\[
K_{max}=3
\]

\[
L_{max}=3
\]

đúng theo plan. fileciteturn2file0L443-L471

---

# 31. 3.6 Contextual Query Extractor

Không nên encode từng keyword độc lập.

Instruction được contextualize.

Output:

\[
Q=
\{
q_{target},
q_{relation_l},
q_{anchor_k}
\}
\]

Các query slots sau đó tham gia cross-attention với visual features.

---

# 32. 3.7 Relation-conditioned cross-modal fusion

Đây là **trái tim của P-CRA-U**.

Bạn có thể dùng đúng formulation trong plan:

\[
g_i=
\sigma
\left(
W_g
[
R_{0i};
D_{0i};
Q_{relation}
]
\right)
\]

và:

\[
F_i=
LN
\left(
R_{0i}
+
g_i\odot W_dD_{0i}
+
A_{rel}(R_{0i},D_{0i},Q)
\right)
\]

Trong đó:

\[
g_i
\]

là modality gate,

\[
A_{rel}
\]

là relation-conditioned attention.

Plan định nghĩa đây là working design của sidecar. fileciteturn2file0L545-L554

---

# 33. 3.8 Query-centric Relation Head

Head này dự đoán:

\[
H_{anchor_k}
\]

và:

\[
E_{relation}
\]

Mỗi node là spatial distribution:

\[
node_{target}=H_{target}
\]

\[
node_{anchor_k}=H_{anchor_k}
\]

Edge evidence dựa trên:

\[
geometry
\]

\[
depth
\]

\[
relation\ compatibility
\]

\[
reference\ frame
\]

Plan mô tả đúng cơ chế node/edge này. fileciteturn2file0L473-L505

---

# 34. 3.9 Spatial Prediction Head

Dự đoán:

\[
H_{target}(u,v)
\]

với:

\[
\sum_{u,v}H_{target}(u,v)=1
\]

Lấy MAP:

\[
p_{PCRA}
=
\arg\max_{u,v}
H_{target}(u,v)
\]

Nhưng khi multimodal:

\[
\{m_1,m_2,...,m_K\}
\]

phải giữ top-k modes.

Không dùng global mean khi heatmap có nhiều mode. Đây là yêu cầu rõ trong plan. fileciteturn2file0L556-L578

---

# 35. 3.10 Spatial uncertainty features

Từ heatmap lấy:

### Entropy

\[
\mathcal H
=
-\sum_{u,v}
H(u,v)\log H(u,v)
\]

### Peak margin

\[
\Delta
=
p_1-p_2
\]

### Mode count

\[
N_{mode}
\]

### Local covariance

\[
\Sigma_{local}
\]

### RoboRefer disagreement

\[
d_{RR}
=
\|
p_{RR}-p_{PCRA}
\|
\]

Đây là evidence cho calibrator.

---

# 36. 3.11 Source Uncertainty Head

Output:

\[
s_{src}\in\mathbb R^5
\]

với:

\[
[
semantic,
relation,
spatial,
depth,
occlusion
]
\]

Lưu ý rất quan trọng:

Đây là **multi-label task**.

Vì vậy:

\[
p_{src}=
sigmoid(s_{src})
\]

không phải:

\[
softmax
\]

và loss:

\[
L_{src}=BCE
\]

Plan ghi rõ source uncertainty là multi-label. fileciteturn2file0L593-L605

---

# 37. 3.12 Answerability Head

Output:

\[
s_{ans}
\in
\mathbb R^4
\]

Softmax:

\[
p_{ans}
=
softmax(s_{ans})
\]

4 classes:

\[
FOUND
\]

\[
AMBIGUOUS
\]

\[
ABSENT
\]

\[
INSUFFICIENT\_EVIDENCE
\]

---

# 38. 3.13 Metric 3D localization

Sau khi chọn mode:

\[
(u_k,v_k)
\]

lấy:

\[
z_k
\]

từ metric depth.

Back-project:

\[
P_{camera}
=
z_k
K^{-1}
\begin{bmatrix}
u_k\\
v_k\\
1
\end{bmatrix}
\]

Sau đó:

\[
P_{base}
=
T_{base}^{camera}
P_{camera}
\]

3D uncertainty chỉ nên là secondary output đúng như plan. fileciteturn2file0L607-L618

---

# 39. 3.14 Independent calibrator

Đây là điểm phải tách khỏi P-CRA-U.

Evidence vector:

\[
e=
[
entropy,
peak\ margin,
mode\ count,
d_{RR},
s_{ans},
s_{src},
relation\ consistency,
sensor\ quality,
geometry
]
\]

Sau đó:

\[
r_{ground}
=
C_\phi(e)
\]

Plan ưu tiên calibrator nhỏ:

\[
Logistic
\]

hoặc:

\[
MLP
\]

fileciteturn2file0L642-L676

---

# 40. 3.15 Spatial confidence region

Nếu gate mở:

\[
C_\alpha
\]

được tạo thông qua conformal calibration.

Đây là optional.

Không được biến nó thành core requirement nếu dữ liệu chưa đủ.

---

# 41. 3.16 Hàm mất mát

Đúng theo plan:

### Location mass

\[
L_{loc}
=
-\log
\left(
\sum_{(u,v)\in M_t}
H_t(u,v)+\epsilon
\right)
\]

### Interior

\[
L_{int}
=
-\log
\left(
\sum_{(u,v)\in M_{int}}
H_t(u,v)+\epsilon
\right)
\]

### Mask

\[
L_{mask}
=
Dice/BCE(H_t,M_t)
+
\sum_k
valid_k
Dice/BCE(H_{a_k},M_{a_k})
\]

### Relation

\[
L_{rel}
=
CE/BCE(\hat E,E)
\]

### Answerability

\[
L_{ans}
=
CE(s_{ans},y_{ans})
\]

### Source

\[
L_{src}
=
BCE(s_{src},y_{src})
\]

### Counterfactual ranking

\[
L_{rank}
=
\max
(
0,
m-
(u_{perturbed}-u_{clean})
)
\]

Tổng:

\[
\boxed{
L=
\lambda_{loc}L_{loc}
+
\lambda_{int}L_{int}
+
\lambda_{mask}L_{mask}
+
\lambda_{rel}L_{rel}
+
\lambda_{ans}L_{ans}
+
\lambda_{src}L_{src}
+
\lambda_{rank}L_{rank}
}
\]

Đây chính xác là loss hiện được plan định nghĩa. fileciteturn2file0L804-L851

---

# 42. 3.17 Selective decision policy

Policy:

\[
a=
\pi(r_{ground},answerability,geometry)
\]

Có thể formalize:

\[
a=
\begin{cases}
EXECUTE, & FOUND,\ r<\tau_l,\ geometry=pass\\
REOBSERVE, & depth/occlusion\ uncertain\\
ASK\_USER, & AMBIGUOUS\\
ABSTAIN, & ABSENT\ \lor r>\tau_h
\end{cases}
\]

Đây đúng logic policy trong plan. fileciteturn2file0L1083-L1093

---

# 43. CHƯƠNG 4 – XÂY DỰNG HỆ THỐNG VÀ THỰC NGHIỆM

Chương này trả lời:

> Tôi implement và train như thế nào?

Không giải thích lại theory.

---

# 44. 4.1 Hạ tầng

Ghi:

- RTX 2000 Ada 16GB;
- ROS 2 Humble;
- Gazebo Fortress;
- MoveIt 2;
- UR3;
- RGB-D D435i.

Đây chính là setup trong plan. fileciteturn2file0L5-L13

---

# 45. 4.2 Dataset

Phân biệt:

### Prototype

\[
50\ families
\]

\[
250\ samples
\]

### Development

\[
300-500\ families
\]

### Candidate full

\[
2000-4000\ families
\]

Plan định hướng scale như vậy. fileciteturn2file0L741-L759

---

# 46. 4.3 Data split

Split theo:

\[
family\_id
\]

không theo từng image.

Các variant:

- clean;
- corruption;
- paraphrase;
- alternate views

của cùng family phải ở cùng split.

Đây là cực kỳ quan trọng để tránh leakage. fileciteturn2file0L705-L724

---

# 47. 4.4 Counterfactual dataset

Phải mô tả các perturbation:

- flat depth;
- shuffled depth;
- inverted depth;
- missing depth;
- occlusion;
- relation counterfactual;
- target absent;
- ambiguous instances.

Mục tiêu là kiểm tra:

\[
u_{depth}\uparrow
\]

khi depth hỏng,

\[
u_{relation}\uparrow
\]

khi relation sai,

\[
u_{occ}\uparrow
\]

khi occlusion tăng.

---

# 48. 4.5 Training stages

Nên trình bày đúng T0–T5:

### T0

Feature contract.

### T1

Overfit / smoke test.

### T2

Development training.

### T3

Candidate freeze.

### T4

Calibration.

### T5

Locked evaluation.

Plan quy định pipeline này rất rõ. fileciteturn2file0L853-L896

---

# 49. 4.6 Baselines

Đây là phần rất quan trọng.

Tối thiểu:

| Code | Model |
|---|---|
| B0 | RoboRefer RGB-only |
| B1 | RoboRefer RGB-D |
| B2 | RoboRefer + geometry gate |
| G0 | Rule-based relation |
| U0 | Heuristic uncertainty |
| U1 | P-CRA-U without relation slots |
| U2 | P-CRA-U + predicted graph |
| C0 | U2 + scalar calibration |
| P1 | U2 + source-conditioned calibration |
| OG | Oracle graph upper bound |

Đây cũng chính là baseline matrix của plan. fileciteturn2file0L912-L929

---

# 50. 4.7 Ablation

Tối thiểu:

\[
-depth
\]

\[
-relation\ slots
\]

\[
-anchor\ heatmap
\]

\[
-answerability
\]

\[
-source\ supervision
\]

\[
-counterfactual\ ranking
\]

\[
-geometry
\]

\[
scalar\ calibration
vs
source-conditioned
\]

Plan cũng quy định ablation theo đúng các thành phần này. fileciteturn2file0L931-L944

---

# 51. CHƯƠNG 5 – KẾT QUẢ VÀ ĐÁNH GIÁ

Đây là chương không nên viết theo từng block architecture.

Nên viết theo **Research Questions**.

Tôi khuyên 5 câu.

## RQ1

Relation-aware representation có cải thiện grounding không?

## RQ2

Spatial distribution + answerability có phát hiện failure tốt không?

## RQ3

Source uncertainty có phản ứng đúng loại corruption không?

## RQ4

Calibration có làm risk đáng tin hơn không?

## RQ5

Calibrated risk có giúp robot giảm unsafe execute không?

Đây gần như đúng với RQ1–RQ5 trong plan. fileciteturn2file0L313-L329

---

# 52. 5.1 Grounding evaluation

Báo:

\[
point\text{-}in\text{-}target
\]

\[
point\text{-}in\text{-}interior
\]

\[
normalized\ distance
\]

\[
target\ instance\ accuracy
\]

\[
mass\text{-}in\text{-}target
\]

\[
top-k\ recall
\]

Plan có đúng nhóm metric này. fileciteturn2file0L948-L961

---

# 53. 5.2 Relation graph evaluation

Báo:

\[
target/anchor\ extraction\ accuracy
\]

\[
relation\ macro-F1
\]

\[
edge\ precision/recall/F1
\]

\[
multi-anchor\ exact\ match
\]

và:

\[
oracle-gap
\]

---

# 54. 5.3 Answerability

Báo:

- confusion matrix;
- Macro F1;
- balanced accuracy;
- per-class recall;
- false FOUND trên ABSENT/AMBIGUOUS.

Đây quan trọng hơn accuracy đơn thuần vì class imbalance.

---

# 55. 5.4 Uncertainty evaluation

Báo:

### Error detection

\[
AUROC
\]

\[
AUPRC
\]

### Localization correlation

\[
corr(U,error)
\]

### Failure ranking

high uncertainty phải ưu tiên các sample dễ sai.

---

# 56. 5.5 Source attribution

Ví dụ depth corruption:

\[
\Delta u_{depth}
=
u_{depth}^{corrupted}
-
u_{depth}^{clean}
\]

mong đợi:

\[
\Delta u_{depth}>0
\]

nhưng:

\[
u_{semantic}
\]

không nên tăng quá mạnh.

Có thể báo:

- macro/micro F1;
- per-source AUPRC;
- top-1 source;
- specificity.

---

# 57. 5.6 Calibration

Bảng chính:

| Model | Brier ↓ | NLL ↓ | ECE ↓ | AURC ↓ |
|---|---:|---:|---:|---:|
| Raw | | | | |
| Scalar calibration | | | | |
| Source-conditioned calibration | | | | |

Có:

- reliability diagram;
- risk–coverage curve.

Plan coi đây là core metric. fileciteturn2file0L1000-L1010

---

# 58. 5.7 Spatial confidence region

Nếu làm:

\[
Coverage
\]

\[
Coverage\ gap
\]

\[
Region\ area
\]

\[
Coverage-Area\ curve
\]

Không chỉ báo coverage.

---

# 59. 5.8 Ablation study

Đây là bảng quan trọng nhất để chứng minh P-CRA-U.

Ví dụ:

| Variant | Grounding | Answerability F1 | AUPRC | Brier |
|---|---:|---:|---:|---:|
| Full | | | | |
| - depth | | | | |
| - relation | | | | |
| - anchor | | | | |
| - source | | | | |
| - ranking | | | | |

Từ đó trả lời:

> Mỗi thành phần có đóng góp hay không?

---

# 60. 5.9 IID 



# 61. 5.10 Selective decision evaluation

Metric:

\[
unsafe\ execute\ rate
\]

\[
unnecessary\ intervention
\]

\[
reobserve\ recovery
\]

\[
coverage
\]

\[
budget\ exhaustion
\]

Plan cũng định nghĩa đúng nhóm này. fileciteturn2file0L1020-L1028

---

# 62. 5.11 Gazebo / UR3 downstream validation

So sánh:

\[
Forced\ point
\]

vs

\[
Geometry\ gate
\]

vs

\[
Calibrated\ selective\ policy
\]

Plan yêu cầu đúng ba điều kiện này. fileciteturn2file0L1117-L1127

Các metric:

\[
correct\ target\ handoff
\]

\[
wrong-object\ execution
\]

\[
safe\ abstention
\]

\[
IK/planning
\]

\[
grasp/lift/place
\]

\[
end-to-end
\]

Nhưng robot metrics phải tách khỏi perception calibration metrics. fileciteturn2file0L1030-L1041

---

# 63. 5.12 Case study

Tôi rất khuyên làm 4–6 case.

### Case 1 — Clear target

Một mode:

\[
r_{ground}\downarrow
\]

→ EXECUTE.

### Case 2 — Two equivalent targets

Hai mode:

\[
N_{mode}=2
\]

→ ASK_USER.

### Case 3 — Missing depth

\[
u_{depth}\uparrow
\]

→ REOBSERVE.

### Case 4 — Target absent

\[
ABSENT
\]

→ ABSTAIN.

### Case 5 — Occluded anchor

\[
u_{relation}\uparrow
\]

\[
u_{occ}\uparrow
\]

→ REOBSERVE.

Đây cũng chính là demo scenarios mà plan đã quy định. fileciteturn2file0L1131-L1153

---

# 64. 5.13 Failure analysis

Nên tách failure theo layer:

\[
Language\ parser
\]

\[
Grounding
\]

\[
Relation
\]

\[
Spatial\ distribution
\]

\[
Source\ UQ
\]

\[
Calibration
\]

\[
Robot
\]

Không được gộp controller timeout thành perception failure.

Plan cũng cảnh báo điều này. fileciteturn2file0L622-L640

---

# 65. CHƯƠNG 6 – KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1 Kết quả đạt được

Tóm lại:

\[
Query\ graph
\]

\[
P\text{-}CRA\text{-}U
\]

\[
Spatial\ distribution
\]

\[
Source\ uncertainty
\]

\[
Calibration
\]

\[
Selective\ robot\ decision
\]

---

# 66. 6.2 Hạn chế

Phần này bạn đã có khá nhiều limitation thật từ plan.

Ví dụ:

- novelty chưa được xác nhận hoàn toàn;
- relation graph inference có thể lỗi;
- dataset imbalance;
- synthetic-to-real gap;
- source uncertainty không nhất thiết causal;
- 3D covariance có assumptions;
- robot infrastructure failure.

Plan liệt kê rất rõ các rủi ro này. fileciteturn2file0L1409-L1469

---

# 67. 6.3 Hướng phát triển

Có thể đề xuất:

\[
query\ graph
\rightarrow
full\ scene\ graph
\]

nếu thực sự cần;

\[
simulation
\rightarrow
real\ D435i/UR3
\]

\[
2D\ uncertainty
\rightarrow
3D\ calibrated\ region
\]

\[
fixed\ selective\ policy
\rightarrow
learned\ decision\ policy
\]

\[
single-view
\rightarrow
multi-view
\]

\[
P\text{-}CRA\text{-}U
\rightarrow
P\text{-}CRA\text{-}F
\]

nhưng P-CRA-F chỉ khi có bằng chứng cần fusion sâu hơn.

---

# 68. Mapping kiến trúc mới sang các chương

| Khối kiến trúc | Phần trong luận văn |
|---|---|
| 1. Multimodal input | 3.2 |
| 2. Multimodal fusion | 3.3 |
| 3. RoboRefer backbone | 2.6 + 3.4 |
| 4. Language Query Graph | 2.4 + 3.5–3.6 |
| 5.1 Query-centric Relation Head | 3.8 |
| 5.2 Spatial Prediction Head | 3.9–3.10 |
| 5.3 Uncertainty & Answerability | 3.11–3.12 |
| Independent Calibrator | 2.10 + 3.14 |
| Spatial confidence region | 2.13 + 3.15 |
| Decision Module | 3.17 |
| Gazebo/UR3 validation | 4.x + 5.11 |

---

# 69. Bộ Research Question nên dùng

Tôi khuyên khóa luận cuối cùng khóa khoảng **5 RQ chính**:

\[
RQ1:
\]

> Query-centric relation representation có cải thiện spatial grounding trên các câu lệnh relation-dependent không?

\[
RQ2:
\]

> Spatial distribution, answerability và uncertainty evidence có phát hiện grounding failure tốt hơn point-only/heuristic baseline không?

\[
RQ3:
\]

> Source-aware uncertainty có phản ứng đúng với depth, relation, spatial ambiguity và occlusion perturbation không?

\[
RQ4:
\]

> Source-conditioned calibration có cải thiện độ tin cậy của grounding risk so với raw score và scalar calibration không?

\[
RQ5:
\]

> Calibrated grounding risk có giảm unsafe execution mà vẫn duy trì coverage hợp lý trong selective robot manipulation không?

Đây là bộ câu hỏi đủ mạnh để tổ chức toàn bộ Chương 5.

---

# 70. Bộ hình tối thiểu nên có

1. Tổng quan bài toán Spatial Vision-Language Grounding.
2. Kiến trúc RoboRefer.
3. Kiến trúc tổng thể bạn vừa gửi.
4. Language Query Graph.
5. Relation-conditioned sidecar fusion.
6. Target/anchor heatmap.
7. Multi-mode spatial distribution.
8. Source uncertainty visualization.
9. Independent calibrator.
10. Reliability diagram.
11. Risk–coverage curve.
12. Spatial confidence region.
13. Gazebo/UR3 setup.
14. 4 decision cases.
15. Failure cases.

---

# 71. Bộ bảng nên có

1. Related works.
2. Dataset statistics.
3. Relation/state/source distribution.
4. Training hyperparameters.
5. Baselines.
6. Main grounding result.
7. Relation result.
8. Answerability result.
9. Source uncertainty result.
10. Calibration result.
11. Ablation.
12. IID.
13. Selective decision.
14. Robot validation.
15. Resource usage.

---

# 72. Cấu trúc mục lục hoàn chỉnh tôi đề xuất

Đây là phiên bản tôi thấy **khớp nhất với kiến trúc mới**:

> **CHƯƠNG 1. TỔNG QUAN**  
> 1.1 Bối cảnh nghiên cứu  
> 1.2 Bài toán Spatial Vision-Language Grounding trong thao tác robot  
> 1.3 Những thách thức hiện tại  
> 1.4 Mục tiêu nghiên cứu  
> 1.5 Đối tượng và phạm vi nghiên cứu  
> 1.6 Đóng góp của khóa luận  
> 1.7 Cấu trúc khóa luận  
>
> **CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ NGHIÊN CỨU LIÊN QUAN**  
> 2.1 Vision-Language Model  
> 2.2 Visual Grounding và Referring Expression  
> 2.3 Spatial Relation Reasoning  
> 2.4 Query-centric Relation Graph  
> 2.5 RGB-D Perception và hình học camera  
> 2.6 RoboRefer-2B-SFT  
> 2.7 Spatial Distribution cho Visual Grounding  
> 2.8 Multimodal Uncertainty Quantification  
> 2.9 Answerability  
> 2.10 Uncertainty Calibration  
> 2.11 Calibration Metrics  
> 2.12 Selective Prediction  
> 2.13 Spatial Confidence Region  
> 2.14 Các nghiên cứu liên quan  
> 2.15 Khoảng trống nghiên cứu  
>
> **CHƯƠNG 3. PHƯƠNG PHÁP ĐỀ XUẤT**  
> 3.1 Tổng quan kiến trúc  
> 3.2 Đầu vào đa phương thức  
> 3.3 Trích xuất và hợp nhất đặc trưng RGB-D-Language  
> 3.4 RoboRefer-2B-SFT Backbone  
> 3.5 Language Query Graph  
> 3.6 Contextual Query Extractor  
> 3.7 Relation-Conditioned Cross-Modal Fusion  
> 3.8 Query-Centric Relation Head  
> 3.9 Spatial Prediction Head  
> 3.10 Spatial Uncertainty Evidence  
> 3.11 Source Uncertainty Head  
> 3.12 Answerability Head  
> 3.13 Metric 3D Localization  
> 3.14 Independent Multimodal Calibrator  
> 3.15 Spatial Confidence Region  
> 3.16 Hàm mất mát đa nhiệm  
> 3.17 Chính sách quyết định chọn lọc  
>
> **CHƯƠNG 4. XÂY DỰNG HỆ THỐNG VÀ THIẾT LẬP THỰC NGHIỆM**  
> 4.1 Môi trường phát triển  
> 4.2 Dataset prototype và dataset mở rộng  
> 4.3 Chiến lược chia dữ liệu  
> 4.4 Counterfactual và corruption generation  
> 4.5 Tiền xử lý và feature cache  
> 4.6 Thiết lập huấn luyện P-CRA-U  
> 4.7 Huấn luyện và hiệu chuẩn theo giai đoạn  
> 4.8 Baselines  
> 4.9 Ablation settings  
> 4.10 Thiết lập Test-IID 
> 4.11 Gazebo/UR3 setup  
> 4.12 Quy trình selective robot validation  
>
> **CHƯƠNG 5. KẾT QUẢ VÀ ĐÁNH GIÁ**  
> 5.1 Câu hỏi nghiên cứu và tiêu chí đánh giá  
> 5.2 Kết quả Spatial Grounding  
> 5.3 Kết quả Relation Reasoning  
> 5.4 Kết quả Answerability  
> 5.5 Đánh giá Spatial Uncertainty  
> 5.6 Đánh giá Source Uncertainty  
> 5.7 Đánh giá Calibration  
> 5.8 Spatial Confidence Region  
> 5.9 Ablation Study  
> 5.10 IID và Stress Test  
> 5.11 Selective Decision Evaluation  
> 5.12 Gazebo/UR3 Downstream Validation  
> 5.13 Phân tích định tính  
> 5.14 Failure Analysis  
> 5.15 Thảo luận  
>
> **CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**  
> 6.1 Kết quả đạt được  
> 6.2 Hạn chế  
> 6.3 Hướng phát triển

Điểm khác lớn nhất so với cấu trúc trước đây là **P-CRA-U giờ thực sự trở thành trung tâm của Chương 3**, còn calibration là một tầng độc lập sau P-CRA-U. RoboRefer chỉ giữ vai trò frozen backbone và baseline point grounding. Điều này khớp cả với sơ đồ mới lẫn với plan: sidecar sinh target/anchor heatmap, relation evidence, answerability và source uncertainty; calibrator dùng các observable evidence đó để sinh calibrated grounding risk. fileciteturn2file0L374-L411

Và khi viết luận văn, nên giữ đúng giới hạn học thuật trong plan: **P-CRA-U hiện vẫn là tên làm việc nội bộ**, chưa nên tuyên bố đây là một phương pháp hoàn toàn mới trước khi literature/novelty review hoàn tất. 
