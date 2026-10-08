# P-CRA-U v0 — Overfit-smoke contract

> **Mục đích duy nhất:** kiểm tra train-only loader, spatial mapping, loss, gradient, checkpoint/resume và khả năng memorize 15 mẫu WP3 smoke. Đây không phải development training và không tạo claim generalization/calibration.

## 1. Quyền đọc và bất biến

- Chỉ đọc 15 entry `train` trong `protocol/wp3_smoke_manifest.json`.
- Mỗi entry thuộc một family khác nhau.
- Feature chỉ đọc từ `results/wp3_feature_hook_smoke_20260821/hook_a/cache`.
- `hook_b` chỉ là determinism evidence WP3, không dùng làm training augmentation.
- Target mask, answerability và source labels là `evaluator_only` nhưng được phép dùng làm **train supervision** trong smoke này; chúng không được ghi vào inference feature cache.
- Không đọc `dev`, `calibration`, prototype `test`, Test-IID hoặc Test-OOD.
- RoboRefer checkpoint/tower/projector/generate path đóng băng hoàn toàn; optimizer chỉ nhận parameter của P-CRA-U sidecar.
- Không scale dataset, không tạo family/variant mới, không publish target sang robot.
- WP0–WP3, dataset và official feature-cache source giữ nguyên hash trước/sau.

## 2. Tập smoke khóa trước optimizer step

- 15 samples / 15 families;
- answerability: 8 `FOUND`, 4 `INSUFFICIENT_EVIDENCE`, 2 `AMBIGUOUS`, 1 `ABSENT`;
- source: semantic, relation, spatial, depth, occlusion;
- có direct, 2D relation, depth relation, occlusion và multi-anchor;
- mỗi sample chỉ xuất hiện đúng một lần trong full batch.

Manifest train-only phải lưu SHA-256 của record, RGB, depth, target mask, cache JSON và safetensors. Manifest có label nên mang policy `TRAIN_SUPERVISION_ONLY_NEVER_INFERENCE_PAYLOAD`.

## 3. Spatial mapping

R0/D0 runtime:

```text
[13, 1024, 1152]
```

- tile 0–11: local tiles, row-major, grid `3×4`, mỗi tile patch grid `32×32`;
- tile 12: thumbnail, không ghép như local spatial tile;
- local patch được average-pool cố định `4×4`, thành `8×8` mỗi tile;
- ghép `3×4` tile thành global grid `24×32`;
- target mask gốc `480×640` được area-resize về `24×32`;
- thumbnail mean feature chỉ được cộng vào global context của answerability/source/query, không phát trực tiếp thành heatmap pixel.

Pooling là phép biến đổi cố định không có parameter. RGB/depth phải dùng cùng tile order và mapping.

## 4. P-CRA-U v0 tối thiểu

### 4.1. Language input

- tokenize instruction bằng lowercase alphanumeric regex;
- token ID là SHA-256 deterministic modulo vocab `4096`, dành ID 0 cho padding;
- mean learned token embedding;
- language-derived relation parser dùng lexical rule khóa trong config, không đọc evaluator relation graph;
- relation embedding được cộng vào query vector.

### 4.2. Relation-conditioned fusion

Với hidden dimension 96:

```text
r = W_r R0
d = W_d D0
q = Query(instruction, parsed_relation)
g = sigmoid(W_g [r; d; q])
f = LN(r + g ⊙ d)
f = FiLM(f | q)
```

Spatial head đọc `f` và xuất target logits `[24,32]`. Global head đọc mean/max spatial feature + thumbnail context + query.

### 4.3. Output heads

- `target_heatmap`: binary spatial logits;
- `answerability`: 4 logits theo thứ tự `FOUND`, `AMBIGUOUS`, `ABSENT`, `INSUFFICIENT_EVIDENCE`;
- `source`: 5 independent logits theo thứ tự `semantic`, `relation`, `spatial`, `depth`, `occlusion`.

Không có calibrator; source logits là raw score, không gọi là probability/confidence.

## 5. Loss

```text
L = 1.0 * L_heatmap + 0.5 * L_answerability + 0.5 * L_source
```

- `L_heatmap`: weighted BCE-with-logits + soft Dice, chỉ active trên ground-truth `FOUND`;
- positive weight tính từ target fraction từng sample, clamp `[1,20]`;
- `L_answerability`: class-weighted cross entropy, inverse-frequency weight được tính một lần từ 15 sample và normalize mean=1;
- `L_source`: BCE-with-logits với per-source positive weight, clamp `[1,14]`;
- không dùng dev/calibration/test để đặt weight hoặc threshold.

## 6. Optimization khóa

- seed `8132026`;
- deterministic algorithms; CuBLAS workspace `:4096:8` phải được set trước import Torch;
- float32 sidecar training; feature input chuyển FP32 sau khi pool;
- full batch = 15;
- AdamW, LR `0.003`, weight decay `0`;
- gradient clip norm `5.0`;
- 600 optimizer steps;
- không scheduler, dropout=0;
- metrics mỗi step; detailed evaluation mỗi 25 step;
- checkpoint ở step 250, 255 và 600;
- model selection trong smoke dùng `train_total_loss`, mode `min`; không tạo final scientific model từ best smoke checkpoint.

Hai run A/B dùng byte-identical manifest/config/code/seed. Run A có resume replay từ step 250 → 255 để so exact với checkpoint step 255.

## 7. One-batch gate trước training

Một forward/backward không optimizer step phải pass:

- tensor shapes đúng;
- total/component losses finite;
- output logits finite;
- tất cả active head có tổng gradient norm hữu hạn và > `1e-10`;
- RoboRefer/feature input không có gradient;
- trainable parameters chỉ thuộc sidecar;
- không có dev/calibration/test read;
- checkpoint manager unit tests đã pass.

Fail thì dừng `FIX_LOADER_MODEL_LOSS_FIRST`; được sửa lỗi implementation nhưng chưa chạy optimizer. Sau khi pass mới khóa code/config hash trong execution lock.

## 8. Overfit gate khóa trước run

Run được `GO_DATA_EXPANSION_DESIGN` khi đồng thời:

1. initial/final loss ratio, dùng median step 1–20 và 581–600, `<= 0.35`;
2. final answerability train accuracy `>= 14/15`;
3. final source micro-F1 tại raw sigmoid threshold 0.5 `>= 0.90`;
4. MAP point-in-target trên 8 `FOUND` samples `>= 7/8`;
5. mean heatmap mass-in-target trên `FOUND` `>= 0.50`;
6. toàn bộ loss/gradient/logit finite và không có skipped optimizer step;
7. run A/B final `model.safetensors` SHA-256 giống nhau và metric JSON canonical giống nhau;
8. resume step 250 → 255 tạo model SHA-256 giống checkpoint A step 255;
9. checkpoint root audit, manifest/source hash và no-split-leakage pass;
10. WP0–WP3/dataset/feature source hashes trước/sau giống nhau.

Nếu loss giảm nhưng spatial/answerability/source gate không pass, quyết định là `FIX_PCRA_U_SMOKE_FIRST`, không scale data. Không được nới threshold sau khi xem kết quả.

## 9. Artifact được phép cập nhật

Trong actual run mới:

- Bảng 7 training/checkpoint;
- F02 total/component loss, gradient norm và LR;
- heatmap/GT/MAP overlay cho 15 train samples;
- runtime/VRAM diagnostic;
- one-batch report, checkpoint audit, determinism/resume comparison và final manifest.

Bảng 2–6, calibration, risk–coverage, ablation và locked-test rows tiếp tục `NOT_RUN`.

## 10. Quyết định sau smoke

PASS chỉ cho phép thiết kế data expansion theo family để bổ sung negative states cho dev/calibration và tạo Test-IID/Test-OOD. PASS không tự động cho phép development training trên prototype hiện tại.
