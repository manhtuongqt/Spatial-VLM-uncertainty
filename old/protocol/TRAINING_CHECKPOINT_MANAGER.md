# Training checkpoint manager

Tool: `protocol/training_checkpoint_manager.py`

Môi trường chạy:

```bash
.conda-roborefer/bin/python protocol/training_checkpoint_manager.py audit \
  results/pcra_u_runs/<run_id>/checkpoints
```

## Tích hợp vào training loop

```python
from protocol.training_checkpoint_manager import save_checkpoint

saved = save_checkpoint(
    run_dir / "checkpoints",
    run_id=run_id,
    model=pcra_u,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    epoch=epoch,
    global_step=global_step,
    selection_split="dev",       # overfit smoke dùng "train"
    selection_metric="dev_loss",
    selection_mode="min",
    selection_value=dev_loss,
    metrics={
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "gradient_norm": gradient_norm,
    },
    identity={
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "code_sha256": code_sha256,
        "feature_contract_sha256": feature_contract_sha256,
    },
    sampler_state=sampler.state_dict(),
    extra_state={"gradient_accumulation_offset": accumulation_offset},
    runtime={"gpu": gpu_name, "torch": torch.__version__, "dtype": str(dtype)},
)
```

Tool cập nhật hai pointer JSON:

- `last.json`: checkpoint mới nhất để resume;
- `best_<metric>.json`: checkpoint tốt nhất theo metric/split/mode đã khóa.

`calibration`, `test_iid` và `test_ood` bị từ chối khi dùng làm `selection_split`.

## Resume đầy đủ

```python
from protocol.training_checkpoint_manager import resume_training

state = resume_training(
    run_dir / "checkpoints",
    pointer="last",
    model=pcra_u,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    expected_identity=identity,
)

start_epoch = state["metadata"]["epoch"]
global_step = state["metadata"]["global_step"]
sampler.load_state_dict(state["sampler_state"])
```

Resume kiểm tra SHA-256 trước khi đọc `training_state.pt`, sau đó khôi phục model, optimizer, scheduler, scaler, RNG, sampler và extra state. Identity khác config/dataset/split/code/contract sẽ bị từ chối.

## Evaluation an toàn

Dùng `load_model_for_evaluation(...)`. Hàm này chỉ đọc `model.safetensors` sau khi verify manifest và không unpickle `training_state.pt`.

## Policy lưu

- lưu `last` theo interval và cuối epoch;
- đánh giá/chọn `best` chỉ tại evaluation interval đã khóa;
- crash checkpoint nếu cần phải có loại riêng, không được tự promote thành best;
- không tự động xóa checkpoint;
- chỉ distributed rank 0 được ghi;
- checkpoint directory đã publish là bất biến;
- chạy `audit` trước resume và trước archive/freeze.

## Test

```bash
.conda-roborefer/bin/python protocol/test_training_checkpoint_manager.py
```

Test hiện kiểm tra atomic publish, `last/best`, exact resume, identity guard, split leakage guard, NaN guard và tamper detection.
