# P-CRA-U development checkpoint policy

- Manager: `protocol/training_checkpoint_manager.py`.
- Checkpoint directories are immutable; no overwrite and no automatic deletion.
- Store sidecar weights in `model.safetensors`; optimizer, scheduler and RNG state
  remain in the trusted local `training_state.pt`.
- Save after each development epoch and at locked canary steps.
- `last.json` is resume-only. `best_dev_total_loss.json` is the only development
  model-selection pointer.
- Best selection is restricted to split `dev`, metric `dev_total_loss`, mode
  `min`; Calibration/Test are forbidden.
- Identity must bind config, train manifest, feature-cache index, execution lock,
  code and feature contract hashes. Resume with any mismatch must fail closed.
- Audit the entire checkpoint root after canary, after resume replay and after
  full training. Published archives contain checkpoint indexes/hashes only; they
  do not duplicate binary checkpoint payloads.
- Canary checkpoints cannot be promoted into the full development run. Full
  development checkpoints cannot be called final-test or calibrated models.
