# P-CRA-U one-batch forward/backward gate

> **LOCK_EXECUTION_AND_RUN_OVERFIT**

Không có optimizer step; đây chỉ là preflight gradient.

## Gates

| Gate | Result |
|---|---:|
| `spec_lock_verified` | PASS |
| `checkpoint_manager_tests_pass` | PASS |
| `batch_shape_r0` | PASS |
| `batch_shape_d0` | PASS |
| `heatmap_shape` | PASS |
| `answerability_shape` | PASS |
| `source_shape` | PASS |
| `finite_losses` | PASS |
| `finite_logits` | PASS |
| `active_gradients_nonzero` | PASS |
| `feature_inputs_have_no_gradient` | PASS |
| `all_parameters_sidecar_trainable` | PASS |
| `protected_inputs_unchanged` | PASS |
| `no_optimizer_step` | PASS |
| `train_only_manifest` | PASS |

## Loss

- `total`: 3.47880197
- `heatmap`: 2.20457625
- `heatmap_bce`: 1.29776156
- `heatmap_dice`: 0.90681463
- `answerability`: 1.39376712
- `source`: 1.15468466

## Gradient norms

- `target_head`: 1.06005865e+00
- `answer_head`: 1.10281140e+00
- `source_head`: 4.94486733e-01
- `relation_embedding`: 2.22344041e-02
- `fusion_gate`: 1.15637106e-01
