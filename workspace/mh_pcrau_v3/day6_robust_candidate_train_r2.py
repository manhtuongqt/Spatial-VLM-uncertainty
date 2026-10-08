#!/usr/bin/env python3
"""Revision 2: preserve the original 432-update budget on the 3x dataset.

Attempt 01 used 54 epochs on three examples per family, tripling the original
number of optimizer updates and saturating log-variance at -8.  This wrapper
reuses the locked implementation with explicit mechanical substitutions for a
lock-defined fixed epoch count and update-count check.  All substitutions are
validated before execution.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import workspace.mh_pcrau_v3.day6_robust_candidate_train as attempt1


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY6_ROBUST_CANDIDATE_R2_LOCK.json"
OUT = ROOT / "ketqua1/07_huan_luyen/ngay_06/amendment_05_robust_candidate_r2"
CHECKPOINT_DIR = ROOT / "workspace/mh_pcrau_v3/checkpoints/day06_robust_candidate_r2"


def revised_main() -> None:
    source = inspect.getsource(attempt1.main)
    substitutions = {
        "evaluate(model, features, target, train_indices, weights, 54, \"augmented_train\")":
            "evaluate(model, features, target, train_indices, weights, lock['training']['fixed_epochs'], \"augmented_train\")",
        "evaluate(model, features, target, old_val_indices, weights, 54, \"old_val_monitor_only\")":
            "evaluate(model, features, target, old_val_indices, weights, lock['training']['fixed_epochs'], \"old_val_monitor_only\")",
        "evaluate(model, features, target, indices, weights, 54, name)":
            "evaluate(model, features, target, indices, weights, lock['training']['fixed_epochs'], name)",
        "\"epoch\": 54,": "\"epoch\": lock['training']['fixed_epochs'],",
        "\"fixed_54_epochs_exact\": len(epoch_rows) == 54,":
            "\"fixed_epochs_exact\": len(epoch_rows) == lock['training']['fixed_epochs'],",
        "global_step == 54 * 24": "global_step == lock['training']['fixed_epochs'] * 24",
    }
    for old, new in substitutions.items():
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f"Expected exactly one source substitution, got {count}: {old}")
        source = source.replace(old, new)
    namespace = dict(attempt1.__dict__)
    namespace.update({"LOCK": LOCK, "OUT": OUT, "CHECKPOINT_DIR": CHECKPOINT_DIR})
    exec(source, namespace)
    namespace["main"]()


if __name__ == "__main__":
    revised_main()
