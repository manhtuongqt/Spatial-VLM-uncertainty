from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def confusion_metrics(truth: Sequence[int], prediction: Sequence[int], classes: int) -> dict[str, Any]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, predicted in zip(truth, prediction):
        matrix[int(target), int(predicted)] += 1
    rows, f1_values = [], []
    for index in range(classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"precision": precision, "recall": recall, "f1": f1})
        f1_values.append(f1)
    return {
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": rows,
    }


def multilabel_f1(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    tp = (truth & prediction).sum(axis=0)
    fp = ((~truth) & prediction).sum(axis=0)
    fn = (truth & (~prediction)).sum(axis=0)
    per_class = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    return {
        "micro_f1": float(2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn)),
        "macro_f1": float(per_class.mean()),
        "per_class_f1": per_class.tolist(),
    }

