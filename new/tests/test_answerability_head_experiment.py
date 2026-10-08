from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def experiment():
    spec = spec_from_file_location("answerability_experiment", Path(__file__).parents[1] / "scripts/experiment_answerability_head.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_rejects_recall_gain_with_more_false_found():
    module = experiment()
    baseline = module.answer_metrics([0, 0, 1, 2, 3], [0, 2, 1, 2, 3])
    increased_false_found = module.answer_metrics([0, 0, 1, 2, 3], [0, 0, 1, 0, 3])
    assert increased_false_found["per_class"][0]["recall"] > baseline["per_class"][0]["recall"]
    assert not module.eligible(increased_false_found, baseline)


def test_selection_accepts_corrected_found_without_losing_other_classes():
    module = experiment()
    baseline = module.answer_metrics([0, 0, 1, 2, 3], [0, 2, 1, 2, 3])
    corrected = module.answer_metrics([0, 0, 1, 2, 3], [0, 0, 1, 2, 3])
    assert module.eligible(corrected, baseline)
    assert module.selection_key(corrected) > module.selection_key(baseline)
    assert corrected["false_found"] == 0 and corrected["missed_found"] == 0
