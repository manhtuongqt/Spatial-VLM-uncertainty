"""Locked gates, family pairing, evaluator events and residual checkpoint checks."""
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from pcrau.anchor_shadow import AnchorShadow
from pcrau.anchor_shadow_experiment import ShadowPilotRunner, state_digest
from pcrau.anchor_shadow_pilot import aggregate, family_bootstrap, locked_gates, pair_rows, score_predictions, spatial_summary
from pcrau.model import PCRAUTargetV2
from pcrau.utils import load_config


def rows_fixture(m1_hits=49, m2_hits=47):
    rows, visible = [], 0
    for f in range(16):
        for j, variant in enumerate(("clean", "relation_counterfactual", "depth_corruption", "occlusion_view_counterfactual")):
            empty = f == 15 and j < 3
            models = {}
            for arm, hits in (("M0", 44), ("M1", m1_hits), ("M2", m2_hits)):
                hit = None if empty else visible < hits
                models[arm] = {"anchor_hit": hit, "anchor_cell_overlap": bool(hit), "target_hit": True,
                    "found_both_hit": bool(hit) if j == 0 else None, "sigmoid_max": .8,
                    "anchor_inside_target": False, "map_grid_xy": [j, 0]}
            rows.append({"family_id": str(f), "sample_id": f"{f}_{variant}", "split": "dev", "variant": variant,
                "anchor_pixels": 0 if empty else 1, "truth": "FOUND" if j == 0 else "ABSENT",
                "feature_key": str(f), "parsed": {"anchors": [{"text": "cube" if j == 0 else "apple"}]}, "models": models})
            visible += int(not empty)
    return rows


def test_full_pixel_hit_and_empty_not_counted_as_visible_miss():
    logits = np.zeros((24, 32)); logits[0, 0] = 5
    summary = spatial_summary(logits)
    assert summary["map_pixel_xy"] == [10, 10]
    mask = np.zeros((480, 640), dtype=bool); mask[10, 11] = True
    label = {"anchor_full": mask, "target_full": np.zeros_like(mask), "anchor_small": np.ones((24, 32)),
        "truth": "ABSENT", "variant": "clean", "anchor_ids": ["a"], "anchor_mask_path": "a", "target_mask_path": "t", "rgb_path": "r"}
    pred = {"sample_id": "x", "family_id": "f", "split": "dev", "feature_key": "f", "parsed": {"anchors": [{"text": "cube"}]},
        **{arm: {"anchor": summary} for arm in ("M0", "M1", "M2")}}
    pred["M0"]["target"] = summary
    row = score_predictions([pred], {"x": label})[0]
    assert row["models"]["M1"]["anchor_hit"] is False
    assert row["models"]["M1"]["anchor_cell_overlap"] is True
    label["anchor_full"] = np.zeros_like(mask)
    empty = score_predictions([pred], {"x": label})[0]
    assert empty["models"]["M1"]["anchor_hit"] is None
    assert aggregate([empty])["visible"] == 0


def test_missing_guard_checks_each_case_even_when_mean_improves():
    rows = rows_fixture()
    result = locked_gates(rows, True, True)
    assert result["pass"]
    missing = [r for r in rows if not r["anchor_pixels"]]
    missing[0]["models"]["M1"]["sigmoid_max"] = .80001
    for r in missing[1:]: r["models"]["M1"]["sigmoid_max"] = .3
    result = locked_gates(rows, True, True)
    assert not result["gates"]["empty_each_no_increase_1e_6"]
    assert aggregate(rows)["models"]["M1"]["empty_mean_sigmoid_max"] < .8


def test_phrase_gate_requires_nonregression_in_other_metric():
    rows = rows_fixture()
    assert locked_gates(rows, True, True)["gates"]["phrase_specific_vs_M2"]
    # Give M2 more complete swap pairs with the same lower total hit count.
    visible = [r for r in rows if r["anchor_pixels"]]
    chosen = [r for r in visible if int(r["family_id"]) < 14 and r["variant"] in {"clean", "relation_counterfactual"}]
    chosen += [r for r in visible if r not in chosen][:47-len(chosen)]
    chosen_ids = {r["sample_id"] for r in chosen}
    for row in visible:
        row["models"]["M2"]["anchor_hit"] = row["sample_id"] in chosen_ids
    result = locked_gates(rows, True, True)
    assert result["phrase_deltas"]["visible_hits"] == 2 and result["phrase_deltas"]["swap_both_hits"] < 0
    assert not result["gates"]["phrase_specific_vs_M2"]


def test_pair_eligibility_and_family_bootstrap_not_sample_bootstrap():
    rows = rows_fixture()
    assert sum(p["eligible"] for p in pair_rows(rows)) == 15
    rows[0]["feature_key"] = "different"
    assert sum(p["eligible"] for p in pair_rows(rows)) == 14
    base = rows_fixture(m1_hits=61, m2_hits=61)
    for r in base:
        if r["anchor_pixels"]: r["models"]["M0"]["anchor_hit"] = False
    result = family_bootstrap(base, "M1")
    assert result["families"] == 16 and result["anchor_hit_delta"]["percentile_ci95"] == [1., 1.]
    assert result == family_bootstrap(base, "M1")


def test_same_noun_phrase_remains_in_locked_matched_subset():
    rows = rows_fixture()
    rows[1]["parsed"]["anchors"][0]["text"] = "cube"
    pairs = pair_rows(rows)
    assert sum(p["eligible"] for p in pairs) == 15
    assert pairs[0]["eligible"] and not pairs[0]["anchor_phrase_changed"]


def test_optimizer_only_updates_residual_and_safetensors_reload():
    cfg = deepcopy(load_config()); cfg["model"].update({"feature_dim": 8, "grid_height": 2, "grid_width": 3})
    baseline = PCRAUTargetV2(cfg)
    branch = AnchorShadow(baseline)
    runner = ShadowPilotRunner(branch)
    frozen = state_digest(baseline.state_dict()); initial = state_digest(branch.residual.state_dict())
    runner.start_optimizer({"status": "S1_SHADOW_PREFLIGHT_PASS", "technical_gates_pass": True,
        "initialization_sha256": {"M1": initial}, "frozen_model_state_sha256": frozen})
    assert {id(p) for g in runner.optimizer.param_groups for p in g["params"]} == {id(p) for p in branch.residual.parameters()}
    features = torch.randn(4, 128)
    loss = (branch.residual(features)-1).square().mean()
    loss.backward(); runner.optimizer.step()
    assert state_digest(branch.residual.state_dict()) != initial
    assert state_digest(baseline.state_dict()) == frozen and all(p.grad is None for p in baseline.parameters())
    with tempfile.TemporaryDirectory(prefix="pcrau_pilot_unit_") as directory:
        path = Path(directory)/"mlp.safetensors"
        save_file(branch.residual.state_dict(), str(path))
        other = AnchorShadow(baseline)
        other.residual.load_state_dict(load_file(str(path)), strict=True)
        torch.testing.assert_close(branch.residual(features), other.residual(features), rtol=0, atol=0)
