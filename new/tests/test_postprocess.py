from __future__ import annotations

import torch

from pcrau.postprocess import highest_density_region, summarize_heatmap


def test_spatial_summary_maps_grid_to_pixels() -> None:
    logits = torch.full((24, 32), -20.0)
    logits[12, 16] = 20.0
    summary = summarize_heatmap(logits)
    assert summary["map_grid_xy"] == [16, 12]
    assert summary["map_pixel_xy"] == [330, 250]
    assert summary["mode_count"] == 1
    assert 0.0 <= summary["entropy_normalized"] <= 1.0


def test_highest_density_region_reaches_requested_mass() -> None:
    region = highest_density_region([[0.5, 0.3], [0.15, 0.05]], 0.75)
    assert region["grid_cells_xy"] == [[0, 0], [1, 0]]
    assert region["actual_probability_mass"] >= 0.75
