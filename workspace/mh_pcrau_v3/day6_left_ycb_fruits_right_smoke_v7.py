#!/usr/bin/env python3
"""Generate a compact layout smoke with all eligible fruits on image-right."""

from pathlib import Path

from workspace.mh_pcrau_v3 import day6_left_ycb_cluster_smoke_v6 as base


base.SOURCE = base.ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_cluster_smoke_v6"
base.OUT = base.ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_fruits_right_smoke_v7"
base.PROTOCOL = "mh_pcrau_v3_left_ycb_fruits_right_smoke_v7"

# Smaller Y projects to image-right in this calibrated top-camera view.  The
# five eligible fruits occupy y=0.30..0.44; boxes stay central and non-fruit
# clutter moves to y=0.66..0.88.  Lemon remains outside the protocol.
base.POSES = {
    "ycb_apple": [-0.42, 0.34, 0.0],
    "ycb_orange": [-0.30, 0.30, 0.0],
    "mango": [-0.18, 0.34, 0.0],
    "ycb_pear": [-0.08, 0.38, 0.0],
    "ycb_plum": [-0.30, 0.44, 0.0],
    "ycb_bleach_cleanser": [-0.44, 0.52, 0.0],
    "ycb_cracker_box": [-0.29, 0.59, 0.0],
    "ycb_sugar_box": [-0.16, 0.55, 0.0],
    "ycb_tomato_soup_can": [-0.44, 0.66, 0.0],
    "ycb_mustard_bottle": [-0.32, 0.72, 0.0],
    "ycb_tuna_fish_can": [-0.19, 0.69, 0.0],
    "ycb_power_drill": [-0.46, 0.88, 0.0],
    "ycb_banana": [-0.24, 0.86, 0.0],
    "ycb_mug": [-0.08, 0.78, 0.0],
    "ycb_lemon": [5.34, 3.0, 0.0],
}


if __name__ == "__main__":
    base.main()
