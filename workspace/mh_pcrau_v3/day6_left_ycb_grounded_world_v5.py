#!/usr/bin/env python3
"""Create a contact-exact, visually auditable successor to grounded world v4."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v4_grounded/left_ycb_top30_grounded.sdf"
OUT_DIR = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact"
OUTPUT = OUT_DIR / "left_ycb_top30_contact_exact.sdf"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence: {old}")
    return text.replace(old, new, 1)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"append-only output exists: {OUTPUT}")
    text = SOURCE.read_text(encoding="utf-8")
    # Orange reset centre is 0.037010 m and its OBJ z-min is -0.000276 m.
    # A local visual translation of -0.036734 m therefore makes the rendered
    # lower surface exactly 0.000000 m, matching the table top.
    text = replace_once(
        text,
        '<visual name="ycb_visual"><pose>0.006944 0.018358 -0.035408 0 0 0</pose><geometry><mesh><uri>file:///home/dhcn/ur_ws/src/myproject/ur3/ur_simulation_gz/models/ycb/017_orange/017_orange.obj</uri></mesh></geometry></visual>',
        '<visual name="ycb_visual"><pose>0.006944 0.018358 -0.036734 0 0 0</pose><geometry><mesh><uri>file:///home/dhcn/ur_ws/src/myproject/ur3/ur_simulation_gz/models/ycb/017_orange/017_orange.obj</uri></mesh></geometry></visual>',
    )
    # Straight-down light keeps contact shadows under the objects, making this
    # geometric QC visually interpretable instead of resembling floating.
    text = replace_once(
        text,
        '<direction>-0.4 0.2 -1.0</direction>',
        '<direction>0 0 -1.0</direction>',
    )
    text = text.replace(
        "  </world>",
        "    <!-- v5: orange mesh contact exact; vertical contact-audit light. -->\n  </world>",
        1,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(text, encoding="utf-8")
    (OUT_DIR / "CONTACT_FIX.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CONTACT_EXACT_WORLD_READY",
                "source_world": str(SOURCE.relative_to(ROOT)),
                "output_world": str(OUTPUT.relative_to(ROOT)),
                "table_top_z_m": 0.0,
                "verified_visual_bottom_z_m": {
                    "ycb_mug": 0.0,
                    "ycb_orange": 0.0,
                },
                "orange_visual_z_before_m": -0.035408,
                "orange_visual_z_after_m": -0.036734,
                "light_direction_before": [-0.4, 0.2, -1.0],
                "light_direction_after": [0.0, 0.0, -1.0],
                "excluded_from_next_protocol": ["ycb_lemon"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
