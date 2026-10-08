#!/usr/bin/env python3
"""Create the grounded successor to the immutable left-YCB v3 world.

The extra YCB meshes added in v3 use asset coordinates whose lower surface is
near local z=0.  Their collision primitives, however, are centred at the model
origin, so the model pose must remain at the collision support height.  v3 did
not compensate the visual origin and therefore rendered those meshes floating
above the table.  This append-only revision translates each visual so its mesh
lower surface is exactly at world z=0 when reset by the capture configuration.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v3/left_ycb_top30.sdf"
OUT_DIR = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v4_grounded"
OUTPUT = OUT_DIR / "left_ycb_top30_grounded.sdf"

# pose = (-mesh x-centre, -mesh y-centre, -configured support z - mesh z-min)
# Bounds were measured directly from the vendored OBJ vertex records.
VISUAL_POSES = {
    "ycb_lemon": (0.0105805, -0.0216475, -0.0337740),
    "ycb_pear": (0.0333135, -0.0180080, -0.0398410),
    "ycb_plum": (0.0077990, -0.0194380, -0.0277000),
    "ycb_bleach_cleanser": (0.0216510, -0.0116920, -0.1195240),
    "ycb_mug": (0.0088535, -0.0173415, -0.0444300),
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"append-only output exists: {OUTPUT}")
    text = SOURCE.read_text(encoding="utf-8")
    replacements = {}
    for model, (x_value, y_value, z_value) in VISUAL_POSES.items():
        model_start = text.index(f'<model name="{model}">')
        model_end = text.index("</model>", model_start)
        block = text[model_start:model_end]
        old = '<visual name="ycb_visual"><geometry>'
        if block.count(old) != 1:
            raise RuntimeError(f"unexpected visual structure for {model}")
        new = (
            '<visual name="ycb_visual">'
            f'<pose>{x_value:.7f} {y_value:.7f} {z_value:.7f} 0 0 0</pose>'
            '<geometry>'
        )
        replacements[model] = (old, new)
        patched = block.replace(old, new, 1)
        text = text[:model_start] + patched + text[model_end:]

    provenance = """\n    <!-- v4 grounded visual-origin correction. The v3 world remains immutable. -->\n"""
    text = text.replace("  </world>", provenance + "  </world>", 1)
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(text, encoding="utf-8")
    (OUT_DIR / "GROUNDING_FIX.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "GROUNDED_WORLD_READY",
                "source_world": str(SOURCE.relative_to(ROOT)),
                "output_world": str(OUTPUT.relative_to(ROOT)),
                "corrected_visual_pose_xyz": {
                    name: list(values) for name, values in VISUAL_POSES.items()
                },
                "excluded_from_next_protocol": ["ycb_lemon"],
                "reason": "v3 omitted mesh visual-origin compensation and rendered five added YCB meshes above the table",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
