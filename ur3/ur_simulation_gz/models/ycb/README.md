# YCB Gazebo subset

This directory contains a lightweight subset of the
[`ai-habitat/ycb`](https://huggingface.co/datasets/ai-habitat/ycb) dataset for
the UR3 Gazebo demo:

- 003 cracker box
- 004 sugar box
- 005 tomato soup can
- 006 mustard bottle
- 007 tuna fish can
- 011 banana
- 013 apple
- 014 lemon
- 016 pear
- 017 orange
- 018 plum
- 021 bleach cleanser
- 025 mug
- 035 power drill

The optimized `textured.glb` files preserve the downloaded source subset.
Ignition Gazebo 6 cannot render GLB directly, so each object also has a
generated uniquely named OBJ, MTL, and PNG files for each object. Unique
resource and material names are required because Ignition Gazebo 6 otherwise
may reuse the first loaded YCB texture for later objects. Gazebo uses simple
SDF collision primitives sized from the GLB position bounds. This avoids the
cost of loading convex-decomposition collision meshes. Generated textures are
downsampled from 4096 px to 1024 px to limit disk and GPU memory use. Embedded
raw Basis textures are decoded with the official Basis Universal CLI; KTX2
payloads are supported through the Khronos KTX CLI.

The legacy YCB derivatives in this repository use the converter's default
`--uv-v-mode flip`. The raw Basis-texture assets added for inventory V2
(`007_tuna_fish_can`, `014_lemon`, `016_pear`, `018_plum`, and `025_mug`)
must instead be converted with `--uv-v-mode preserve`. Their GLB UV
coordinates already match the decoded Basis texture orientation; flipping V
again maps much of the mesh to unrelated dark atlas regions. This distinction
is locked by the runtime color-fix audit in
`results/ycb_inventory_v2_color_fix_20260821_135403`.

`config/ycb_inventory_v2.json` pins the source revision and GLB hashes for the
nine classes used by the V2 asset-qualification gallery. The gallery creates
three independently named and labelled instances of apple, orange and banana
while sharing their immutable render asset. It also contains one instance of
lemon, pear, plum, tuna fish can, mug and bleach cleanser. This gallery is not
part of the locked WP2 dataset; a dataset-expansion contract is required before
new captures are admitted into train/dev/calibration/test.

The UR3 main-world integration intentionally omits bleach cleanser and removes
the original `cup` and `kettle` from its derived world, following the scene
cleanup policy in `config/ycb_inventory_v2.json`. The source gallery and source
assets remain available for provenance; the original `ur3_pick_place.sdf` is
not overwritten.

The source assets and this subset are licensed under CC BY 4.0. See
`LICENSE.txt`. The Gazebo SDF integration re-centers the visual origins and
adds primitive collision/inertial properties; the texture and render meshes
are otherwise unchanged. OBJ/PNG derivatives are generated from the pinned
`textured.glb` sources using `scripts/convert_ycb_glb_to_obj.py`.
