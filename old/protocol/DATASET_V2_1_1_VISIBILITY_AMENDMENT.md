# Dataset V2.1.1 visibility amendment

Status: locked before restarting official development capture.

The first V2.1 `batch_001` attempt stopped at `v21dev_family_000113` because
semantic label `4` (the yellow-cube anchor) was absent in the clean frame. An
identical-seed resume reproduced the failure. The candidate was inside the
camera field of view and relation-valid, but a nearer mustard bottle covered it
in the image plane.

V2.1.1 adds one static, deterministic candidate predicate: reject a layout when
the calibrated projection of a nearer asset conservatively covers the core of
any required visible asset. The predicate uses only asset dimensions, locked
camera intrinsics/extrinsics and candidate poses; it never reads captured data.
It is applied uniformly to all 400 families and to clean/occlusion layouts.

To avoid post-observation sample repair, V2.1.1 uses a new protocol namespace,
master seed, family/capture IDs, instructions, layouts and output root. The 128
V2.1 captures remain engineering diagnostic evidence and are excluded from the
official train/dev dataset. Baseline launch files, URDF, controllers and Gazebo
world are unchanged.
