# UR3 spatial representation and reproducible dataset

This package records the fixed eye-in-hand `VIEW_POSE` protocol described in
the research plan.  All canonical object poses and dimensions use `base_link`.
Image relations (`left_of`, `right_of`, `above`, `below`) use the frozen camera
image.  `front_of` and `behind` use optical depth; `inside` and `near` use metric
3D AABBs.

Gazebo ground truth is written only to dataset/evaluation outputs.  It is never
published as a manipulation target and is not part of the control path.

Run one visible episode:

```bash
ros2 launch ur3_spatial_dataset spatial_dataset_demo.launch.py \
  run_id:=run_demo random_seed:=23 episode_count:=1
```

Every episode contains a rosbag with RGB, depth, camera info, TF, joint state,
perception, oracle evaluation, spatial scene, and episode events.  A synchronized
snapshot is also exported to portable PNG, NPY, and JSON files.

Validate an offline run:

```bash
ros2 run ur3_spatial_dataset validate_dataset /tmp/ur3_spatial_dataset/run_demo
```

Never reuse a `run_id`; run directories are immutable to prevent accidental
mixing of seeds or configuration versions.  `manifest.json` stores hashes for
the object registry, Gazebo world, and control configuration plus the exact
reproduction command.  That command preserves the seed and configuration but
adds `_repro` to the output ID so the original evidence remains immutable.
