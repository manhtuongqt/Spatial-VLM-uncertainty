# Gazebo → RGB-D → best V2: observation-only demo and pre-MoveIt audit

This is a **single-frame perception demonstration**, not robot manipulation or
an accuracy evaluation. The default command uses the existing Gazebo Fortress
tabletop world and its **fixed overhead RGB-D camera**, with no UR3 spawned.
The optional four-panel dashboard uses a separate generated preview scene with
a **static UR3, SusGrip and wrist RGB-D camera**. Neither mode starts a robot
controller or MoveIt action, and neither sends a motion command. An image-space
point is not a safe grasp point.

Important domain boundary: best V2 and its calibrator were assessed with
wrist-camera Test-IID images. The overhead view in this live demo differs from
that evaluated view. Its displayed risk is the frozen model's numerical output,
**not validated calibration on overhead images**. Do not quote this demo as
Test-IID accuracy or robot task success.

## Run

From the workspace root:

```bash
cd /home/dhcn/ur_ws/src/myproject
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 new/demo_gazebo/run_demo.py
```

Add `--gui` to see the Gazebo window if `DISPLAY` is available. Use `--prompt`
to select another object or relation in the currently rendered scene. The
default asks for the banana. The runner uses the existing
`ur3_pick_place_uq_occlusion_v2.sdf` world, captures synchronized `rgb8` and
`32FC1` camera messages, applies the same relative-depth transform as
Test-IID, extracts frozen RoboRefer tower features, runs the frozen V2 epoch-13
checkpoint, and applies the frozen independent calibrator and policy.

For an oblique view that makes near/far occlusion easier to inspect, capture
just the synchronized RGB-D pair:

```bash
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 new/demo_gazebo/run_demo.py --camera oblique --capture-only
```

The oblique camera is outside the evaluated wrist-camera distribution. For a
layout preview from the opposite side of the table with a stationary UR3 at
its original base position:

```bash
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 new/demo_gazebo/run_demo.py --camera opposite --with-static-robot --capture-only
```

For the separate tabletop layout preview requested by the user (YCB objects
clustered on the robot's left in world coordinates, two boxes beside the bin,
and a fixed left-oblique camera), capture either view with:

```bash
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 new/demo_gazebo/run_demo.py --camera wrist --with-static-robot --demo-layout --capture-only
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 new/demo_gazebo/run_demo.py --camera left_oblique --with-static-robot --demo-layout --capture-only
```

These alternate views are capture-only, not Test-IID evaluations. The static
preview uses the real UR3, SusGrip 2F and D435i meshes/mount, but **bakes a
preview-only joint pose** read directly from the locked Test-IID capture plan
(`camera_v2_1_relation`: `[1.7315, -2.0273, 1.2428, -1.5363, -1.5708,
-1.9601]`). The robot base remains `(0, 0, 0)`; no controller or movement command is used. The
world and positions for this preview are generated inside its own run directory;
the source world and evaluation data are unchanged. `show_two_cameras.py`
subscribes to the left-oblique and wrist RGB topics and never commands a robot.

## Four-panel best-V2 demo

The runnable observation-only dashboard captures one timestamp-aligned
left-oblique overview and wrist RGB-D pair from the generated preview world,
then runs the frozen best-V2 checkpoint and policy on **wrist RGB-D only**:

```bash
cd /home/dhcn/ur_ws/src/myproject
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 new/demo_gazebo/run_demo.py --dashboard
```

To open the Gazebo GUI and keep the generated four-panel image on screen until
you close it, use `--show-dashboard` instead. `--prompt` can override the
preselected apple-left-of-soup-can query. The dashboard is a
**single synchronized observation**, not a video or continuous inference loop.
`dashboard.png`, `prediction.json`, `decision.json`, and `run_manifest.json`
are saved in the printed run directory. The cyan cross is the model's candidate
MAP pixel even if the policy does not choose `EXECUTE`. The true apple location
is not injected into the model or panel. Source bars are raw scores; the risk
on this altered pose/layout has not been validated against Test-IID. No robot
trajectory, grasp, or pick-success result is produced. The original source
Gazebo world and URDF files are not modified.

The runner prints the unique output directory under `runs/`. It contains:

- `capture/rgb_original.png`, `capture/depth_metric.npy`, and the exact RGB/depth
  model inputs;
- `prediction.json`, `decision.json`, `demo_panel.png`, `run_manifest.json`;
- `gazebo.log`, `bridge.log`, `capture.log`, `inference.log`.

The cyan × on `demo_panel.png` is a **candidate MAP pixel**, even when the
policy selects an action other than `EXECUTE`. It is never sent to a robot.
The saved prediction has no ground-truth mask, evaluator label, or success flag.

## Live four-panel monitor

This continuously displays the two **live Gazebo** RGB streams, while frozen
V2 inference runs asynchronously on timestamp-matched wrist RGB-D snapshots.
The heatmap panel deliberately shows the **captured inference frame**, not an
old point overlaid on a newer moving frame. The right panel shows the V2
decision and, if a separate controller is running, its ROS episode events.
The view refreshes continuously; V2 does **not** infer at camera frame rate.

```bash
cd /home/dhcn/ur_ws/src/myproject
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 \
  new/demo_gazebo/run_demo.py --live-dashboard --record-video
```

Press `q` or Esc in the four-panel window to finish. `last_dashboard.png`,
optional `live_dashboard.mp4`, and per-inference RGB, depth, prediction and
decision are saved under the printed `runs/.../live/` directory. The scene is
still the generated **static preview**, and no target selection or motion
command is published. The source Gazebo world and URDF are untouched.

This is **not a best-V2 pick-and-place demo**. The earlier close-up pose (now
available only as `--view-pose closeup_legacy` in the world builder) gave
`ASK_USER` and often selected the wrong object. The current selective policy
therefore blocks any honest V2
handoff to the UR3. The repository has a separate older RoboRefer/five-step
pick-and-place launch, but running it would demonstrate that older controller,
not a V2-controlled grasp. A correct V2 RGB-D target, metric 3-D grasp
generation, reachability/collision checks, and a robot-action handoff must be
verified before calling the integrated demo a successful pick-and-place.

## Five-frame Test-IID-pose audit before MoveIt

```bash
cd /home/dhcn/ur_ws/src/myproject
PYTHONNOUSERSITE=1 .conda-roborefer/bin/python3.10 \
  new/demo_gazebo/run_iid_pose_audit.py --frames 5
```

This creates a *new generated world* with the Test-IID wrist joint pose while
leaving the source world and URDF untouched. It uses an apple-left-of-soup-can
prompt with one relation, matching the structure of Test-IID prompts. Five
distinct ROS-timestamped wrist RGB, registered depth, and semantic-label
messages are captured. The sequence is deliberately separated:

1. Frozen V2 runs on RGB and relative depth, then prediction and decision
   hashes are locked in `prediction_lock.json`.
2. A separate, conservative **RGB-D-only** apple check verifies that the
   predicted point lies inside a unique sufficiently large, round red
   component and has compatible metric depth. It is locked in
   `independent_rgbd_check_lock.json` without opening semantic labels.
3. Only then does the evaluator open the Gazebo semantic-label frames (label
   26 for apple), measure whether the point and RGB-D check truly hit the
   target, and render `audit_montage.png` and `audit_summary.json`.

The RGB-D checker is apple-/scene-specific and **not independently validated
for arbitrary objects or camera views**. The Gazebo labels are a simulation
oracle used only for post-prediction evaluation, never for V2 input or robot
control. Matching the camera pose does not make the custom object layout an
official Test-IID sample, and the frozen risk calibration has not been
validated on this layout. Even a passing simulation audit is **not** a MoveIt
handoff: this script launches neither MoveIt nor a robot controller and sends
no motion command.

If capture or inference fails, the runner retains its run directory and logs
for diagnosis; it stops only the Gazebo/bridge processes that it started.
