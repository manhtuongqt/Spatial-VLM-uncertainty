#!/usr/bin/env python3
"""Read-only, worst-case interval certificate; never generates a v5 scene."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V5_DATA_DESIGN_LOCK.json"
EXPECTED_SHA256 = "64a9747052947103bf48a896a9ad38b15c14614146f49148a634d6685a36aa65"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v5/PRE_RENDER_INTERVAL_CERTIFICATE.json"
FREEZE = ROOT / "protocol/GAZEBO_CALIBRATION_V5_PRE_RENDER_CERTIFICATE_LOCK.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lock() -> dict:
    if sha256(LOCK) != EXPECTED_SHA256:
        raise RuntimeError("v5 data-design lock hash drift")
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    for group in ("predecessor", "frozen_sources_sha256"):
        for path, expected in data[group].items():
            if sha256(ROOT / path) != expected:
                raise RuntimeError(f"frozen source drift: {path}")
    return data


def rotation(q: list[float]) -> list[list[float]]:
    x, y, z, w = q
    norm = math.sqrt(sum(value*value for value in q))
    x, y, z, w = (value/norm for value in (x, y, z, w))
    return [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]


def matmul(matrix: list[list[float]], vector: tuple[float, float, float]) -> list[float]:
    return [sum(row[i]*vector[i] for i in range(3)) for row in matrix]


def inverse(u: float, v: float, z: float, origin: list[float], orient: list[list[float]], k: dict) -> tuple[float, float, float]:
    ray = matmul(orient, ((u*640-k["cx"])/k["fx"], (v*480-k["cy"])/k["fy"], 1.0))
    if ray[2] >= -1e-12:
        raise RuntimeError("locked camera ray points away from the table plane")
    t = (z-origin[2])/ray[2]
    if not (math.isfinite(t) and 0.1 <= t <= 2.0):
        raise RuntimeError("camera ray intersects outside inherited valid depth range")
    return origin[0]+t*ray[0], origin[1]+t*ray[1], t


def forward(xyz: tuple[float, float, float], origin: list[float], orient: list[list[float]], k: dict) -> tuple[float, float, float]:
    diff = tuple(xyz[i]-origin[i] for i in range(3))
    cam = tuple(sum(orient[j][i]*diff[j] for j in range(3)) for i in range(3))
    if cam[2] <= 0:
        raise RuntimeError("forward projection behind camera")
    return (k["fx"]*cam[0]/cam[2]+k["cx"])/640, (k["fy"]*cam[1]/cam[2]+k["cy"])/480, cam[2]


def interval_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, a[0]-b[1], b[0]-a[1])


def rectangle_distance(a: dict, b: dict) -> float:
    return math.hypot(interval_gap(a["x"], b["x"]), interval_gap(a["y"], b["y"]))


def canonical_json(path: Path, item: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def compute() -> dict:
    lock = load_lock()
    paths = lock["frozen_sources_sha256"]
    world = ET.parse(ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf")
    radius = {}
    for name in ("ycb_apple", "ycb_orange", "mango"):
        node = world.find(f".//model[@name='{name}']/link/collision/geometry/sphere/radius")
        if node is None:
            raise RuntimeError(f"missing SDF collision radius for {name}")
        radius[name] = float(node.text)
    table_node = world.find(".//model[@name='work_table']/link/collision/geometry/box/size")
    table_pose = world.find(".//model[@name='work_table']/pose")
    panel_node = world.find(".//model[@name='uq_neutral_occluder']/link/visual/geometry/box/size")
    if any(x is None for x in (table_node, table_pose, panel_node)):
        raise RuntimeError("frozen world lacks table or panel dimensions")
    table_size = [float(x) for x in table_node.text.split()]
    table_pos = [float(x) for x in table_pose.text.split()]
    panel_size = [float(x) for x in panel_node.text.split()]
    table = {"x": (table_pos[0]-table_size[0]/2, table_pos[0]+table_size[0]/2),
             "y": (table_pos[1]-table_size[1]/2, table_pos[1]+table_size[1]/2)}
    if table != {"x": (-0.8, 0.8), "y": (-0.6500000000000001, 1.55)}:
        # Check with tolerance to avoid incidental binary float representation.
        if any(abs(table[axis][i]-bound) > 1e-9 for axis, bounds in
               {"x":(-0.8,0.8),"y":(-0.65,1.55)}.items() for i,bound in enumerate(bounds)):
            raise RuntimeError("locked work table bounds mismatch")
    tf_path = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01/gazebo_calibration_v3_100/input/tf_snapshot.json"
    tf = json.loads(tf_path.read_text(encoding="utf-8"))["camera_color_optical_frame"]
    if tf["parent_frame"] != "base_link" or tf["child_frame"] != "camera_color_optical_frame":
        raise RuntimeError("fixed camera TF frame mismatch")
    origin, orient = tf["position"], rotation(tf["orientation_xyzw"])
    parent = json.loads((ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json").read_text(encoding="utf-8"))
    k = parent["camera_and_geometry"]["camera"]["intrinsics"]
    scenes = yaml.safe_load((ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml").read_text(encoding="utf-8"))
    ann = yaml.safe_load((ROOT / "ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml").read_text(encoding="utf-8"))["scenes"]
    ie_rows = [row for row in scenes["scenes"] if ann[row["scene_id"]]["state"]=="INSUFFICIENT_EVIDENCE"]
    targets = {tuple(row["poses"]["mango"]) for row in ie_rows}
    panels = {tuple(row["poses"]["uq_neutral_occluder"]) for row in ie_rows}
    if len(ie_rows)!=32 or len(targets)!=1 or len(panels)!=1:
        raise RuntimeError("the locked IE target/panel pair differs by template")
    target, panel_pose = next(iter(targets)), next(iter(panels))
    heights = {name: scenes["objects"][name]["z"] for name in radius}
    target_uv = forward((target[0],target[1],heights["mango"]),origin,orient,k)
    panel_uv = forward((panel_pose[0],panel_pose[1],scenes["objects"]["uq_neutral_occluder"]["z"]),origin,orient,k)
    panel_angle=panel_pose[2]
    panel_half = (abs(math.cos(panel_angle))*panel_size[0]+abs(math.sin(panel_angle))*panel_size[1])/2
    panel_box={"x":(panel_pose[0]-panel_half,panel_pose[0]+panel_half),
               "y":(panel_pose[1]-panel_half,panel_pose[1]+panel_half)}
    target_box={"x":(target[0],target[0]),"y":(target[1],target[1])}
    margin=0.010
    rows=[]
    for relation, anchors in lock["exact_geometry_revision"]["relation_aware_context_projected_center_targets_normalized"].items():
        candidates=[]
        reasons=[]
        for name, anchor in zip(("ycb_apple","ycb_orange"),anchors):
            urange=(anchor[0]-0.008,anchor[0]+0.008)
            vrange=(anchor[1]-0.010,anchor[1]+0.010)
            corners=[(u,v) for u in urange for v in vrange]
            poses=[inverse(u,v,heights[name],origin,orient,k) for u,v in corners]
            box={"x":(min(p[0] for p in poses),max(p[0] for p in poses)),
                 "y":(min(p[1] for p in poses),max(p[1] for p in poses))}
            # On a rectangle a positive-denominator linear-fractional function
            # attains its extrema at a vertex. The camera-to-plane inverse
            # projection has exactly this form for each world coordinate.
            errors=[max(abs(forward((pose[0],pose[1],heights[name]),origin,orient,k)[i]-uv[i]) for i in (0,1))
                    for pose,uv in zip(poses,corners)]
            if max(errors)>1e-10: raise RuntimeError("camera inverse/forward corner regression")
            table_margins={axis:min(box[axis][0]-table[axis][0],table[axis][1]-box[axis][1])-radius[name]
                           for axis in ("x","y")}
            target_clearance=rectangle_distance(box,target_box)-radius[name]-radius["mango"]
            panel_clearance=rectangle_distance(box,panel_box)-radius[name]
            target_img=math.hypot(interval_gap(urange,(target_uv[0],target_uv[0])),
                                  interval_gap(vrange,(target_uv[1],target_uv[1])))
            panel_img=math.hypot(interval_gap(urange,(panel_uv[0],panel_uv[0])),
                                 interval_gap(vrange,(panel_uv[1],panel_uv[1])))
            conditions={
                "inside_image_center_guard": 0.08<=urange[0]<=urange[1]<=0.92 and 0.08<=vrange[0]<=vrange[1]<=0.92,
                "table_margin_10mm": all(value>=margin for value in table_margins.values()),
                "target_sphere_margin_10mm":target_clearance>=margin,
                "panel_conservative_box_margin_10mm":panel_clearance>=margin,
                "target_projected_center_gap_0_08":target_img>=0.08,
                "panel_projected_center_gap_0_08":panel_img>=0.08,
            }
            candidates.append({"name":name,"image_u_interval":urange,"image_v_interval":vrange,
                               "xy_interval_m":box,"collision_radius_m":radius[name],
                               "table_edge_clearance_min_m":min(table_margins.values()),
                               "target_clearance_lower_bound_m":target_clearance,
                               "panel_clearance_lower_bound_m":panel_clearance,
                               "target_image_gap_lower_bound":target_img,
                               "panel_image_gap_lower_bound":panel_img,"checks":conditions})
            reasons.extend(f"{name}:{key}" for key,value in conditions.items() if not value)
        a,b=candidates
        pair_clearance=rectangle_distance(a["xy_interval_m"],b["xy_interval_m"])-radius["ycb_apple"]-radius["ycb_orange"]
        pair_image=math.hypot(interval_gap(a["image_u_interval"],b["image_u_interval"]),
                              interval_gap(a["image_v_interval"],b["image_v_interval"]))
        relation_u=target_uv[0]
        relation_ok={
            "leftmost":relation_u<min(a["image_u_interval"][0],b["image_u_interval"][0]),
            "rightmost":relation_u>max(a["image_u_interval"][1],b["image_u_interval"][1]),
            "second_from_left": a["image_u_interval"][1]<relation_u<b["image_u_interval"][0],
            "second_from_right":a["image_u_interval"][1]<relation_u<b["image_u_interval"][0],
        }[relation]
        pair_conditions={"candidate_spheres_separated_by_10mm":pair_clearance>=margin,
                         "candidate_image_centers_separated_by_0_12":pair_image>=0.12,
                         "target_relation_order":relation_ok}
        reasons.extend(name for name,passed in pair_conditions.items() if not passed)
        rows.append({"relation":relation,"candidate_intervals":candidates,
                     "candidate_pair_clearance_lower_bound_m":pair_clearance,
                     "candidate_pair_image_distance_lower_bound":pair_image,
                     "pair_and_relation_checks":pair_conditions,
                     "failures":reasons})
    target_check=(-0.2<=target_uv[0]<1.2 and -0.2<=target_uv[1]<1.2 and 0.1<=target_uv[2]<=2.0)
    checks={"lock_and_12_upstream_hashes_verified":True,
            "world_sdf_collision_radii_match_preregistered_values": all(abs(radius[n]-v)<1e-9 for n,v in lock["observed_pre_render_failure"]["collision_radii_m"].items()),
            "pilot_4x4x2_and_calibration_4x4x8_preregistered":lock["population_and_independence"]["pilot"]["family_count"]==32 and lock["population_and_independence"]["calibration"]["family_count"]==128,
            "same_target_panel_pair_all_v3_ie_templates":len(targets)==len(panels)==1,
            "target_projected_center_in_inherited_gate":target_check,
            "all_jitter_envelope_relation_geometry_guards_proved":all(not row["failures"] for row in rows),
            "no_v5_scene_or_manifest_generated":all(not path.exists() for suffix in ("gazebo_calibration_v5","gazebo_calibration_v5_geometry_pilot") for path in (
                ROOT/f"ur3/ur3_perception/config/{suffix}_scenes.yaml",
                ROOT/f"protocol/{suffix}_family_manifest.jsonl")),
            "no_v5_capture_or_dataset": not (ROOT/"datasets/Gazebo_calibration_v5").exists() and all(not (ROOT/f"results/spatial_vlm_refspatial_v1/{suffix}/capture_attempt_01").exists() for suffix in ("gazebo_calibration_v5","gazebo_calibration_v5_geometry_pilot")),
            "test_and_robot_sealed": lock["policy"]["test_iid_ood_access"] is False and lock["policy"]["robot_access"] is False}
    return {"schema_version":1,"protocol_id":"gazebo_calibration_v5",
            "status":"PASS" if all(checks.values()) else "BLOCKED_PRE_RENDER_GEOMETRY",
            "checked_at_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"pre_render_analytic_jitter_interval_only_no_scene_generation",
            "proof_method":"Inverse camera ray to fixed support z is affine-over-affine with denominator strictly signed across the jitter rectangle; evaluate four vertices to enclose world x/y; subtract SDF collision radii from minimum AABB separations (a conservative lower bound). Panel is enclosed by its rotated square AABB. All inequalities must hold throughout every possible h-derived jitter; neither sampled seeds nor RGB are read.",
            "checks":checks,"candidate_collision_radii_m":radius,"table_xy_bounds_m":table,
            "panel_enclosing_xy_box_m":panel_box,"target_projected_xy_depth":target_uv,
            "panel_projected_xy_depth":panel_uv,"minimum_physical_margin_m":margin,
            "relations":rows,"data_design_lock_sha256":sha256(LOCK),
            "verification_script_sha256":sha256(Path(__file__)),
            "sealed_test_iid_ood_access":False,"robot_access":False,
            "pilot_or_calibration_captured":False,"model_inference_run":False,
            "calibrator_fit_call_count":0,"scientific_decision_exists":False}


def main() -> None:
    if RESULT.exists() or FREEZE.exists():
        raise FileExistsError("refusing to rerun or overwrite a v5 pre-render certificate")
    report=compute()
    canonical_json(RESULT,report)
    sources=(LOCK,ROOT/"ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
             ROOT/"ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml",
             ROOT/"ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml",
             ROOT/"results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01/gazebo_calibration_v3_100/input/tf_snapshot.json",
             Path(__file__).resolve(),RESULT)
    lock={"schema_version":1,"protocol_id":"gazebo_calibration_v5",
          "status":"PRE_RENDER_INTERVAL_CERTIFICATE_FROZEN_PASS" if report["status"]=="PASS" else "PRE_RENDER_INTERVAL_CERTIFICATE_FROZEN_BLOCKED",
          "locked_at_utc":datetime.now(timezone.utc).isoformat(),
          "source_artifact_sha256":{str(path.relative_to(ROOT)):sha256(path) for path in sources},
          "scene_generation_authorized":report["status"]=="PASS",
          "capture_authorized":False,"scientific_result_exists":False,
          "test_iid_ood_access":False,"robot_access":False}
    canonical_json(FREEZE,lock)
    print(json.dumps({"status":report["status"],"checks":report["checks"],
                      "failing_relations":{row["relation"]:row["failures"] for row in report["relations"] if row["failures"]},
                      "certificate_sha256":sha256(RESULT),"freeze_sha256":sha256(FREEZE)},indent=2,sort_keys=True))
    if report["status"]!="PASS":raise SystemExit(2)


if __name__=="__main__":main()
