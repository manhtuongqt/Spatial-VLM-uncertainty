#!/usr/bin/env python3
"""Projection-calibrated Day-8 canary/bulk revision 3."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib, json, random, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import workspace.mh_pcrau_v3.day8_development_r2_prepare as base

OUT = base.RAW / "canary_v3"
# Empirical world-v5 projection fitted only from failed canary-v2 semantic masks.
U_X, U_Y, U_C = -389.92823857, 1259.59975910, -91.06046007

def y_for_u(x: float, u: float) -> float:
    return float((u - U_X * x - U_C) / U_Y)

def desired_u(stage: str, relation: str, repeat: int) -> list[float]:
    rng = random.Random(int(hashlib.sha256(f"{base.NAMESPACE}|{stage}|{relation}|{repeat}|projection-v3".encode()).hexdigest()[:16], 16))
    if relation == "leftmost":
        target = rng.uniform(475., 595.); values = [target, target - rng.uniform(65., 90.), target - rng.uniform(145., 175.)]
    elif relation == "rightmost":
        target = rng.uniform(190., 325.); values = [target + rng.uniform(145., 175.), target + rng.uniform(65., 90.), target]
    else:
        target = rng.choice((rng.uniform(300., 405.), rng.uniform(500., 545.)))
        values = [target + rng.uniform(70., 95.), target, target - rng.uniform(70., 95.)]
    return values

def projection_positions(stage: str, state: str, relation: str, repeat: int) -> list[list[float]]:
    rng = random.Random(int(hashlib.sha256(f"{base.NAMESPACE}|{stage}|{state}|{relation}|{repeat}|pose-v3".encode()).hexdigest()[:16], 16))
    us = desired_u(stage, relation, repeat)
    xs = [-.29 + rng.uniform(-.012, .012), -.24 + rng.uniform(-.012, .012), -.19 + rng.uniform(-.012, .012)]
    if state == "AMBIGUOUS":
        ties = base.tie_indices(relation); tie_u = us[base.rank_index(relation)]
        us[ties[0]] = tie_u; us[ties[1]] = tie_u
        xs[ties[0]], xs[ties[1]] = -.30, -.18
    return [[x, y_for_u(x, u), rng.uniform(-.35, .35)] for x, u in zip(xs, us)]

def build_scene(stage: str, state: str, relation: str, repeat: int, geometry: dict):
    scene, ann, index = base.make_scene(stage, state, relation, repeat, geometry)
    design = base.common_design(stage, relation, repeat); fruits = design["fruits"]; target_i = base.rank_index(relation); target = fruits[target_i]
    positions = projection_positions(stage, state, relation, repeat)
    for i, fruit in enumerate(fruits):
        if state == "ABSENT" and i == target_i: scene["poses"].pop(fruit, None)
        else: scene["poses"][fruit] = positions[i]
    covered = target if state == "INSUFFICIENT_EVIDENCE" else design["decoy"]
    if state != "INSUFFICIENT_EVIDENCE": scene["poses"][covered] = [-.24, .38, 0.0]
    else: scene["poses"][target] = [-.24, .38, 0.0]
    occluder = design["occluder"]; cal = geometry["pairs"][occluder][covered]
    tx = -.10 if occluder == "ycb_sugar_box" and covered == "ycb_apple" else -.24
    scene["poses"][covered] = [tx, .38, 0.0]
    scene["poses"][occluder] = [tx - float(cal["forward_separation_m"]), .38 + float(cal["signed_lateral_offset_m"]), float(cal.get("occluder_yaw_rad", 0.0))]
    signature = hashlib.sha256(json.dumps(scene["poses"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scene["layout_signature_sha256"] = signature; ann["layout_signature_sha256"] = signature; index["layout_signature_sha256"] = signature
    return scene, ann, index

def add_lock(folder: Path, reason: str) -> None:
    path = folder / "CAPTURE_SOURCE_LOCK.json"; lock = json.loads(path.read_text()); rel = str(Path(__file__).relative_to(ROOT))
    lock["source_artifact_sha256"][rel] = base.sha256(Path(__file__)); lock["revision_reason"] = reason; lock["projection_model"] = {"u_x": U_X, "u_y": U_Y, "u_c": U_C, "source": "failed canary-v2 semantic observations only"}
    path.write_text(json.dumps(lock, indent=2) + "\n")

def main(stage: str) -> None:
    geometry = json.loads(base.GEOMETRY.read_text()); src_scenes = yaml.safe_load(base.SOURCE_SCENES.read_text()); src_ann = yaml.safe_load(base.SOURCE_ANNOTATIONS.read_text()); src_gate = yaml.safe_load(base.SOURCE_GATE.read_text())
    if stage == "canary":
        if OUT.exists(): raise FileExistsError(OUT)
        previous = base.REPORT / "CANARY_QC.json"
        if not previous.is_file() or json.loads(previous.read_text()).get("status") != "FAIL": raise RuntimeError("canary-v3 requires preserved v2 FAIL")
        rows, anns, index = [], {}, []
        for state in base.STATES:
            for ri, relation in enumerate(base.RELATIONS):
                row, ann, item = build_scene("canary_v3", state, relation, ri, geometry); rows.append(row); anns[row["scene_id"]] = ann; index.append(item)
        base.write_batch(OUT, "canary_v3", base.CAMERAS[0], rows, anns, src_scenes, src_ann, src_gate); add_lock(OUT, "Projection-calibrated revision after preserved canary-v1/v2 failures.")
        (OUT / "DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in index)); print(json.dumps({"status":"CANARY_V3_READY","families":16,"path":str(OUT.relative_to(ROOT))},indent=2)); return
    if json.loads((base.REPORT / "CANARY_QC.json").read_text()).get("status") != "PASS": raise RuntimeError("bulk sealed until canary PASS")
    bulk = base.RAW / "bulk"
    if bulk.exists(): raise FileExistsError(bulk)
    bulk.mkdir(); batches = {i:([],{},[]) for i in range(4)}; all_index=[]
    for state in base.STATES:
        for relation in base.RELATIONS:
            for repeat in range(32):
                row,ann,item=build_scene("bulk",state,relation,repeat,geometry); b=repeat%4; batches[b][0].append(row); batches[b][1][row["scene_id"]]=ann; batches[b][2].append(item); all_index.append(item)
    for b,(rows,anns,index) in batches.items():
        random.Random(base.SEED+b).shuffle(rows); folder=bulk/f"batch_{b}"; base.write_batch(folder,f"bulk_v3_b{b}",base.CAMERAS[b],rows,anns,src_scenes,src_ann,src_gate); add_lock(folder,"Bulk projection-calibrated by canary-v3."); (folder/"DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in index))
    (bulk/"DESIGN_INDEX.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in all_index)); cells=Counter((x["state"],x["relation"]) for x in all_index)
    summary={"schema_version":1,"status":"PASS_CAPTURE_NOT_STARTED","families":len(all_index),"states":dict(Counter(x["state"] for x in all_index)),"relations":dict(Counter(x["relation"] for x in all_index)),"splits":dict(Counter(x["split"] for x in all_index)),"cells":{f"{s}|{r}":cells[(s,r)] for s in base.STATES for r in base.RELATIONS},"unique_family_ids":len({x["family_id"] for x in all_index}),"unique_seeds":len({x["seed"] for x in all_index}),"unique_layouts":len({x["layout_signature_sha256"] for x in all_index})}; (bulk/"DESIGN_STATIC_QC.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("canary","bulk"));main(p.parse_args().stage)
