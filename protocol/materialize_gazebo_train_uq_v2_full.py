#!/usr/bin/env python3
"""Lock, all-or-nothing QC, and materialize Gazebo Train-UQ/Val-UQ v2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

import materialize_gazebo_train_uq_v1 as base

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "gazebo_train_uq_v2_full"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / f"{PROTOCOL}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL}_gate.yaml"
PRIMARY_LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_contract_lock.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_capture_compatibility_lock.json"
CAPTURE = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full/capture_attempt_01"
OUT = ROOT / "datasets/Gazebo_train_uq_v2_full"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full"
LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_materialization_lock.json"
QC = RESULT / "GAZEBO_TRAIN_UQ_V2_FULL_GEOMETRY_QC.json"
DUPLICATE_QC = RESULT / "GAZEBO_TRAIN_UQ_V2_FULL_RGB_DUPLICATE_QC.json"


def configure():
    for name, value in {"PROTOCOL": PROTOCOL, "SCENES": SCENES, "ANNOTATIONS": ANNOTATIONS,
                        "GATE": GATE, "PRIMARY_LOCK": PRIMARY_LOCK, "CAPTURE_LOCK": CAPTURE_LOCK,
                        "CAPTURE": CAPTURE, "OUT": OUT, "RESULT": RESULT, "LOCK": LOCK, "QC": QC}.items():
        setattr(base, name, value)


def lock_materialization():
    configure(); base.lock_materialization()
    payload = json.loads(LOCK.read_text())
    payload["wrapper_source_sha256"] = base.sha(Path(__file__).resolve())
    payload["full_pipeline_source_sha256"] = base.sha(ROOT / "protocol/gazebo_train_uq_v2_full_pipeline.py")
    payload["inference_contract"] = json.loads(PRIMARY_LOCK.read_text())["inference_lock"]
    payload["risk_feature_schema"] = json.loads(PRIMARY_LOCK.read_text())["risk_feature_schema"]
    payload["selection_rule"] = json.loads(PRIMARY_LOCK.read_text())["selection_rule"]
    payload["locked_at_utc_final"] = datetime.now(timezone.utc).isoformat()
    base.dump(LOCK, payload)
    print(json.dumps({"status": "LOCKED_FINAL", "sha256": base.sha(LOCK)}, indent=2))


def duplicate_audit():
    inputs = base.jsonl(CAPTURE / "input_manifest.jsonl")
    images = []
    for row in inputs:
        path = CAPTURE / row["input_files"]["rgb"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(path)
        images.append((row["scene_id"], row["input_sha256"]["rgb"], image))
    groups = {}
    for sid,digest,_ in images: groups.setdefault(digest,[]).append(sid)
    exact = [group for group in groups.values() if len(group)>1]
    near, closest = [], []
    # A conservative thumbnail screen selects candidates; the locked full-frame
    # metric is then computed exactly for every candidate.
    thumbs = [(sid, cv2.resize(image,(80,60),interpolation=cv2.INTER_AREA)) for sid,_,image in images]
    by_id = {sid:image for sid,_,image in images}
    for (left, a),(right,b) in itertools.combinations(thumbs,2):
        thumb_mad=float(np.mean(cv2.absdiff(a,b)))
        if thumb_mad < 2.0:
            diff=cv2.absdiff(by_id[left],by_id[right]); mad=float(np.mean(diff)); changed=float(np.mean(diff>=3))
            item={"left":left,"right":right,"gray_mad":mad,"changed_pixel_fraction":changed,"thumbnail_mad":thumb_mad}
            closest.append(item)
            if mad<0.05 and changed<0.002: near.append(item)
    closest=sorted(closest,key=lambda x:(x["gray_mad"],x["changed_pixel_fraction"]))[:20]
    report={"schema_version":1,"protocol_id":PROTOCOL,"status":"PASS" if not exact and not near else "REJECT",
            "checked_at_utc":datetime.now(timezone.utc).isoformat(),"rgb_count":len(images),"unique_rgb_sha256":len(groups),
            "exact_duplicate_groups":exact,"perceptual_near_duplicate_pairs":near,"closest_pairs":closest,
            "full_frame_locked_rule":{"gray_absdiff_threshold_for_changed_pixel":3,"near_duplicate_if_gray_mad_below":0.05,"and_changed_pixel_fraction_below":0.002},
            "candidate_screen":{"resolution":[80,60],"thumbnail_mad_below":2.0,"purpose":"conservative compute acceleration only"}}
    base.dump(DUPLICATE_QC,report)
    if report["status"]!="PASS": raise SystemExit(2)
    return report


def materialize(output: Path):
    configure()
    lock = json.loads(LOCK.read_text())
    if lock.get("wrapper_source_sha256") != base.sha(Path(__file__).resolve()): raise ValueError("wrapper changed after materialization lock")
    duplicate_audit()
    source = inspect.getsource(base.materialize).replace('"Gazebo_train_uq_v1"', '"Gazebo_train_uq_v2_full"')
    scope = dict(base.__dict__); scope.update(globals()); exec(source, scope); scope["materialize"](output)
    qc=json.loads(QC.read_text()); dup=json.loads(DUPLICATE_QC.read_text())
    qc["rgb_duplicate_qc_sha256"]=base.sha(DUPLICATE_QC);qc["rgb_duplicate_status"]=dup["status"]
    qc["full_geometry_and_duplicate_gate_passed"]=qc.get("status")=="PASS" and dup["status"]=="PASS"
    base.dump(QC,qc)
    manifest_path=output/"manifest.json"; manifest=json.loads(manifest_path.read_text());manifest["qc_report_sha256"]=base.sha(QC);manifest["rgb_duplicate_qc_sha256"]=base.sha(DUPLICATE_QC);base.dump(manifest_path,manifest)
    print(json.dumps({"status":"PASS","records":qc["records"],"failed":qc["failed_scene_count"],"duplicate_status":dup["status"]},indent=2))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True);sub.add_parser("lock-materialization");p=sub.add_parser("materialize");p.add_argument("--output",type=Path,default=OUT);args=parser.parse_args()
    if args.command=="lock-materialization":lock_materialization()
    else:materialize(args.output.resolve())


if __name__=="__main__":main()
