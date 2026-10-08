#!/usr/bin/env python3
"""Measured v8 canary correction; preserves all prior failed revisions."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as v7

OUT=v7.RAW/'canary_v8'
V7_BUILD=v7.build

def build(stage,state,relation,repeat):
    # v7's prepare loop is reused, but the token namespace remains fresh.
    scene,ann,index=V7_BUILD('canary_v8',state,relation,repeat)
    # Measured v7 correction: make the leftmost tie co-linear in image space.
    if state=='AMBIGUOUS' and relation=='leftmost':
        scene['poses']['ycb_pear'][1] += .035
    # v7 showed 160 px only at camera-0; 4 mm advances the same calibrated
    # sugar cover while retaining 99/102/99 px at the other three strata.
    if state=='INSUFFICIENT_EVIDENCE':
        scene['poses']['ycb_sugar_box'][1] = .400
    sig=hashlib.sha256(json.dumps(scene['poses'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    scene['layout_signature_sha256']=sig;ann['layout_signature_sha256']=sig;index['layout_signature_sha256']=sig
    return scene,ann,index

def add_lock_hashes():
    mine=str(Path(__file__).relative_to(v7.ROOT)); digest=v7.base.sha256(Path(__file__))
    for lock in OUT.glob('camera_*/CAPTURE_SOURCE_LOCK.json'):
        value=json.loads(lock.read_text());value['source_artifact_sha256'][mine]=digest;value['layout_revision']='v8 measured tie/occlusion adjustment';lock.write_text(json.dumps(value,indent=2)+'\n')

def main(stage):
    if stage!='canary': raise RuntimeError('v8 is canary-only; bulk remains sealed until QC PASS')
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(OUT)
    old_out,old_build=v7.OUT,v7.build
    try:
        v7.OUT=OUT;v7.build=build;v7.prepare('canary');add_lock_hashes()
    finally:
        v7.OUT=old_out;v7.build=old_build

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('canary',));main(p.parse_args().stage)
