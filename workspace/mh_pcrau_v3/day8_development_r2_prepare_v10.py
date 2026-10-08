#!/usr/bin/env python3
"""v10 measured final canary correction; no threshold changes."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as v7
OUT=v7.RAW/'canary_v10'; BUILD=v7.build
def build(stage,state,relation,repeat):
 scene,ann,index=BUILD('canary_v10',state,relation,repeat)
 if state=='AMBIGUOUS' and relation=='leftmost':scene['poses']['ycb_pear'][1]+=.035
 if state=='AMBIGUOUS' and relation=='second_from_right':scene['poses']['ycb_pear'][1]+=.024
 if state=='INSUFFICIENT_EVIDENCE':scene['poses']['ycb_sugar_box'][1]=.402
 sig=hashlib.sha256(json.dumps(scene['poses'],sort_keys=True,separators=(',',':')).encode()).hexdigest();scene['layout_signature_sha256']=sig;ann['layout_signature_sha256']=sig;index['layout_signature_sha256']=sig
 return scene,ann,index
def main():
 if OUT.exists():raise FileExistsError(OUT)
 oldout,oldbuild=v7.OUT,v7.build
 try:
  v7.OUT=OUT;v7.build=build;v7.prepare('canary')
  mine=str(Path(__file__).relative_to(v7.ROOT));dig=v7.base.sha256(Path(__file__))
  for p in OUT.glob('camera_*/CAPTURE_SOURCE_LOCK.json'):
   x=json.loads(p.read_text());x['source_artifact_sha256'][mine]=dig;x['layout_revision']='v10 measured IE/tie correction';p.write_text(json.dumps(x,indent=2)+'\n')
 finally:v7.OUT=oldout;v7.build=oldbuild
if __name__=='__main__':argparse.ArgumentParser().parse_args();main()
