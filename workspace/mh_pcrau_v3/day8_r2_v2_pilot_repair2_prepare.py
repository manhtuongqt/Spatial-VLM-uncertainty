#!/usr/bin/env python3
"""Three measured camera-3 IE replacements; preserves all accepted pilot data."""
from __future__ import annotations
import json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as safe
OUT=safe.RAW/'pilot_r2_v2_repair2'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 ss=yaml.safe_load(safe.base.SOURCE_SCENES.read_text());sa=yaml.safe_load(safe.base.SOURCE_ANNOTATIONS.read_text());sg=yaml.safe_load(safe.base.SOURCE_GATE.read_text()); rows=[];anns={}
 for rel in ('leftmost','second_from_left','second_from_right'):
  row,ann,_=safe.build('pilot_r2_v2_repair2','INSUFFICIENT_EVIDENCE',rel,3)
  # calibrated camera-3 strengthening: sugar box overlaps apple more than v7
  row['poses']['ycb_sugar_box'][1]=.410
  ann['split']='pilot_validation';ann['failure_tags']+=['targeted_repair2_camera3_ie','sugar_y_0p410']
  rows.append(row);anns[row['scene_id']]=ann
 OUT.mkdir();f=OUT/'batch_3';safe.base.write_batch(f,'pilot_r2_v2_repair2_b3',safe.base.CAMERAS[3],rows,anns,ss,sa,sg);safe.lock(f,safe.worlds()[3]);(OUT/'REPAIR_SCOPE.json').write_text(json.dumps({'replacements':3,'retained':61,'camera':3,'state':'INSUFFICIENT_EVIDENCE','bulk_prohibited':True},indent=2)+'\n');print('READY 3')
if __name__=='__main__':main()
