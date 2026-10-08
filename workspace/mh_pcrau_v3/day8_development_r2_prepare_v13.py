#!/usr/bin/env python3
"""Measured one-scene repair for the remaining ambiguous Day-8 cell."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as v7
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v11 as v11
OUT=v7.RAW/'canary_repair_v13'
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 scene,ann,index=v11.replacement(3,'AMBIGUOUS','second_from_right','canary_v7')
 token=hashlib.sha256(f'{v7.base.NAMESPACE}|canary_repair_v13|ambiguous|second_from_right'.encode()).hexdigest()[:20]
 sid=f'd8r2_canary_repair_v13_{token}';family=f'{v7.base.NAMESPACE}/canary_repair_v13/{token}'
 # Measured in v11/v12: u_pear changes +627 px/m in y.  y=.54333
 # targets u_pear≈u_apple≈302 px with a 3 px safety margin to the 12 px gate.
 scene['poses']['ycb_pear'][1]-=.050
 sig=hashlib.sha256(json.dumps(scene['poses'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 scene.update({'scene_id':sid,'scene_family_id':family,'layout_id':sid,'layout_signature_sha256':sig})
 ann.update({'family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'seed':int(token[:12],16),'camera_stratum':3,'failure_tags':list(ann.get('failure_tags',[]))+['targeted_repair_of_repair_v12_only','measured_pear_y_offset_minus_0p050m']})
 index.update({'scene_id':sid,'family_id':family,'seed':ann['seed'],'layout_signature_sha256':sig})
 ss=yaml.safe_load(v7.base.SOURCE_SCENES.read_text());sa=yaml.safe_load(v7.base.SOURCE_ANNOTATIONS.read_text());sg=yaml.safe_load(v7.base.SOURCE_GATE.read_text())
 OUT.mkdir(parents=True);folder=OUT/'camera_3';v7.base.write_batch(folder,'canary_repair_v13_c3',v7.base.CAMERAS[0],[scene],{sid:ann},ss,sa,sg);v7.lock(folder,v7.worlds()[3]);(folder/'DESIGN_INDEX.jsonl').write_text(json.dumps(index,sort_keys=True)+'\n')
 (OUT/'REPAIR_SCOPE.json').write_text(json.dumps({'status':'LOCKED_BEFORE_CAPTURE','accepted_v10_observations_retained':14,'accepted_v11_ie_repair_retained':1,'replacement_observations':1,'replaced_repair_v12_scene_id':'d8r2_canary_repair_v12_133ccd50b4ef9f9f58c7','repair_cell':'AMBIGUOUS|second_from_right','pear_y_offset_m':-.050,'no_bulk_capture':True},indent=2)+'\n')
 print(json.dumps({'status':'TARGETED_REPAIR_V13_READY','families':1,'retained':15},indent=2))
if __name__=='__main__':main()
