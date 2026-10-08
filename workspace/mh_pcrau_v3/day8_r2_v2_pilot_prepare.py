#!/usr/bin/env python3
"""Prepare the 64-family anti-shortcut pilot before any R2-v2 bulk capture."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare as base
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as worlds
OUT=base.RAW/'pilot_r2_v2'
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 ss=yaml.safe_load(base.SOURCE_SCENES.read_text());sa=yaml.safe_load(base.SOURCE_ANNOTATIONS.read_text());sg=yaml.safe_load(base.SOURCE_GATE.read_text()); ws=worlds.worlds();OUT.mkdir()
 packs={i:([],{},[]) for i in range(4)}; allidx=[]
 for state in base.STATES:
  for relation in base.RELATIONS:
   for repeat in range(4):
    row,ann,index=base.make_scene('pilot_r2_v2',state,relation,repeat,json.loads(base.GEOMETRY.read_text()))
    ann['split']='pilot_train' if repeat<3 else 'pilot_validation'; index['split']=ann['split'];ann['failure_tags']+=['r2_v2_state_matched_visual_distribution','pilot_only']
    camera=repeat; packs[camera][0].append(row);packs[camera][1][row['scene_id']]=ann;packs[camera][2].append(index);allidx.append(index)
 for camera,(rows,anns,idx) in packs.items():
  folder=OUT/f'batch_{camera}';base.write_batch(folder,f'pilot_r2_v2_b{camera}',base.CAMERAS[camera],rows,anns,ss,sa,sg);worlds.lock(folder,ws[camera]);(folder/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in idx))
 (OUT/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in allidx));(OUT/'PILOT_CONTRACT.json').write_text(json.dumps({'families':64,'per_cell':4,'split':{'pilot_train':48,'pilot_validation':16},'bulk_prohibited_until':'semantic_and_shortcut_pilot_pass'},indent=2)+'\n')
 print(json.dumps({'status':'PILOT_R2_V2_READY','families':len(allidx),'cells':{f'{s}|{r}':sum(x['state']==s and x['relation']==r for x in allidx) for s in base.STATES for r in base.RELATIONS}},indent=2))
if __name__=='__main__':main()
