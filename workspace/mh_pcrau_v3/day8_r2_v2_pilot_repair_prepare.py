#!/usr/bin/env python3
"""Prepare targeted replacements for the 11 failed R2-v2 pilot families."""
from __future__ import annotations
import json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare_v7 as safe
from workspace.mh_pcrau_v3.day8_development_r2_qc import collect,inspect
OUT=safe.RAW/'pilot_r2_v2_repair'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 base=safe.RAW/'pilot_r2_v2'; records=sum((collect(base/f'batch_{i}') for i in range(4)),[]);_,fail,_=inspect(records,False); bad={x.split(':',1)[0] for x in fail}
 ss=yaml.safe_load(safe.base.SOURCE_SCENES.read_text());sa=yaml.safe_load(safe.base.SOURCE_ANNOTATIONS.read_text());sg=yaml.safe_load(safe.base.SOURCE_GATE.read_text());packs={i:([],{},[]) for i in range(4)};OUT.mkdir()
 for old in records:
  if old['scene_id'] not in bad:continue
  state,rel,cam=old['state'],old['relation'],old['camera_stratum']
  scene,ann,index=safe.build('pilot_r2_v2_repair',state,rel,cam)
  ann['split']=old['split'];index['split']=old['split'];ann['failure_tags']+=['targeted_pilot_semantic_repair',f'replaces:{old["scene_id"]}']
  packs[cam][0].append(scene);packs[cam][1][scene['scene_id']]=ann;packs[cam][2].append(index)
 for cam,(rows,anns,idx) in packs.items():
  if not rows:continue
  folder=OUT/f'batch_{cam}';safe.base.write_batch(folder,f'pilot_r2_v2_repair_b{cam}',safe.base.CAMERAS[cam],rows,anns,ss,sa,sg);safe.lock(folder,safe.worlds()[cam]);(folder/'REPLACEMENTS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in idx))
 (OUT/'REPAIR_SCOPE.json').write_text(json.dumps({'retained_pass':64-len(bad),'replacements':len(bad),'replaced_scene_ids':sorted(bad),'bulk_prohibited':True},indent=2)+'\n');print(json.dumps({'status':'READY','replacements':len(bad),'by_batch':{k:len(v[0]) for k,v in packs.items()}},indent=2))
if __name__=='__main__':main()
