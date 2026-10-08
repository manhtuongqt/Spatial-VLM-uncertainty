#!/usr/bin/env python3
"""Top-table-camera, four-world-strata Development-R2 revision 5."""
from __future__ import annotations
import argparse,hashlib,json,random,sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import workspace.mh_pcrau_v3.day8_development_r2_prepare as base

OUT=base.RAW/'canary_v5';WORLDS=ROOT/'workspace/mh_pcrau_v3/generated/day8_development_r2_worlds_v1'
POSES=(
 '-0.975085296109 0.550000000000 1.100000000000 0 1.047197551197 0',
 '-0.960085296109 0.530000000000 1.090000000000 0 1.037197551197 0',
 '-0.990085296109 0.570000000000 1.110000000000 0 1.057197551197 0',
 '-0.975085296109 0.550000000000 1.080000000000 0 1.042197551197 0')
BASE_POSE='<pose>-0.975085296109 0.550000000000 1.100000000000 0 1.047197551197 0</pose>'

def make_worlds():
 WORLDS.mkdir(parents=True,exist_ok=True);text=base.WORLD.read_text()
 if text.count(BASE_POSE)!=1:raise RuntimeError('top camera pose marker drift')
 paths=[]
 for i,pose in enumerate(POSES):
  p=WORLDS/f'world_camera_{i}.sdf';content=text.replace(BASE_POSE,f'<pose>{pose}</pose>')
  if p.exists() and p.read_text()!=content:raise RuntimeError(f'world drift {p}')
  if not p.exists():p.write_text(content)
  paths.append(p)
 return paths

def base_positions(state,relation,dy):
 if state!='AMBIGUOUS':vals=[(-.34,.48),(-.24,.36),(-.14,.24)]
 elif relation=='leftmost':vals=[(-.31,.43),(-.17,.43),(-.24,.24)]
 elif relation=='rightmost':vals=[(-.24,.48),(-.31,.29),(-.17,.29)]
 elif relation=='second_from_left':vals=[(-.24,.48),(-.31,.36),(-.17,.36)]
 else:vals=[(-.31,.36),(-.17,.36),(-.24,.24)]
 return [[x,y+dy,0.] for x,y in vals]

def build(stage,state,relation,repeat,geometry):
 scene,ann,index=base.make_scene(stage,state,relation,repeat,geometry);design=base.common_design(stage,relation,repeat);fruits=design['fruits'];ti=base.rank_index(relation);target=fruits[ti]
 common=random.Random(int(hashlib.sha256(f'{base.NAMESPACE}|{stage}|{relation}|{repeat}|top-layout'.encode()).hexdigest()[:16],16));dy=common.uniform(-.14,.18);positions=base_positions(state,relation,dy)
 for i,fruit in enumerate(fruits):
  if state=='ABSENT' and i==ti:scene['poses'].pop(fruit,None)
  else:scene['poses'][fruit]=[positions[i][0],positions[i][1],common.uniform(-.35,.35)]
 covered=target if state=='INSUFFICIENT_EVIDENCE' else design['decoy'];occ=design['occluder'];g=geometry['pairs'][occ][covered];tx=-.10 if occ=='ycb_sugar_box' and covered=='ycb_apple' else -.24
 scene['poses'][covered]=[tx,.38,0.];scene['poses'][occ]=[tx-float(g['forward_separation_m']),.38+float(g['signed_lateral_offset_m']),float(g.get('occluder_yaw_rad',0.))]
 if state=='INSUFFICIENT_EVIDENCE':
  other=[f for f in fruits if f!=target]
  scene['poses'][other[0]]=[-.12,.22,0.];scene['poses'][other[1]]=[-.38,.23,0.]
 safe=[(-.46,.66),(-.34,.76),(-.20,.70),(-.08,.82),(-.48,.88),(-.10,.60)]
 for name,(x,y) in zip(base.CLUTTER,safe):scene['poses'][name]=[x+common.uniform(-.018,.018),y+common.uniform(-.018,.018),common.uniform(-.5,.5)]
 sig=hashlib.sha256(json.dumps(scene['poses'],sort_keys=True,separators=(',',':')).encode()).hexdigest();scene['layout_signature_sha256']=sig;ann['layout_signature_sha256']=sig;index['layout_signature_sha256']=sig
 return scene,ann,index

def lock_world(folder,world):
 p=folder/'CAPTURE_SOURCE_LOCK.json';lock=json.loads(p.read_text());rel=str(Path(__file__).relative_to(ROOT));wr=str(world.relative_to(ROOT));lock['source_artifact_sha256'][rel]=base.sha256(Path(__file__));lock['source_artifact_sha256'][wr]=base.sha256(world);lock['world_file']=wr;lock['camera_randomization']='four immutable top-table camera pose strata';p.write_text(json.dumps(lock,indent=2)+'\n')

def main(stage):
 worlds=make_worlds();geometry=json.loads(base.GEOMETRY.read_text());ss=yaml.safe_load(base.SOURCE_SCENES.read_text());sa=yaml.safe_load(base.SOURCE_ANNOTATIONS.read_text());sg=yaml.safe_load(base.SOURCE_GATE.read_text())
 if stage=='canary':
  if OUT.exists():raise FileExistsError(OUT)
  if json.loads((base.REPORT/'CANARY_QC.json').read_text()).get('status')!='FAIL':raise RuntimeError('v5 requires preserved v4 FAIL')
  OUT.mkdir();all_index=[]
  for ri,relation in enumerate(base.RELATIONS):
   rows=[];anns={};index=[]
   for state in base.STATES:
    row,ann,item=build('canary_v5',state,relation,ri,geometry);rows.append(row);anns[row['scene_id']]=ann;index.append(item);all_index.append(item)
   folder=OUT/f'camera_{ri}';base.write_batch(folder,f'canary_v5_c{ri}',base.CAMERAS[0],rows,anns,ss,sa,sg);lock_world(folder,worlds[ri]);(folder/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in index))
  (OUT/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in all_index));print(json.dumps({'status':'CANARY_V5_READY','families':16,'camera_strata':4},indent=2));return
 if json.loads((base.REPORT/'CANARY_QC.json').read_text()).get('status')!='PASS':raise RuntimeError('bulk sealed')
 bulk=base.RAW/'bulk'
 if bulk.exists():raise FileExistsError(bulk)
 bulk.mkdir();batches={i:([],{},[]) for i in range(4)};all_index=[]
 for state in base.STATES:
  for relation in base.RELATIONS:
   for repeat in range(32):
    row,ann,item=build('bulk',state,relation,repeat,geometry);b=repeat%4;batches[b][0].append(row);batches[b][1][row['scene_id']]=ann;batches[b][2].append(item);all_index.append(item)
 for b,(rows,anns,index) in batches.items():
  random.Random(base.SEED+b).shuffle(rows);folder=bulk/f'batch_{b}';base.write_batch(folder,f'bulk_v5_b{b}',base.CAMERAS[0],rows,anns,ss,sa,sg);lock_world(folder,worlds[b]);(folder/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in index))
 (bulk/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in all_index));cells=Counter((x['state'],x['relation']) for x in all_index);summary={'schema_version':1,'status':'PASS_CAPTURE_NOT_STARTED','families':len(all_index),'states':dict(Counter(x['state'] for x in all_index)),'relations':dict(Counter(x['relation'] for x in all_index)),'splits':dict(Counter(x['split'] for x in all_index)),'cells':{f'{s}|{r}':cells[(s,r)] for s in base.STATES for r in base.RELATIONS},'unique_family_ids':len({x['family_id'] for x in all_index}),'unique_seeds':len({x['seed'] for x in all_index}),'unique_layouts':len({x['layout_signature_sha256'] for x in all_index}),'camera_strata':4};(bulk/'DESIGN_STATIC_QC.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('stage',choices=('canary','bulk'));main(p.parse_args().stage)
