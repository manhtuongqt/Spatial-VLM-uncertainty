#!/usr/bin/env python3
"""Lock the fresh 256-family Day-6 Anti-Shortcut validation on world v5."""
from __future__ import annotations
import ast,hashlib,json,random
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import yaml
from workspace.mh_pcrau_v3.day6_left_ycb_prepare import BOXES,BOX_LABELS,CLUTTER,FRUIT_LABELS,ROOT,VIEW_POSE,add_inventory,tie_indices,rank_index

OUT=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_final_v11'; WORLD=ROOT/'workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf'
GEOMETRY=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v13_sugar_apple_near/SELECTED_15_PAIR_GEOMETRY.json'; CANARY_QC=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v13_sugar_apple_near/FINAL_15_PAIR_CANARY_QC.json'
CONFIG=ROOT/'ur3/ur3_perception/config'; SS=CONFIG/'gazebo_train_uq_v2_full_r3_scenes.yaml'; SA=CONFIG/'gazebo_train_uq_v2_full_r3_annotations.yaml'; SG=CONFIG/'gazebo_train_uq_v2_full_r3_gate.yaml'
CAPTURE=ROOT/'ur3/ur3_perception/scripts/roborefer_pilot_capture.py';LAUNCH=ROOT/'workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py';CHECKPOINT=ROOT/'workspace/mh_pcrau_v3/checkpoints/day06_robust_candidate_r2/s1a_robust_fixed_epoch54.pt'
PROTOCOL='mh_pcrau_v3_anti_shortcut_left_ycb_final_v11';SEED=25092051
STATES=('FOUND','AMBIGUOUS','ABSENT','INSUFFICIENT_EVIDENCE');RELATIONS=('leftmost','rightmost','second_from_left','second_from_right')
FRUITS=('ycb_apple','ycb_orange','mango','ycb_pear','ycb_plum'); NAMES={'ycb_apple':'apple','ycb_orange':'orange','mango':'mango','ycb_pear':'pear','ycb_plum':'plum'}
SUFFIX='If exactly one requested target is identifiable and sufficiently visible, return exactly POINT [(x, y)], using one normalized image point with x and y between 0 and 1. If the request is ambiguous, the target is absent, or visual evidence is insufficient, return exactly ABSTAIN. Return no other text.'
PHRASES={
 'leftmost':['identify the leftmost object in the image','find the leftmost item in the image','select the object that is leftmost in the image','return the leftmost object in the image'],
 'rightmost':['identify the rightmost object in the image','find the rightmost item in the image','select the object that is rightmost in the image','return the rightmost object in the image'],
 'second_from_left':['identify the second object from left to right in the image','find the second item from the left in the image','select the object that is second from the left in the image','return the second object from the left in the image'],
 'second_from_right':['identify the second object from right to left in the image','find the second item from the right in the image','select the object that is second from the right in the image','return the second object from the right in the image']}
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def required_sources():
 tree=ast.parse(CAPTURE.read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REQUIRED_SOURCE_ARTIFACTS' for t in n.targets):return set(ast.literal_eval(n.value))
 raise RuntimeError('inventory missing')
def base_positions(state,relation):
 if state!='AMBIGUOUS': return [(-.34,.48),(-.24,.36),(-.14,.24)]
 if relation=='leftmost':return [(-.31,.43),(-.17,.43),(-.24,.24)]
 if relation=='rightmost':return [(-.24,.48),(-.31,.29),(-.17,.29)]
 if relation=='second_from_left':return [(-.24,.48),(-.31,.36),(-.17,.36)]
 return [(-.31,.36),(-.17,.36),(-.24,.24)]
def prompt(relation,fruits,variant):
 names=', '.join(NAMES[x] for x in fruits[:-1])+' and '+NAMES[fruits[-1]]
 return f'Among the {names}, {PHRASES[relation][variant]}. {SUFFIX}'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 qc=json.loads(CANARY_QC.read_text());geo=json.loads(GEOMETRY.read_text())
 if qc['status']!='PASS' or geo['status']!='LOCKED':raise RuntimeError('15-pair canary not eligible')
 ss=yaml.safe_load(SS.read_text());sa=yaml.safe_load(SA.read_text());sg=yaml.safe_load(SG.read_text());objects=add_inventory(ss['objects']);objects['ycb_lemon']['storage_pose']=[5.34,3.,0.]
 OUT.mkdir(parents=True);all_index=[];signatures=set();cells=Counter();boxpairs=Counter()
 for batch in range(4):
  folder=OUT/f'batch_{batch}';folder.mkdir();rows=[];anns={}
  for si,state in enumerate(STATES):
   for ri,relation in enumerate(RELATIONS):
    for repeat in range(4):
     token=hashlib.sha256(f'{PROTOCOL}|{SEED}|{batch}|{state}|{relation}|{repeat}'.encode()).hexdigest()[:16];rng=random.Random(int(token,16));all_fruits=list(FRUITS);rng.shuffle(all_fruits);fruits=all_fruits[:3];target=fruits[rank_index(relation)];decoy=all_fruits[3]
     positions=base_positions(state,relation);dx=rng.uniform(-.105,.105);dy=rng.uniform(-.008,.008);poses={}
     for i,(fruit,(x,y)) in enumerate(zip(fruits,positions)):
      if state=='ABSENT' and i==rank_index(relation):continue
      poses[fruit]=[x+dx,y+dy,rng.uniform(-.32,.32)]
     # Put one calibrated box-fruit occlusion in every state so a 16x12
     # thumbnail cannot identify answerability merely from box presence.
     if state=='INSUFFICIENT_EVIDENCE':covered=target
     elif state=='FOUND':covered=next(x for x in fruits if x!=target)
     elif state=='AMBIGUOUS':covered=next(x for x in fruits if x not in [fruits[i] for i in tie_indices(relation)])
     else:
      covered=decoy;poses[decoy]=[-.24,.38,0.0]
     occluder=BOXES[(batch*16+ri*4+repeat)%3];g=geo['pairs'][occluder][covered]
     target_x=-.10 if occluder=='ycb_sugar_box' and covered=='ycb_apple' else -.24
     poses[covered]=[target_x,.38,0.0];tx,ty,_=poses[covered]
     poses[occluder]=[tx-float(g['forward_separation_m']),ty+float(g['signed_lateral_offset_m']),float(g.get('occluder_yaw_rad',0.0))]
     if state=='INSUFFICIENT_EVIDENCE':boxpairs[(occluder,target)]+=1
     fixed=[(-.48,.70),(-.36,.82),(-.22,.72),(-.08,.84),(-.47,.93),(-.09,.62)]
     rng.shuffle(fixed)
     for n,(x,y) in zip(CLUTTER,fixed):poses[n]=[x+rng.uniform(-.035,.035),y+rng.uniform(-.035,.035),rng.uniform(-.8,.8)]
     for bj,b in enumerate(BOXES):
      if b!=occluder:poses[b]=[-.48+.22*bj,.98,0.0]
     poses['ycb_lemon']=[5.34,3.,0.]
     sig=hashlib.sha256(json.dumps(poses,sort_keys=True,separators=(',',':')).encode()).hexdigest()
     if sig in signatures:
      raise RuntimeError('duplicate layout')
     signatures.add(sig)
     sid=f'fresh_{token}';family=f'mh_pcrau_v3/anti_shortcut_v10/{token}';instruction=prompt(relation,fruits,repeat)
     rows.append({'scene_id':sid,'scene_family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'task_type':'left_ycb_ordinal_grounding','instruction':instruction,'poses':poses})
     ann={'state':state,'target_id':target if state!='AMBIGUOUS' else None,'target_label':FRUIT_LABELS[target] if state!='AMBIGUOUS' else None,'candidate_ids':fruits if state!='ABSENT' else [],'candidate_labels':[FRUIT_LABELS[x] for x in fruits] if state!='ABSENT' else [],'rank_from':'left' if relation in ('leftmost','second_from_left') else 'right','rank':1 if relation in ('leftmost','rightmost') else 2,'relation_variant':relation,'family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'split':'anti_shortcut_val_v1','camera_stratum':batch,'prompt_variant':repeat,'failure_tags':['fresh_family','world_v5','lemon_excluded']}
     if state=='AMBIGUOUS':
      valid=[fruits[i] for i in tie_indices(relation)];ann.update({'valid_target_ids':valid,'valid_target_labels':[FRUIT_LABELS[x] for x in valid]})
     elif state=='ABSENT':
      context=[x for x in fruits if x!=target];ann.update({'context_ids':context,'context_labels':[FRUIT_LABELS[x] for x in context]})
     elif state=='INSUFFICIENT_EVIDENCE':ann.update({'occluder_ids':[occluder],'occluder_labels':[BOX_LABELS[occluder]]})
     anns[sid]=ann;cells[(state,relation)]+=1;all_index.append({'scene_id':sid,'family_id':family,'batch':batch,'state':state,'relation':relation,'target':target,'primary_occluder':occluder,'layout_signature_sha256':sig})
  random.Random(SEED+batch).shuffle(rows);scenes=deepcopy(ss);scenes.update({'protocol_id':f'{PROTOCOL}_b{batch}','expected_scene_count':64,'random_seed':SEED+batch,'view_joint_pose':VIEW_POSE,'camera_frame':'top_table_camera_optical_frame','objects':objects,'scenes':rows})
  oracle=deepcopy(sa);oracle.update({'protocol_id':scenes['protocol_id'],'oracle_usage':'evaluator_only_after_prediction_lock','scenes':anns})
  gate=deepcopy(sg);gate.update({'protocol_id':scenes['protocol_id'],'parent_family_count':64,'state_quota':{'anti_shortcut_val_v1':{s:16 for s in STATES}},'relation_variant_quota':{'anti_shortcut_val_v1':{r:16 for r in RELATIONS}},'state_relation_cell_quota':{'anti_shortcut_val_v1':4},'camera':{'frame':'top_table_camera_optical_frame','resolution':[640,480],'view_joint_pose':VIEW_POSE,'depth_unit':'metre','valid_depth_range_m':[.1,3.]},'policies':{**sg['policies'],'no_training':True,'one_shot_model_inference':True,'no_materialization_before_qc':True}})
  for n,p in [('scenes.yaml',scenes),('annotations.yaml',oracle),('gate.yaml',gate)]: (folder/n).write_text(yaml.safe_dump(p,sort_keys=False,allow_unicode=True,width=180))
  rel=lambda p:str(Path(p).relative_to(ROOT));src=required_sources()|{rel(WORLD),rel(Path(__file__)),rel(LAUNCH),rel(GEOMETRY),rel(CANARY_QC),rel(folder/'scenes.yaml'),rel(folder/'annotations.yaml'),rel(folder/'gate.yaml'),'ur3/ur3_perception/launch/roborefer_uq_capture.launch.py'}
  lock={'schema_version':1,'protocol_id':scenes['protocol_id'],'status':'LOCKED_BEFORE_CAPTURE_AND_INFERENCE','locked_at_utc':datetime.now(timezone.utc).isoformat(),'workspace_root':str(ROOT),'source_artifact_sha256':{x:sha256(ROOT/x) for x in sorted(src)},'model_inventory_sha256':sha256(CHECKPOINT),'frozen_model':rel(CHECKPOINT),'planned_scene_count':64,'view_joint_pose':VIEW_POSE,'world_file':rel(WORLD),'lemon_excluded':True}
  (folder/'CAPTURE_SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
 ip=OUT/'DESIGN_INDEX.jsonl';ip.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in all_index));report={'schema_version':1,'status':'PASS_CAPTURE_NOT_STARTED','families':len(all_index),'state_counts':dict(Counter(x['state'] for x in all_index)),'relation_counts':dict(Counter(x['relation'] for x in all_index)),'cell_counts':{f'{s}|{r}':cells[(s,r)] for s in STATES for r in RELATIONS},'ie_pair_coverage':{f'{b}|{f}':boxpairs[(b,f)] for b in BOXES for f in FRUITS},'unique_layouts':len(signatures),'lemon_excluded':True,'world_sha256':sha256(WORLD),'canary_qc_sha256':sha256(CANARY_QC),'checkpoint_sha256':sha256(CHECKPOINT)}
 (OUT/'DESIGN_STATIC_QC.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
