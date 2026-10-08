#!/usr/bin/env python3
"""Final two-pair sugar-box geometry canary (wide-face apple, corrected pear)."""
from __future__ import annotations
import ast,hashlib,json,math
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import yaml
from workspace.mh_pcrau_v3.day6_left_ycb_prepare import BOXES,BOX_LABELS,CLUTTER,FRUIT_LABELS,ROOT,VIEW_POSE,add_inventory
OUT=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v10_sugar';WORLD=ROOT/'workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf';CONFIG=ROOT/'ur3/ur3_perception/config';CAPTURE=ROOT/'ur3/ur3_perception/scripts/roborefer_pilot_capture.py';LAUNCH=ROOT/'workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py'
SS=CONFIG/'gazebo_train_uq_v2_full_r3_scenes.yaml';SA=CONFIG/'gazebo_train_uq_v2_full_r3_annotations.yaml';SG=CONFIG/'gazebo_train_uq_v2_full_r3_gate.yaml';PROTOCOL='mh_pcrau_v3_left_ycb_canary_v10_sugar'
SCAN={'ycb_apple':[(-.04,math.pi/2),(-.03,math.pi/2),(-.02,math.pi/2),(-.01,math.pi/2),(0.,math.pi/2),(.01,math.pi/2),(.02,math.pi/2),(.03,math.pi/2),(.04,math.pi/2)],'ycb_pear':[(-.006,0.),(-.002,0.),(.002,0.),(.006,0.),(.010,0.),(.014,0.),(.018,0.)]}
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def sources():
 for n in ast.parse(CAPTURE.read_text()).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REQUIRED_SOURCE_ARTIFACTS' for t in n.targets):return set(ast.literal_eval(n.value))
 raise RuntimeError
def main():
 if OUT.exists():raise FileExistsError(OUT)
 ss=yaml.safe_load(SS.read_text());sa=yaml.safe_load(SA.read_text());sg=yaml.safe_load(SG.read_text());objects=add_inventory(ss['objects']);rows=[];anns={}
 for pi,(fruit,values) in enumerate(SCAN.items()):
  for oi,(off,yaw) in enumerate(values):
   sid=f'v10can_{pi}_{oi}';others=[x for x in ('ycb_apple','ycb_orange','mango','ycb_pear','ycb_plum') if x!=fruit][:2];poses={fruit:[-.24,.38,0.],others[0]:[-.12,.22,0.],others[1]:[-.38,.23,0.],'ycb_sugar_box':[-.326,.38+off,yaw]}
   for n,xy in zip(CLUTTER,[(-.46,.66),(-.34,.76),(-.20,.70),(-.08,.82),(-.48,.88),(-.10,.60)]):poses[n]=[xy[0],xy[1],0.]
   poses['ycb_cracker_box']=[-.48,.98,0.];poses['ycb_bleach_cleanser']=[-.04,.98,0.];poses['ycb_lemon']=[5.34,3.,0.];family=f'mh_pcrau_v3/canary_v10/sugar/{fruit}/{oi}';sig=hashlib.sha256(json.dumps(poses,sort_keys=True).encode()).hexdigest();rows.append({'scene_id':sid,'scene_family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'task_type':'projection_calibration_canary_only','instruction':f'Locate {fruit} behind sugar box.','poses':poses});anns[sid]={'state':'INSUFFICIENT_EVIDENCE','target_id':fruit,'target_label':FRUIT_LABELS[fruit],'candidate_ids':[fruit,*others],'candidate_labels':[FRUIT_LABELS[x] for x in [fruit,*others]],'occluder_ids':['ycb_sugar_box'],'occluder_labels':[BOX_LABELS['ycb_sugar_box']],'family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'split':'projection_canary_only','projection_calibration':{'box':'ycb_sugar_box','fruit':fruit,'forward_separation_m':.086,'signed_lateral_offset_m':off,'occluder_yaw_rad':yaw}}
 scenes=deepcopy(ss);scenes.update({'protocol_id':PROTOCOL,'expected_scene_count':len(rows),'random_seed':25092051,'view_joint_pose':VIEW_POSE,'camera_frame':'top_table_camera_optical_frame','objects':objects,'scenes':rows});oracle=deepcopy(sa);oracle.update({'protocol_id':PROTOCOL,'oracle_usage':'projection_calibration_qc_only','scenes':anns});gate=deepcopy(sg);gate.update({'protocol_id':PROTOCOL,'parent_family_count':len(rows),'policies':{**sg['policies'],'no_training':True,'no_model_inference':True}});OUT.mkdir(parents=True)
 for n,p in [('scenes.yaml',scenes),('annotations.yaml',oracle),('gate.yaml',gate)]: (OUT/n).write_text(yaml.safe_dump(p,sort_keys=False,allow_unicode=True,width=160))
 rel=lambda p:str(Path(p).relative_to(ROOT));src=sources()|{rel(WORLD),rel(Path(__file__)),rel(LAUNCH),rel(OUT/'scenes.yaml'),rel(OUT/'annotations.yaml'),rel(OUT/'gate.yaml'),'ur3/ur3_perception/launch/roborefer_uq_capture.launch.py'};lock={'schema_version':1,'protocol_id':PROTOCOL,'status':'LOCKED_BEFORE_CAPTURE_AND_INFERENCE','locked_at_utc':datetime.now(timezone.utc).isoformat(),'workspace_root':str(ROOT),'source_artifact_sha256':{x:sha256(ROOT/x) for x in sorted(src)},'model_inventory_sha256':'NO_MODEL_INFERENCE_PROJECTION_CANARY','planned_scene_count':len(rows),'view_joint_pose':VIEW_POSE,'world_file':rel(WORLD)};(OUT/'CAPTURE_SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print({'status':'READY','scenes':len(rows)})
if __name__=='__main__':main()
