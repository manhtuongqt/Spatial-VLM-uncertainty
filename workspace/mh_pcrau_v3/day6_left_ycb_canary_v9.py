#!/usr/bin/env python3
"""Targeted v5 projection rescan for the nine v8 failures."""
from __future__ import annotations
import ast, hashlib, json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import yaml
from workspace.mh_pcrau_v3.day6_left_ycb_prepare import BOXES, BOX_LABELS, CLUTTER, FRUIT_LABELS, ROOT, VIEW_POSE, add_inventory

OUT=ROOT/"ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v9_targeted"
WORLD=ROOT/"workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf"
CONFIG=ROOT/"ur3/ur3_perception/config"; CAPTURE=ROOT/"ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
LAUNCH=ROOT/"workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"; PROTOCOL="mh_pcrau_v3_left_ycb_canary_v9_targeted"
SOURCE_SCENES=CONFIG/"gazebo_train_uq_v2_full_r3_scenes.yaml"; SOURCE_ANN=CONFIG/"gazebo_train_uq_v2_full_r3_annotations.yaml"; SOURCE_GATE=CONFIG/"gazebo_train_uq_v2_full_r3_gate.yaml"
FRUITS=("ycb_apple","ycb_orange","mango","ycb_pear","ycb_plum")
SCAN={
 ("ycb_cracker_box","ycb_apple"):[-.046,-.042,-.038,-.034,-.030,-.026,-.022],
 ("ycb_cracker_box","ycb_orange"):[-.046,-.042,-.038,-.034,-.030,-.026,-.022],
 ("ycb_cracker_box","mango"):[.075,.081,.087,.093,.099,.105,.111],
 ("ycb_cracker_box","ycb_pear"):[-.046,-.042,-.038,-.034,-.030,-.026,-.022],
 ("ycb_sugar_box","ycb_apple"):[.02,.03,.04,.05,.06,.07,.08,.09,.10],
 ("ycb_sugar_box","ycb_pear"):[-.006,-.002,.002,.006,.010,.014,.018],
 ("ycb_bleach_cleanser","ycb_apple"):[-.004,.000,.004,.008,.012,.016,.020,.024,.028],
 ("ycb_bleach_cleanser","ycb_orange"):[-.004,.000,.004,.008,.012,.016,.020,.024,.028],
 ("ycb_bleach_cleanser","ycb_pear"):[-.010,-.004,.002,.008,.014,.020,.026,.032,.038],
}
FORWARD={"ycb_cracker_box":.102,"ycb_sugar_box":.135,"ycb_bleach_cleanser":.092}
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def required_sources():
 tree=ast.parse(CAPTURE.read_text())
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REQUIRED_SOURCE_ARTIFACTS' for t in node.targets): return set(ast.literal_eval(node.value))
 raise RuntimeError('inventory missing')
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 ss=yaml.safe_load(SOURCE_SCENES.read_text()); sa=yaml.safe_load(SOURCE_ANN.read_text()); sg=yaml.safe_load(SOURCE_GATE.read_text()); objects=add_inventory(ss['objects'])
 rows=[]; anns={}; idx=[]
 for pi,((box,fruit),offsets) in enumerate(SCAN.items()):
  for oi,offset in enumerate(offsets):
   sid=f'v9can_{pi}_{oi}'; family=f'mh_pcrau_v3/canary_v9/{box}/{fruit}/{oi}'; others=[x for x in FRUITS if x!=fruit][:2]
   poses={fruit:[-.24,.38,0.0],others[0]:[-.12,.22,0.0],others[1]:[-.38,.23,0.0],box:[-.24-FORWARD[box],.38+offset,0.0]}
   fixed=[(-.46,.66),(-.34,.76),(-.20,.70),(-.08,.82),(-.48,.88),(-.10,.60)]
   for n,xy in zip(CLUTTER,fixed): poses[n]=[xy[0],xy[1],0.0]
   for bj,b in enumerate(BOXES):
    if b!=box: poses[b]=[-.48+.22*bj,.98,0.0]
   poses['ycb_lemon']=[5.34,3.,0.]; sig=hashlib.sha256(json.dumps(poses,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   rows.append({'scene_id':sid,'scene_family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'task_type':'projection_calibration_canary_only','instruction':f'Locate {fruit} behind {box}.','poses':poses})
   anns[sid]={'state':'INSUFFICIENT_EVIDENCE','target_id':fruit,'target_label':FRUIT_LABELS[fruit],'candidate_ids':[fruit,*others],'candidate_labels':[FRUIT_LABELS[x] for x in [fruit,*others]],'occluder_ids':[box],'occluder_labels':[BOX_LABELS[box]],'family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'split':'projection_canary_only','projection_calibration':{'box':box,'fruit':fruit,'forward_separation_m':FORWARD[box],'signed_lateral_offset_m':offset}}
   idx.append({'scene_id':sid,'box':box,'fruit':fruit,'signed_lateral_offset_m':offset,'layout_signature_sha256':sig})
 scenes=deepcopy(ss);scenes.update({'protocol_id':PROTOCOL,'expected_scene_count':len(rows),'random_seed':25092049,'view_joint_pose':VIEW_POSE,'camera_frame':'top_table_camera_optical_frame','objects':objects,'scenes':rows})
 oracle=deepcopy(sa);oracle.update({'protocol_id':PROTOCOL,'oracle_usage':'projection_calibration_qc_only','scenes':anns})
 gate=deepcopy(sg);gate.update({'protocol_id':PROTOCOL,'parent_family_count':len(rows),'projection_calibration_gate':{'targeted_pairs_exact':9,'pass_rule':'each targeted pair has 1<=target_visible_pixels<120'},'policies':{**sg['policies'],'no_training':True,'no_model_inference':True}})
 OUT.mkdir(parents=True)
 for n,p in [('scenes.yaml',scenes),('annotations.yaml',oracle),('gate.yaml',gate)]: (OUT/n).write_text(yaml.safe_dump(p,sort_keys=False,allow_unicode=True,width=160))
 ip=OUT/'DESIGN_INDEX.jsonl';ip.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in idx)); rel=lambda p:str(Path(p).relative_to(ROOT))
 src=required_sources()|{rel(WORLD),rel(Path(__file__)),rel(LAUNCH),rel(OUT/'scenes.yaml'),rel(OUT/'annotations.yaml'),rel(OUT/'gate.yaml'),rel(ip),'ur3/ur3_perception/launch/roborefer_uq_capture.launch.py'}
 lock={'schema_version':1,'protocol_id':PROTOCOL,'status':'LOCKED_BEFORE_CAPTURE_AND_INFERENCE','locked_at_utc':datetime.now(timezone.utc).isoformat(),'workspace_root':str(ROOT),'source_artifact_sha256':{x:sha256(ROOT/x) for x in sorted(src)},'model_inventory_sha256':'NO_MODEL_INFERENCE_PROJECTION_CANARY','planned_scene_count':len(rows),'view_joint_pose':VIEW_POSE,'world_file':rel(WORLD),'lemon_excluded':True}
 (OUT/'CAPTURE_SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps({'status':'READY','targeted_pairs':9,'scenes':len(rows)},indent=2))
if __name__=='__main__':main()
