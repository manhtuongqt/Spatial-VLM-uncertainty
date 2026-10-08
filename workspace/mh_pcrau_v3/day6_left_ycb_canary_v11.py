#!/usr/bin/env python3
"""Seven-frame final sugar/apple refinement with a larger ray separation."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v10_sugar';OUT=ROOT/'ketqua1/09_danh_gia/anti_shortcut_left_ycb_canary_v13_sugar_apple_near';WORLD=ROOT/'workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf';SCRIPT=ROOT/'ur3/ur3_perception/scripts/roborefer_pilot_capture.py';LAUNCH=ROOT/'workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py'
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 scenes=yaml.safe_load((SRC/'scenes.yaml').read_text());ann=yaml.safe_load((SRC/'annotations.yaml').read_text());gate=yaml.safe_load((SRC/'gate.yaml').read_text());template=next(x for x in scenes['scenes'] if x['scene_id']=='v10can_0_0');at=ann['scenes']['v10can_0_0'];rows=[];anns={}
 for i,off in enumerate((.012,.016,.020,.024,.028,.032,.036)):
  sid=f'v13can_{i}';row=deepcopy(template);a=deepcopy(at);row['scene_id']=sid;row['scene_family_id']=f'mh_pcrau_v3/canary_v13/sugar/apple/{i}';row['layout_id']=sid;row['poses']['ycb_apple']=[-.10,.38,0.0];row['poses']['ycb_sugar_box']=[-.186,.38+off,1.5707963267948966];sig=hashlib.sha256(json.dumps(row['poses'],sort_keys=True).encode()).hexdigest();row['layout_signature_sha256']=sig;a.update({'family_id':row['scene_family_id'],'layout_id':sid,'layout_signature_sha256':sig});a['projection_calibration'].update({'forward_separation_m':.086,'signed_lateral_offset_m':off,'occluder_yaw_rad':1.5707963267948966,'target_x_m':-.10});rows.append(row);anns[sid]=a
 scenes.update({'protocol_id':'mh_pcrau_v3_left_ycb_canary_v13_sugar_apple_near','expected_scene_count':7,'scenes':rows});ann.update({'protocol_id':scenes['protocol_id'],'scenes':anns});gate.update({'protocol_id':scenes['protocol_id'],'parent_family_count':7});OUT.mkdir(parents=True)
 for n,p in [('scenes.yaml',scenes),('annotations.yaml',ann),('gate.yaml',gate)]: (OUT/n).write_text(yaml.safe_dump(p,sort_keys=False,allow_unicode=True,width=160))
 srcs=[WORLD,Path(__file__),SCRIPT,LAUNCH,OUT/'scenes.yaml',OUT/'annotations.yaml',OUT/'gate.yaml'];base=json.loads((SRC/'CAPTURE_SOURCE_LOCK.json').read_text())['source_artifact_sha256'];base.update({str(p.relative_to(ROOT)):sha256(p) for p in srcs});lock={'schema_version':1,'protocol_id':scenes['protocol_id'],'status':'LOCKED_BEFORE_CAPTURE_AND_INFERENCE','locked_at_utc':datetime.now(timezone.utc).isoformat(),'workspace_root':str(ROOT),'source_artifact_sha256':base,'model_inventory_sha256':'NO_MODEL_INFERENCE_PROJECTION_CANARY','planned_scene_count':7,'view_joint_pose':scenes['view_joint_pose'],'world_file':str(WORLD.relative_to(ROOT))};(OUT/'CAPTURE_SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print({'status':'READY','scenes':7})
if __name__=='__main__':main()
