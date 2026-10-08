#!/usr/bin/env python3
"""Day-8 revision 7: semantic-safe ordinal layouts for top-table capture."""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys
from collections import Counter
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from workspace.mh_pcrau_v3 import day8_development_r2_prepare as base

RAW, REPORT = base.RAW, base.REPORT
OUT = RAW / 'canary_v7'
WORLDS = ROOT / 'workspace/mh_pcrau_v3/generated/day8_development_r2_worlds_v7'
BASE_WORLD = base.WORLD
MARKER = '<pose>-0.975085296109 0.550000000000 1.100000000000 0 1.047197551197 0</pose>'
# Tiny, actual camera perturbations retain the calibrated visibility envelope.
POSES = ('-0.975085296109 0.550000000000 1.100000000000 0 1.047197551197 0',
         '-0.974085296109 0.550000000000 1.100000000000 0 1.047197551197 0',
         '-0.975085296109 0.551000000000 1.100000000000 0 1.047197551197 0',
         '-0.975085296109 0.550000000000 1.101000000000 0 1.047197551197 0')
NAMES = {'ycb_pear':'pear','ycb_apple':'apple','ycb_orange':'orange'}
LABELS = base.FRUIT_LABELS

def worlds():
    WORLDS.mkdir(parents=True, exist_ok=True); source=BASE_WORLD.read_text()
    if source.count(MARKER)!=1: raise RuntimeError('top camera marker drift')
    out=[]
    for i, pose in enumerate(POSES):
        p=WORLDS/f'world_camera_{i}.sdf'; value=source.replace(MARKER, f'<pose>{pose}</pose>')
        if p.exists() and p.read_text()!=value: raise RuntimeError(f'world drift: {p}')
        if not p.exists(): p.write_text(value)
        out.append(p)
    return out

def family_token(stage,state,relation,repeat):
    return hashlib.sha256(f'{base.NAMESPACE}|v7|{stage}|{state}|{relation}|{repeat}'.encode()).hexdigest()[:20]

def ordered_names(relation):
    # Apple is always the insufficient target and occupies requested rank.
    return {'leftmost':['ycb_apple','ycb_pear','ycb_orange'],
            'rightmost':['ycb_pear','ycb_orange','ycb_apple'],
            'second_from_left':['ycb_pear','ycb_apple','ycb_orange'],
            'second_from_right':['ycb_pear','ycb_apple','ycb_orange']}[relation]

def normal_positions(relation, shift):
    if relation in ('leftmost','second_from_left'):
        raw=[(-.34,.63),(-.24,.51),(-.14,.39)]
    else:
        raw=[(-.34,.38),(-.24,.26),(-.14,.14)]
    return [[x,y+shift,0.] for x,y in raw]

def tie_positions(relation, shift):
    if relation in ('leftmost','second_from_right'):
        raw=[(-.31,.58),(-.17,.58),(-.24,.38)]
    else:
        raw=[(-.24,.38),(-.31,.20),(-.17,.20)]
    return [[x,y+shift,0.] for x,y in raw]

def instruction(relation,names,variant):
    return f"Among the {', '.join(NAMES[n] for n in names[:-1])} and {NAMES[names[-1]]}, {base.PHRASES[relation][variant]}."

def build(stage,state,relation,repeat):
    token=family_token(stage,state,relation,repeat); rng=random.Random(int(token,16)); names=ordered_names(relation)
    target_index=base.rank_index(relation); target=names[target_index]; shift=rng.uniform(-.015,.015)
    positions=tie_positions(relation,shift) if state=='AMBIGUOUS' else normal_positions(relation,shift)
    poses={n:[p[0],p[1],rng.uniform(-.15,.15)] for n,p in zip(names,positions)}
    if state=='ABSENT': poses.pop(target)
    # For IE use the tested sugar-box/apple pair.  Other visible candidates
    # remain away from the calibrated pair; only the target has weak evidence.
    if state=='INSUFFICIENT_EVIDENCE':
        target='ycb_apple'; target_index=names.index(target)
        poses[target]=[-.10,.38,0.]
        poses['ycb_sugar_box']=[-.186,.396,math.pi/2]
        for n,xy in zip([n for n in names if n!=target],[(-.12,.22),(-.38,.23)]): poses[n]=[xy[0],xy[1],rng.uniform(-.15,.15)]
    # All non-IE boxes are present but parked, so no label state depends on a
    # missing object and no FOUND/AMBIGUOUS target is accidentally covered.
    for j,b in enumerate(base.BOXES):
        if not (state=='INSUFFICIENT_EVIDENCE' and b=='ycb_sugar_box'):
            poses[b]=[-.48+.22*j,.98,0.]
    safe=[(-.46,.66),(-.34,.76),(-.20,.70),(-.08,.82),(-.48,.88),(-.10,.60)]
    for n,(x,y) in zip(base.CLUTTER,safe): poses[n]=[x+rng.uniform(-.012,.012),y+rng.uniform(-.012,.012),rng.uniform(-.35,.35)]
    poses['ycb_lemon']=[5.34,3.,0.]
    sid=f'd8r2_{stage}_{token}'; family=f'{base.NAMESPACE}/{stage}/{token}'; sig=hashlib.sha256(json.dumps(poses,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    split='canary' if stage=='canary_v7' else ('train' if repeat<24 else 'development_validation')
    scene={'scene_id':sid,'scene_family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'task_type':'development_r2_ordinal_grounding','instruction':instruction(relation,names,repeat%4),'poses':poses}
    ann={'state':state,'relation_variant':relation,'rank_from':'left' if 'left' in relation else 'right','rank':1 if relation in ('leftmost','rightmost') else 2,'family_id':family,'layout_id':sid,'layout_signature_sha256':sig,'split':split,'seed':int(token[:12],16),'camera_stratum':repeat%4,'prompt_variant':repeat%4,'candidate_ids':[n for n in names if n in poses],'candidate_labels':[LABELS[n] for n in names if n in poses],'failure_tags':['fresh_family','development_r2','world_v5','v7_semantic_safe']}
    if state=='FOUND': ann.update(target_id=target,target_label=LABELS[target])
    elif state=='AMBIGUOUS':
        valid=[names[i] for i in base.tie_indices(relation)]; ann.update(target_id=None,target_label=None,valid_target_ids=valid,valid_target_labels=[LABELS[n] for n in valid])
    elif state=='ABSENT': ann.update(target_id=None,target_label=None,requested_absent_target_id=target,requested_absent_target_label=LABELS[target])
    else: ann.update(target_id=target,target_label=LABELS[target],occluder_ids=['ycb_sugar_box'],occluder_labels=[base.BOX_LABELS['ycb_sugar_box']],insufficient_reason='target_visible_pixels_1_to_119_due_to_controlled_ycb_occlusion',projection_calibration={'forward_separation_m':.086,'signed_lateral_offset_m':.016,'occluder_yaw_rad':math.pi/2})
    return scene,ann,{'scene_id':sid,'family_id':family,'state':state,'relation':relation,'split':split,'seed':ann['seed'],'camera_stratum':ann['camera_stratum'],'prompt_variant':ann['prompt_variant'],'layout_signature_sha256':sig}

def lock(folder,world):
    p=folder/'CAPTURE_SOURCE_LOCK.json'; x=json.loads(p.read_text()); x['source_artifact_sha256'][str(Path(__file__).relative_to(ROOT))]=base.sha256(Path(__file__)); x['source_artifact_sha256'][str(world.relative_to(ROOT))]=base.sha256(world); x['world_file']=str(world.relative_to(ROOT)); x['camera_randomization']='four measured top-table poses within calibrated visibility envelope'; p.write_text(json.dumps(x,indent=2)+'\n')

def prepare(stage):
    g=json.loads(base.GEOMETRY.read_text()); ss=yaml.safe_load(base.SOURCE_SCENES.read_text()); sa=yaml.safe_load(base.SOURCE_ANNOTATIONS.read_text()); sg=yaml.safe_load(base.SOURCE_GATE.read_text()); ws=worlds()
    if stage=='canary':
        if OUT.exists(): raise FileExistsError(OUT)
        if json.loads((REPORT/'CANARY_QC.json').read_text()).get('status')!='FAIL': raise RuntimeError('requires preserved failed v6 canary')
        OUT.mkdir(); allidx=[]
        for i,r in enumerate(base.RELATIONS):
            rows=[]; anns={}; idx=[]
            for s in base.STATES:
                row,ann,item=build('canary_v7',s,r,i); rows.append(row);anns[row['scene_id']]=ann;idx.append(item);allidx.append(item)
            f=OUT/f'camera_{i}';base.write_batch(f,f'canary_v7_c{i}',base.CAMERAS[0],rows,anns,ss,sa,sg);lock(f,ws[i]);(f/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in idx))
        (OUT/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in allidx));print(json.dumps({'status':'CANARY_V7_READY','families':16},indent=2));return
    if json.loads((REPORT/'CANARY_QC.json').read_text()).get('status')!='PASS': raise RuntimeError('bulk sealed')
    bulk=RAW/'bulk_v7'
    if bulk.exists(): raise FileExistsError(bulk)
    bulk.mkdir(); packs={i:([],{},[]) for i in range(4)};allidx=[]
    for s in base.STATES:
      for r in base.RELATIONS:
       for repeat in range(32):
        row,ann,item=build('bulk_v7',s,r,repeat);i=repeat%4;packs[i][0].append(row);packs[i][1][row['scene_id']]=ann;packs[i][2].append(item);allidx.append(item)
    for i,(rows,anns,idx) in packs.items():
      random.Random(base.SEED+i).shuffle(rows);f=bulk/f'batch_{i}';base.write_batch(f,f'bulk_v7_b{i}',base.CAMERAS[0],rows,anns,ss,sa,sg);lock(f,ws[i]);(f/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in idx))
    (bulk/'DESIGN_INDEX.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in allidx)); cells=Counter((x['state'],x['relation']) for x in allidx);(bulk/'DESIGN_STATIC_QC.json').write_text(json.dumps({'families':512,'cells':{f'{s}|{r}':cells[(s,r)] for s in base.STATES for r in base.RELATIONS}},indent=2)+'\n');print(json.dumps({'status':'BULK_V7_READY','families':512},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('canary','bulk'));prepare(p.parse_args().stage)
