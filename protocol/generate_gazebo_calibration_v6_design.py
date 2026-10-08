#!/usr/bin/env python3
"""Construct v6 once after its analytic PASS; no rendering or model access."""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import yaml
import certify_gazebo_calibration_v6_prerender as certificate
import generate_gazebo_calibration_v3_contract as v3
import generate_gazebo_calibration_v4_design as v4

ROOT=certificate.ROOT
LOCK=certificate.LOCK
PILOT='gazebo_calibration_v6_geometry_pilot'
CALIBRATION='gazebo_calibration_v6'
AUDIT=certificate.OUT/'IMPLEMENTATION_DESIGN_AUDIT.json'
REGISTRY=certificate.OUT/'PRIOR_EXCLUSION_REGISTRY.json'
sha256=certificate.sha256
write_new=certificate.write_new


def paths(pid):
    return [ROOT/f'ur3/ur3_perception/config/{pid}_{suffix}.yaml' for suffix in ('scenes','annotations','gate')]+[ROOT/f'protocol/{pid}_family_manifest.jsonl',ROOT/f'protocol/{pid}_split_manifest.json']


def verify_certificate():
    lock=certificate.load_lock()
    frozen=json.loads(certificate.FREEZE.read_text())
    if not frozen['scene_generation_authorized'] or json.loads(certificate.RESULT.read_text())['status']!='PASS':
        raise RuntimeError('pre-render PASS required')
    for name,digest in frozen['source_artifact_sha256'].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError('certificate drift: '+name)
    return lock


def construct(pid,repetitions,lock):
    source=yaml.safe_load(v3.SCENES_PATH.read_text())
    oracle=yaml.safe_load(v3.ANNOTATIONS_PATH.read_text())
    gate=yaml.safe_load(v3.GATE_PATH.read_text())
    templates={}
    for row in source['scenes']:
        a=oracle['scenes'][row['scene_id']]
        templates[a['state'],a['relation_variant'],a['cell_repetition']]=(row,a)
    tf=json.loads(certificate.TF_PATH.read_text())['camera_color_optical_frame']
    origin,rotation=tf['position'],certificate.geometry.rotation(tf['orientation_xyzw'])
    camera=json.loads((ROOT/'protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json').read_text())['camera_and_geometry']['camera']
    population=lock['population_and_independence']; geo=lock['exact_geometry_revision']; tables=geo['all_state_anchor_tables']
    split='development_geometry_only' if pid==PILOT else 'calibration'
    rows=[]; anns={}; families=[]
    for si,state in enumerate(population['states_in_order']):
        for ri,relation in enumerate(population['relations_in_order']):
            for repetition in range(repetitions):
                index=(si*4+ri)*repetitions+repetition
                sid=f'{pid}_{index:03d}'; family=f"spatial_vlm_{pid}/{'development' if pid==PILOT else 'calibration'}/parent_{index:03d}"
                layout=f'{pid}_layout_{index:03d}'
                h=int(hashlib.sha256(f'{pid}_seed_v1|{family}|{state}|{relation}|{repetition}'.encode()).hexdigest()[:8],16)
                template,ann=templates[state,relation,repetition]
                row=copy.deepcopy(template); ann=copy.deepcopy(ann)
                if state=='INSUFFICIENT_EVIDENCE':
                    names=['ycb_apple','ycb_orange']; anchors=geo['relation_aware_context_projected_center_targets_normalized'][relation]
                    poses={n:copy.deepcopy(template['poses'][n]) for n in ('mango','uq_neutral_occluder')}
                elif state=='FOUND':
                    names=[None]*3; slot=ann['rank']-1 if ann['rank_from']=='left' else 3-ann['rank']
                    names[slot]=ann['target_id']; others=iter(sorted(set(ann['candidate_ids'])-{ann['target_id']}))
                    names=[n if n is not None else next(others) for n in names]
                    anchors=tables['FOUND']; poses={}
                elif state=='AMBIGUOUS':
                    names=sorted(ann['valid_target_ids'])+sorted(set(ann['candidate_ids'])-set(ann['valid_target_ids']))
                    anchors=tables['AMBIGUOUS_TIE_'+('LEFT' if relation in ('leftmost','second_from_left') else 'RIGHT')]; poses={}
                else:
                    names=sorted(ann['context_ids']); anchors=tables['ABSENT']; poses={}
                for i,(name,anchor) in enumerate(zip(names,anchors)):
                    u=anchor[0]+.0040*(((h>>(8*i))%5)-2); v=anchor[1]+.0050*(((h>>(8*i+3))%5)-2)
                    x,y,_=certificate.geometry.inverse(u,v,source['objects'][name]['z'],origin,rotation,camera['intrinsics'])
                    poses[name]=[x,y,.0700+.0070*((h>>(16+3*i))%11)]
                row.update(scene_id=sid,scene_family_id=family,layout_id=layout,poses=poses)
                signature=v3.layout_signature(row,source['objects']); row['layout_signature_sha256']=signature
                ann.update(family_id=family,layout_id=layout,layout_signature_sha256=signature,split=split,seed=h,cell_repetition=repetition,
                           geometry_provenance='frozen_v6_analytic_certified_anchors_no_outcome_selection')
                rows.append(row); anns[sid]=ann
                families.append(dict(family_index=index,scene_id=sid,family_id=family,layout_id=layout,layout_signature_sha256=signature,
                                     split=split,state=state,relation_variant=relation,cell_repetition=repetition,deterministic_seed=h,
                                     capture_order_sha256=hashlib.sha256(f'{pid}_capture_order_v1|{family}'.encode()).hexdigest()))
    for i,f in enumerate(sorted(families,key=lambda f:f['capture_order_sha256'])):f['capture_order']=i
    order={f['scene_id']:f['capture_order'] for f in families}; rows.sort(key=lambda r:order[r['scene_id']])
    source.update(protocol_id=pid,contract_lock_sha256=sha256(LOCK),expected_scene_count=len(rows),scenes=rows,capture_order='sha256_sort_v6_family_id')
    oracle.update(protocol_id=pid,contract_lock_sha256=sha256(LOCK),scenes=anns,oracle_usage='geometry_only' if pid==PILOT else 'calibration_after_raw_prediction_lock')
    gate.update(protocol_id=pid,contract_lock_sha256=sha256(LOCK),parent_family_count=len(rows),split_parent_family_count={split:len(rows)},
                state_quota={split:{s:len(rows)//4 for s in population['states_in_order']}},
                relation_variant_quota={split:{r:len(rows)//4 for r in population['relations_in_order']}},state_relation_cell_quota={split:repetitions})
    gate['policies']['no_model_inference']=pid==PILOT; gate['policies']['no_materialization']=pid==PILOT
    gate['policies']['no_materialization_before_128_of_128_geometry_pass']=True
    gate['calibration_protocol']['fit_split']='Gazebo_calibration_v6'
    return source,oracle,gate,families


def prior_registry():
    values={k:set() for k in ('scene_id','family_id','layout_id','deterministic_seed','layout_signature_sha256')}
    sources={}
    def add(row):
        for key in values:
            val=row.get(key)
            if val is not None:values[key].add(val)
        for src,dst in (('scene_family_id','family_id'),('seed','deterministic_seed')):
            if row.get(src) is not None:values[dst].add(row[src])
    for path in sorted((ROOT/'protocol').glob('gazebo_*family_manifest.jsonl')):
        if 'test' in path.name.lower() or 'calibration_v6' in path.name:continue
        sources[str(path.relative_to(ROOT))]=sha256(path)
        for line in path.read_text().splitlines():
            if line:add(json.loads(line))
    configs=sorted((ROOT/'ur3/ur3_perception/config').glob('*_scenes.yaml'))
    for path in configs:
        if 'test' in path.name.lower() or 'calibration_v6' in path.name:continue
        if not (path.name.startswith('gazebo') or path.name.startswith('roborefer_pilot')):continue
        sources[str(path.relative_to(ROOT))]=sha256(path)
        cfg=yaml.safe_load(path.read_text()) or {}
        for row in cfg.get('scenes',[]):
            add(row)
            if row.get('poses'):values['layout_signature_sha256'].add(v3.layout_signature(row,cfg.get('objects',{})))
        ap=path.with_name(path.name.replace('_scenes.yaml','_annotations.yaml'))
        if ap.exists():
            sources[str(ap.relative_to(ROOT))]=sha256(ap)
            for row in (yaml.safe_load(ap.read_text()) or {}).get('scenes',{}).values():add(row)
    # V4 never wrote manifests; reconstruct only its immutable, unrendered
    # preview for exclusion, never as calibration data.
    old,_=v4.generate()
    for plan in old.values():
        for f in plan[3]:add(f)
    sources['protocol/generate_gazebo_calibration_v4_design.py']=sha256(Path(v4.__file__))
    sources['results/spatial_vlm_refspatial_v1/gazebo_calibration_v4/STATIC_DESIGN_AUDIT.json']=sha256(ROOT/'results/spatial_vlm_refspatial_v1/gazebo_calibration_v4/STATIC_DESIGN_AUDIT.json')
    return values,sources


def main():
    if any(p.exists() for pid in (PILOT,CALIBRATION) for p in paths(pid)) or AUDIT.exists() or REGISTRY.exists():
        raise FileExistsError('v6 generation already attempted')
    lock=verify_certificate()
    plans={PILOT:construct(PILOT,2,lock),CALIBRATION:construct(CALIBRATION,8,lock)}
    prior,sources=prior_registry()
    current={k:[] for k in prior}; split_checks={}
    for pid,(scenes,anns,gate,families) in plans.items():
        for f in families:
            for k in current:current[k].append(f[k])
        cells=Counter((f['state'],f['relation_variant']) for f in families)
        split_checks[pid]={'families':len(families),'cells':{s+'|'+r:n for (s,r),n in sorted(cells.items())},
                           'quota_pass':len(cells)==16 and set(cells.values())==({2} if pid==PILOT else {8})}
    overlap={k:sorted(set(v)&prior[k]) for k,v in current.items()}
    checks={'certificate_pass':True,'exact_quotas':all(s['quota_pass'] for s in split_checks.values()),
            'all_160_identifiers_seeds_layouts_unique':all(len(set(v))==160 for v in current.values()),
            'zero_prior_overlap':all(not v for v in overlap.values()),
            'test_not_opened':True,'no_capture_or_inference':True}
    registry={'schema_version':1,'source_artifact_sha256':sources,'identifiers':{k:sorted(v) for k,v in prior.items()},
              'counts':{k:len(v) for k,v in prior.items()},'v4_static_preview_included':True,'v5_scenes_existed':False,'test_data_opened':False}
    write_new(REGISTRY,registry)
    audit={'status':'PASS' if all(checks.values()) else 'BLOCKED_IMPLEMENTATION_DESIGN','checks':checks,'splits':split_checks,
           'overlaps':overlap,'prior_counts':registry['counts'],'data_design_lock_sha256':sha256(LOCK),
           'pre_render_certificate_sha256':sha256(certificate.RESULT),'generator_sha256':sha256(Path(__file__)),
           'exclusion_registry_sha256':sha256(REGISTRY),'capture_authorized':False,'test_iid_ood_access':False}
    write_new(AUDIT,audit)
    if not all(checks.values()):
        raise SystemExit('STOP: implementation design audit failed; no scene files written')
    for pid,(scenes,anns,gate,families) in plans.items():
        text=[yaml.safe_dump(o,sort_keys=False,width=140) for o in (scenes,anns,gate)]
        text.append(''.join(json.dumps(f,sort_keys=True)+'\n' for f in families))
        text.append(json.dumps({'protocol_id':pid,'status':'FROZEN_DESIGN_INPUT','families':len(families),'contract_lock_sha256':sha256(LOCK),
                               'test_iid_ood_sealed':True,'capture_authorized':False},indent=2)+'\n')
        for path,data in zip(paths(pid),text):
            with path.open('x',encoding='utf-8') as stream:stream.write(data)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
