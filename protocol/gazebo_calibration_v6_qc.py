#!/usr/bin/env python3
"""QC using the unchanged frozen state predicate; materialize only all-PASS."""
import argparse
import inspect
import itertools
import json
import os
import textwrap
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import yaml
import materialize_gazebo_train_uq_v1 as base
import generate_gazebo_calibration_v6_design as design

ROOT=design.ROOT
sha=design.sha256
write_new=design.write_new


def state_predicate():
    source=inspect.getsource(base.materialize)
    start=source.index('        if state == "FOUND":')
    stop=source.index('        if not verified: reasons.append',start)
    body=textwrap.dedent(source[start:stop])
    # Execute the original evaluator predicate verbatim, with explicit inputs.
    wrapped='def predicate(state,item,scene,candidates,context,target,label_map,tf,camera,gate,CAPTURE,sid,min_px,tie):\n    projected=None\n'
    wrapped+=textwrap.indent(body,'    ')+'\n    return verified,projected\n'
    env=dict(base.__dict__)
    exec(compile(wrapped,'<frozen_state_predicate>','exec'),env)
    return env['predicate']


def validate_capture(pid):
    from gazebo_calibration_v6_pipeline import validate_implementation, verify_authorization
    validate_implementation(); verify_authorization(pid)
    output=ROOT/f'results/spatial_vlm_refspatial_v1/{pid}'
    capture=output/'capture_attempt_01'
    scenes,ann,gate=(yaml.safe_load(p.read_text()) for p in design.paths(pid)[:3])
    manifest=json.loads((capture/'capture_manifest.json').read_text())
    records=base.jsonl(capture/'input_manifest.jsonl')
    n=32 if pid==design.PILOT else 128
    if manifest.get('status')!='COMPLETE' or manifest.get('protocol_id')!=pid or len(records)!=n or {r['scene_id'] for r in records}!={r['scene_id'] for r in scenes['scenes']}:
        raise RuntimeError('capture incomplete or ID mismatch')
    expected={'pretrial_source_lock_sha256':sha(ROOT/f'protocol/{pid}_capture_compatibility_lock.json'),
              'scene_config_sha256':sha(design.paths(pid)[0]),'annotation_file_sha256':sha(design.paths(pid)[1]),'gate_config_file_sha256':sha(design.paths(pid)[2]),
              'input_manifest_sha256':sha(capture/'input_manifest.jsonl')}
    if any(manifest.get(k)!=v for k,v in expected.items()) or not manifest.get('source_artifacts_verified_before_capture'):
        raise RuntimeError('capture provenance mismatch')
    for record in records:
        for name,relative in record['input_files'].items():
            if name in record.get('input_sha256',{}) and sha(capture/relative)!=record['input_sha256'][name]:raise RuntimeError('captured input changed')
    return output,capture,scenes,ann,gate,records


def duplicate_qc(pid,capture,records):
    from gazebo_calibration_v6_pipeline import IMPLEMENTATION,read
    old_paths=read(IMPLEMENTATION)['prior_rgb_manifest_sha256']
    roots=[ROOT/p for p in old_paths]
    if pid==design.CALIBRATION:roots.append(ROOT/f'results/spatial_vlm_refspatial_v1/{design.PILOT}/capture_attempt_01/input_manifest.jsonl')
    current=[]
    for r in records:
        p=capture/r['input_files']['rgb']; image=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if image is None:raise RuntimeError('invalid RGB '+str(p))
        current.append((r['scene_id'],sha(p),image))
    exact=[]; near=[]; count=0
    def compare(a,b):
        if a[1]==b[1]:exact.append([a[0],b[0]])
        if a[2].shape!=b[2].shape:return
        # Full-frame check avoids a thumbnail heuristic excluding a duplicate.
        d=cv2.absdiff(a[2],b[2]); mad=float(d.mean())
        if mad<.05 and float(np.mean(d>=3))<.002:near.append({'pair':[a[0],b[0]],'gray_mad':mad,'changed_fraction':float(np.mean(d>=3))})
    for a,b in itertools.combinations(current,2):compare(a,b)
    for path in roots:
        for r in base.jsonl(path):
            p=path.parent/r['input_files']['rgb']; im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
            if im is None:raise RuntimeError('prior RGB unavailable '+str(p))
            old=(str(path.parent.relative_to(ROOT))+':'+r['scene_id'],sha(p),im); count+=1
            for new in current:compare(new,old)
    return {'status':'PASS' if not exact and not near else 'REJECT','rgb_count':len(current),'prior_rgb_count':count,
            'unique_rgb_sha256':len({r[1] for r in current}),'exact_duplicates':exact,'perceptual_near_duplicates':near,
            'rule':{'gray_mad_lt':.05,'changed_fraction_lt':.002,'changed_pixel_absdiff_ge':3},
            'source_manifest_sha256':{str(p.relative_to(ROOT)):sha(p) for p in roots}}


def qc(pid):
    output,capture,scenes,annotations,gate,records=validate_capture(pid)
    report_path=output/'GEOMETRY_QC.json'; dup_path=output/'RGB_DUPLICATE_QC.json'
    if report_path.exists() or dup_path.exists():raise FileExistsError('QC already attempted')
    sources={str(p.relative_to(ROOT)):sha(p) for p in capture.rglob('*') if p.is_file()}
    write_new(output/'QC_INPUT_LOCK.json',{'status':'LOCKED_BEFORE_QC','source_artifact_sha256':sources,'qc_code_sha256':sha(Path(__file__))})
    predicate=state_predicate(); lookup={s['scene_id']:s for s in scenes['scenes']}; rows=[]
    for record in records:
        sid=record['scene_id']; item=annotations['scenes'][sid]; scene=lookup[sid]
        labels=base.labels(capture/sid/'evaluator/semantic_labels.png')
        camera=json.loads((capture/record['input_files']['camera_info']).read_text())
        tf=json.loads((capture/record['input_files']['tf_snapshot']).read_text())['camera_color_optical_frame']
        depth=np.load(capture/record['input_files']['depth_m'],allow_pickle=False)
        reasons=base.sensor_reasons(labels,depth,camera,tf)
        candidates=[base.geom(labels,l) for l in item.get('candidate_labels',[])]; context=[base.geom(labels,l) for l in item.get('context_labels',[])]
        target=base.geom(labels,item['target_label']) if item.get('target_label') is not None else None
        passed,projected=predicate(item['state'],item,scene,candidates,context,target,labels,tf,camera,gate,capture,sid,int(gate['min_visible_evidence_px']),float(gate['tie_margin_normalized']))
        if not passed:reasons.append('REQUESTED_STATE_NOT_VERIFIED:'+item['state'])
        rows.append({'scene_id':sid,'state':item['state'],'relation_variant':item['relation_variant'],'state_verified':bool(passed),
                     'candidate_set':candidates,'context_set':context,'target':target,'projected_target':projected,'reasons':reasons})
    duplicates=duplicate_qc(pid,capture,records);write_new(dup_path,duplicates)
    passed=all(not r['reasons'] for r in rows) and duplicates['status']=='PASS'
    report={'status':'PASS' if passed else 'BLOCKED','protocol_id':pid,'records':len(rows),'passed_scene_count':sum(not r['reasons'] for r in rows),
            'verified_state_counts':dict(Counter(r['state'] for r in rows if not r['reasons'])),'scenes':rows,'rgb_qc_sha256':sha(dup_path),
            'input_lock_sha256':sha(output/'QC_INPUT_LOCK.json'),'scientific_hypothesis':'NOT_TESTED','dataset_materialized':False}
    write_new(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))
    if not passed:
        from gazebo_calibration_v6_pipeline import terminal_block
        terminal_block('PILOT_QC' if pid==design.PILOT else 'CALIBRATION_QC',[report_path,dup_path,output/'QC_INPUT_LOCK.json'],pid)
        raise SystemExit(2)


def materialize():
    pid=design.CALIBRATION
    output,capture,scenes,annotations,gate,records=validate_capture(pid)
    qc_path=output/'GEOMETRY_QC.json'; qc_report=json.loads(qc_path.read_text())
    if qc_report['status']!='PASS' or qc_report['passed_scene_count']!=128:raise RuntimeError('128/128 QC required')
    qlock=json.loads((output/'QC_INPUT_LOCK.json').read_text())
    for p,d in qlock['source_artifact_sha256'].items():
        if sha(ROOT/p)!=d:raise RuntimeError('capture drift')
    dataset=ROOT/'datasets/Gazebo_calibration_v6'
    dataset.mkdir()  # fail closed if dataset exists
    byid={r['scene_id']:r for r in qc_report['scenes']}; inference=[]; truth=[]
    for record in records:
        sid=record['scene_id']; ann=annotations['scenes'][sid]; row=byid[sid]
        dest=dataset/'records'/sid/'input';dest.mkdir(parents=True)
        for name in ('rgb','depth_m','camera_info','tf_snapshot'):
            src=capture/record['input_files'][name];os.link(src,dest/src.name)
        depth=np.load(capture/record['input_files']['depth_m'],allow_pickle=False)
        if not cv2.imwrite(str(dest/'depth_view.png'),base.depth_view(depth)):raise RuntimeError('depth view write failed')
        common={'sample_id':sid,'scene_id':sid,'family_id':ann['family_id'],'split':'calibration','relation_variant':ann['relation_variant'],
                'relation':'horizontal_ordinal_ranking_answerability','reference_frame':gate['reference_frame']}
        inference.append({**common,'image':f'records/{sid}/input/rgb.png','depth':f'records/{sid}/input/depth_view.png','metric_depth':f'records/{sid}/input/depth_m.npy',
                          'instruction':record['instruction']+' '+record['coordinate_suffix']})
        truth.append({**common,'answerability_state':ann['state'],'answerability_verified':True,'target_id':ann.get('target_id'),
                      'target_xy':row['target']['centroid_normalized_xy'] if ann['state']=='FOUND' else None})
    for name,rows in (('inference_manifest.jsonl',inference),('evaluator_ground_truth.jsonl',truth)):
        with (dataset/name).open('x') as stream:
            for row in rows:stream.write(json.dumps(row,sort_keys=True)+'\n')
    write_new(dataset/'manifest.json',{'status':'PASS','protocol_id':pid,'records':128,'parent_families':128,'qc_report_sha256':sha(qc_path),'test_iid_ood_access':False})
    paths=[qc_path,output/'QC_INPUT_LOCK.json',output/'RGB_DUPLICATE_QC.json',*dataset.rglob('*')]
    write_new(ROOT/'protocol/GAZEBO_CALIBRATION_V6_MATERIALIZATION_LOCK.json',{'status':'MATERIALIZATION_FROZEN_128_OF_128_QC_PASS',
              'source_artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},'test_iid_ood_access':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['pilot','calibration','materialize']);args=parser.parse_args()
    if args.command=='materialize':materialize()
    else:qc(design.PILOT if args.command=='pilot' else design.CALIBRATION)
