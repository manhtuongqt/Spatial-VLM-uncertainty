#!/usr/bin/env python3
"""One immutable, all-state analytic certificate; no scenes, seeds or renders."""
from __future__ import annotations

import itertools
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml
import certify_gazebo_calibration_v5_prerender as geometry

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'protocol/GAZEBO_CALIBRATION_V6_DATA_DESIGN_LOCK.json'
EXPECTED_SHA256 = '850921e200d921592128c76e11c8deeb5a7d6177ea315f1ff8d286f873084b6d'
OUT = ROOT / 'results/spatial_vlm_refspatial_v1/gazebo_calibration_v6'
RESULT = OUT / 'PRE_RENDER_INTERVAL_CERTIFICATE.json'
FREEZE = ROOT / 'protocol/GAZEBO_CALIBRATION_V6_PRE_RENDER_CERTIFICATE_LOCK.json'
TF_PATH = ROOT / 'results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01/gazebo_calibration_v3_100/input/tf_snapshot.json'
WORLD = ROOT / 'ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf'
sha256 = geometry.sha256
PAD = 1e-12
MARGIN = 0.010


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n')


def load_lock():
    if sha256(LOCK) != EXPECTED_SHA256:
        raise RuntimeError('v6 lock hash drift')
    lock = json.loads(LOCK.read_text())
    for group in ('predecessor', 'frozen_sources_sha256'):
        for name, digest in lock[group].items():
            if sha256(ROOT/name) != digest:
                raise RuntimeError('frozen source drift: '+name)
    return lock


def compute():
    lock = load_lock()
    world = ET.parse(WORLD)
    radii = {}
    for name in ('ycb_apple', 'ycb_orange', 'mango'):
        radii[name] = float(world.find(f".//model[@name='{name}']/link/collision/geometry/sphere/radius").text)
    table_pose = list(map(float, world.find(".//model[@name='work_table']/pose").text.split()))
    table_size = list(map(float, world.find(".//model[@name='work_table']/link/collision/geometry/box/size").text.split()))
    table = {axis: (table_pose[i]-table_size[i]/2, table_pose[i]+table_size[i]/2) for i, axis in enumerate(('x','y'))}
    table_top = table_pose[2]+table_size[2]/2
    panel_size = list(map(float, world.find(".//model[@name='uq_neutral_occluder']/link/visual/geometry/box/size").text.split()))
    parent = json.loads((ROOT/'protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json').read_text())
    k = parent['camera_and_geometry']['camera']['intrinsics']
    tf = json.loads(TF_PATH.read_text())['camera_color_optical_frame']
    origin, orient = tf['position'], geometry.rotation(tf['orientation_xyzw'])
    scenes = yaml.safe_load((ROOT/'ur3/ur3_perception/config/gazebo_calibration_v3_scenes.yaml').read_text())
    ann = yaml.safe_load((ROOT/'ur3/ur3_perception/config/gazebo_calibration_v3_annotations.yaml').read_text())['scenes']
    heights = {name: scenes['objects'][name]['z'] for name in radii}
    templates = [r for r in scenes['scenes'] if ann[r['scene_id']]['state']=='INSUFFICIENT_EVIDENCE']
    targets = {tuple(r['poses']['mango']) for r in templates}
    panels = {tuple(r['poses']['uq_neutral_occluder']) for r in templates}
    if len(targets)!=1 or len(panels)!=1 or len(templates)!=32:
        raise RuntimeError('IE target/panel template inconsistent')
    target, panel = next(iter(targets)), next(iter(panels))
    target_uv = geometry.forward((target[0],target[1],heights['mango']),origin,orient,k)
    panel_uv = geometry.forward((panel[0],panel[1],scenes['objects']['uq_neutral_occluder']['z']),origin,orient,k)
    c,s=abs(math.cos(panel[2])),abs(math.sin(panel[2]))
    half=((c*panel_size[0]+s*panel_size[1])/2,(s*panel_size[0]+c*panel_size[1])/2)
    panel_box={axis:(panel[i]-half[i]-PAD,panel[i]+half[i]+PAD) for i,axis in enumerate(('x','y'))}

    def bound(name, anchor, fixed=False):
        du,dv=(0.,0.) if fixed else (.008,.010)
        urange=(anchor[0]-du,anchor[0]+du); vrange=(anchor[1]-dv,anchor[1]+dv)
        corners=list(itertools.product(urange,vrange))
        positions=[geometry.inverse(u,v,heights[name],origin,orient,k) for u,v in corners]
        box={axis:(min(p[i] for p in positions)-PAD,max(p[i] for p in positions)+PAD) for i,axis in enumerate(('x','y'))}
        roundtrip=max(abs(geometry.forward((p[0],p[1],heights[name]),origin,orient,k)[i]-uv[i]) for p,uv in zip(positions,corners) for i in (0,1))
        table_margin=min(box[axis][0]-table[axis][0]-radii[name] for axis in ('x','y'))
        table_margin=min(table_margin,*(table[axis][1]-box[axis][1]-radii[name] for axis in ('x','y')))
        checks={'image_center_guard':.08<=urange[0]<=urange[1]<=.92 and .08<=vrange[0]<=vrange[1]<=.92,
                'table_edge_margin_10mm':table_margin>=MARGIN,
                'sphere_supported_by_table':abs(heights[name]-radii[name]-table_top)<=1e-6,
                'projection_roundtrip':roundtrip<=1e-10,
                'depth_range':all(.1<=p[2]<=2. for p in positions)}
        return {'name':name,'u':urange,'v':vrange,'xy':box,'depth_range':[min(p[2] for p in positions),max(p[2] for p in positions)],
                'table_clearance_lower_bound_m':table_margin,'checks':checks}

    def image_distance(a,b):
        return math.hypot(geometry.interval_gap(a['u'],b['u']),geometry.interval_gap(a['v'],b['v']))

    rows=[]
    def check_class(name, names, anchors, state, relation=None):
        candidates=[bound(n,a) for n,a in zip(names,anchors)]
        if state=='INSUFFICIENT_EVIDENCE':
            candidates.append(bound('mango',target_uv[:2],fixed=True))
        reasons=[r['name']+':'+key for r in candidates for key,passed in r['checks'].items() if not passed]
        pairs=[]
        for a,b in itertools.combinations(candidates,2):
            clearance=geometry.rectangle_distance(a['xy'],b['xy'])-radii[a['name']]-radii[b['name']]
            pairs.append({'pair':[a['name'],b['name']],'clearance_lower_bound_m':clearance})
            if clearance<MARGIN: reasons.append('pair_margin_10mm:'+a['name']+':'+b['name'])
        panel_bounds=[]
        if state=='INSUFFICIENT_EVIDENCE':
            a,b,t=candidates
            for item in (a,b):
                clearance=geometry.rectangle_distance(item['xy'],panel_box)-radii[item['name']]
                panel_bounds.append({'candidate':item['name'],'clearance_lower_bound_m':clearance})
                if clearance<MARGIN: reasons.append('panel_margin_10mm:'+item['name'])
                pseudo={'u':(panel_uv[0],panel_uv[0]),'v':(panel_uv[1],panel_uv[1])}
                if image_distance(item,t)<.08 or image_distance(item,pseudo)<.08: reasons.append('target_or_panel_image_separation')
            if image_distance(a,b)<.12: reasons.append('candidate_image_separation')
            u=target_uv[0]
            ordered={'leftmost':u<min(a['u'][0],b['u'][0]),'rightmost':u>max(a['u'][1],b['u'][1]),
                     'second_from_left':a['u'][1]<u<b['u'][0],'second_from_right':a['u'][1]<u<b['u'][0]}[relation]
            if not ordered: reasons.append('relation_order')
        elif state=='FOUND':
            if any(b['u'][0]-a['u'][1]<=.03 for a,b in zip(candidates,candidates[1:])): reasons.append('found_rank_separation')
        elif state=='AMBIGUOUS':
            a,b,t=candidates
            gap=max(abs(a['u'][0]-b['u'][1]),abs(a['u'][1]-b['u'][0]))
            if gap>.03: reasons.append('ambiguous_tie_margin')
            side=(t['u'][0]-max(a['u'][1],b['u'][1])) if relation=='left' else (min(a['u'][0],b['u'][0])-t['u'][1])
            if side<=.03: reasons.append('ambiguous_side_rank')
        rows.append({'geometry_class':name,'state':state,'candidates':candidates,'pairs':pairs,'panel_clearances':panel_bounds,'failures':reasons})

    g=lock['exact_geometry_revision']; tables=g['all_state_anchor_tables']
    for relation,anchors in g['relation_aware_context_projected_center_targets_normalized'].items():
        check_class('IE:'+relation,['ycb_apple','ycb_orange'],anchors,'INSUFFICIENT_EVIDENCE',relation)
    for names in itertools.permutations(radii):
        check_class('FOUND:'+','.join(names),names,tables['FOUND'],'FOUND')
    for side in ('left','right'):
        check_class('AMBIGUOUS:'+side,['ycb_apple','ycb_orange','mango'],tables['AMBIGUOUS_TIE_'+side.upper()],'AMBIGUOUS',side)
    for names in itertools.permutations(radii,2):
        check_class('ABSENT:'+','.join(names),names,tables['ABSENT'],'ABSENT')
    checks={
        'all_source_hashes_verified':True,
        'fixed_camera_frames':tf['parent_frame']=='base_link' and tf['child_frame']=='camera_color_optical_frame',
        'sdf_radii_match_lock':radii==lock['observed_pre_render_failure']['collision_radii_m'],
        'all_18_geometry_classes_certified':len(rows)==18 and all(not row['failures'] for row in rows),
        'no_v6_scene_or_manifest_generated':all(not path.exists() for suffix in ('gazebo_calibration_v6','gazebo_calibration_v6_geometry_pilot') for path in (ROOT/f'ur3/ur3_perception/config/{suffix}_scenes.yaml',ROOT/f'protocol/{suffix}_family_manifest.jsonl')),
        'no_v6_capture_or_dataset':all(not (ROOT/f'results/spatial_vlm_refspatial_v1/{s}/capture_attempt_01').exists() for s in ('gazebo_calibration_v6','gazebo_calibration_v6_geometry_pilot')) and not (ROOT/'datasets/Gazebo_calibration_v6').exists(),
        'test_robot_sealed_by_contract':lock['policy']['test_iid_ood_access'] is False and lock['policy']['robot_access'] is False,
    }
    return {'schema_version':1,'protocol_id':'gazebo_calibration_v6','checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'PASS' if all(checks.values()) else 'BLOCKED_PRE_RENDER_GEOMETRY','checks':checks,
            'proof_method':'All asset permutations; continuous full jitter rectangles; affine-over-affine inverse projection with signed ray denominator, corner extrema, outward XY padding 1e-12m, conservative AABB separation using actual SDF collision radii. No generated scenes or sampled seed selection.',
            'minimum_required_margin_m':MARGIN,'classes':rows,'collision_radii_m':radii,'table_xy_bounds_m':table,
            'data_design_lock_sha256':sha256(LOCK),'certificate_code_sha256':sha256(Path(__file__)),
            'scientific_hypothesis':'NOT_TESTED','scientific_decision':None,'pilot_families_captured':0,'calibration_families_captured':0,
            'model_inference_run':False,'calibrator_fit_call_count':0,'test_iid_ood_access':False,'robot_access':False}


def main():
    if RESULT.exists() or FREEZE.exists():
        raise FileExistsError('immutable v6 certificate already exists; no rerun')
    report=compute()
    write_new(RESULT,report)
    sources=(LOCK,Path(__file__).resolve(),Path(geometry.__file__).resolve(),RESULT)
    write_new(FREEZE,{'protocol_id':'gazebo_calibration_v6','status':'PRE_RENDER_CERTIFICATE_FROZEN_'+report['status'],
                     'locked_at_utc':datetime.now(timezone.utc).isoformat(),
                     'source_artifact_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
                     'scene_generation_authorized':report['status']=='PASS','capture_authorized':False,
                     'test_iid_ood_access':False,'robot_access':False})
    if report['status']!='PASS':
        write_new(OUT/'CALIBRATION_V6_FINAL_DECISION.json',{
            'protocol_id':'gazebo_calibration_v6','status':'BLOCKED',
            'decision':'CALIBRATION_V6_BLOCKED_BEFORE_SCIENTIFIC_EVALUATION',
            'blocked_gate':'PRE_RENDER_GEOMETRY','classification':'DATA_DESIGN_FAILURE_NOT_CALIBRATOR_NEGATIVE',
            'scientific_hypothesis':'NOT_TESTED','scientific_decision':None,'raw_metrics':None,'calibrated_metrics':None,'selected_threshold':None,
            'pilot_families_captured':0,'calibration_families_captured':0,'calibrator_fit_call_count':0,
            'source_artifact_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in (*sources,FREEZE)},
            'test_iid_ood_access':False,'robot_access':False})
    print(json.dumps({'status':report['status'],'checks':report['checks'],
                      'failures':{r['geometry_class']:r['failures'] for r in report['classes'] if r['failures']},
                      'minimum_pair_clearance_m':min(p['clearance_lower_bound_m'] for r in report['classes'] for p in r['pairs']),
                      'minimum_panel_clearance_m':min(p['clearance_lower_bound_m'] for r in report['classes'] for p in r['panel_clearances']),
                      'certificate_sha256':sha256(RESULT),'freeze_sha256':sha256(FREEZE)},indent=2))
    if report['status']!='PASS':raise SystemExit(2)


if __name__=='__main__':main()
