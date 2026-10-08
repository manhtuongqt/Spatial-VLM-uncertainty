#!/usr/bin/env python3
"""Append-only v6 implementation freeze and strictly ordered data gates."""
import argparse
import ast
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml
import generate_gazebo_calibration_v6_design as design
import gazebo_calibration_v3_pipeline_r4 as helpers
import gazebo_calibration_v3_pipeline_r5 as environment
import gazebo_calibration_v3_pipeline_r6 as parsers

ROOT=design.ROOT
RESULT=design.certificate.OUT
IMPLEMENTATION=ROOT/'protocol/GAZEBO_CALIBRATION_V6_IMPLEMENTATION_LOCK.json'
sha=design.sha256
write_new=design.write_new
WORLD=design.certificate.WORLD


def read(path):return json.loads(path.read_text())


def source_closure(paths):
    paths=set(paths); pending=list(paths)
    while pending:
        path=pending.pop()
        if path.suffix!='.py':continue
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                dep=ROOT/'protocol'/(name.split('.')[0]+'.py')
                if dep.is_file() and dep not in paths:paths.add(dep);pending.append(dep)
    return paths


def freeze():
    design.verify_certificate()
    if IMPLEMENTATION.exists():raise FileExistsError('implementation already frozen')
    if read(design.AUDIT)['status']!='PASS':raise RuntimeError('design audit must PASS')
    paths=[design.LOCK,design.certificate.FREEZE,design.certificate.RESULT,design.AUDIT,design.REGISTRY,WORLD,Path(__file__),
           ROOT/'protocol/gazebo_calibration_v6_qc.py',ROOT/'protocol/gazebo_calibration_v6_infer.py',ROOT/'protocol/gazebo_calibration_v6_fit.py',
           *design.paths(design.PILOT),*design.paths(design.CALIBRATION)]
    paths+= [ROOT/p for p in environment.CAPTURE_REQUIRED_SOURCES]
    paths+= [ROOT/p for p in read(design.LOCK)['frozen_sources_sha256']]
    paths+= [ROOT/'ur3/ur3_perception/launch/roborefer_uq_capture.launch.py']
    paths=source_closure(paths)
    prior={}
    for folder in sorted((ROOT/'results/spatial_vlm_refspatial_v1').iterdir()):
        if not folder.is_dir() or 'test' in folder.name.lower() or 'calibration_v6' in folder.name:continue
        if not (folder.name.startswith('gazebo') or folder.name.startswith('roborefer')):continue
        for path in folder.glob('**/input_manifest.jsonl'):
            prior[str(path.relative_to(ROOT))]=sha(path)
    registry=read(design.REGISTRY)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    hashes.update(registry['source_artifact_sha256'])
    write_new(IMPLEMENTATION,{'schema_version':1,'status':'IMPLEMENTATION_FROZEN_BEFORE_PREFLIGHT','protocol_id':design.CALIBRATION,
              'locked_at_utc':datetime.now(timezone.utc).isoformat(),'source_artifact_sha256':hashes,'prior_rgb_manifest_sha256':prior,
              'design_audit_sha256':sha(design.AUDIT),'data_design_sha256':sha(design.LOCK),
              'capture_authorized':False,'test_iid_ood_access':False,'robot_access':False,
              'capture_execution':'Use already preflighted simulation; invoke frozen capture node directly once per authorized split, no duplicate simulation launch.',
              'calibration_parent_contract':'protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json'})
    print(json.dumps({'status':'FROZEN','sha256':sha(IMPLEMENTATION),'source_count':len(hashes),'prior_rgb_manifests':len(prior)},indent=2))


def validate_implementation():
    design.verify_certificate(); lock=read(IMPLEMENTATION)
    if lock['status']!='IMPLEMENTATION_FROZEN_BEFORE_PREFLIGHT':raise RuntimeError('implementation status')
    for group in ('source_artifact_sha256','prior_rgb_manifest_sha256'):
        for name,digest in lock[group].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('source drift: '+name)
    return lock


def terminal_block(gate,paths,pid=None):
    counts={}
    for split in (design.PILOT,design.CALIBRATION):
        manifest=ROOT/f'results/spatial_vlm_refspatial_v1/{split}/capture_attempt_01/input_manifest.jsonl'
        counts[split]=len([l for l in manifest.read_text().splitlines() if l]) if manifest.exists() else 0
    paths=[design.LOCK,design.certificate.RESULT,IMPLEMENTATION,*paths]
    write_new(RESULT/'CALIBRATION_V6_FINAL_DECISION.json',{'status':'BLOCKED','decision':'CALIBRATION_V6_BLOCKED_BEFORE_SCIENTIFIC_EVALUATION',
              'blocked_gate':gate,'split':pid,'classification':'UPSTREAM_GATE_FAILURE_NOT_CALIBRATOR_SCIENTIFIC_NEGATIVE',
              'scientific_hypothesis':'NOT_TESTED','scientific_decision':None,'raw_metrics':None,'calibrated_metrics':None,'selected_threshold':None,
              'captured_family_counts':counts,'calibrator_fit_call_count':0,'test_iid_ood_access':False,'robot_access':False,
              'source_artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}})


def static():
    validate_implementation()
    checks={'implementation_hashes':True,'design_audit_pass':read(design.AUDIT)['status']=='PASS','pre_render_pass':read(design.certificate.RESULT)['status']=='PASS',
            'no_capture':all(not (ROOT/f'results/spatial_vlm_refspatial_v1/{p}/capture_attempt_01').exists() for p in (design.PILOT,design.CALIBRATION)),
            'no_dataset':not (ROOT/'datasets/Gazebo_calibration_v6').exists(),'disk_free_over_2gib':shutil.disk_usage(ROOT).free>2*1024**3,
            'test_robot_sealed':all(yaml.safe_load(design.paths(p)[2].read_text())['sealed'][s] is True for p in (design.PILOT,design.CALIBRATION) for s in ('gazebo_test_iid','gazebo_test_ood','robot'))}
    path=RESULT/'PREFLIGHT_STATIC.json';write_new(path,{'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,'implementation_sha256':sha(IMPLEMENTATION)})
    print(json.dumps(read(path),indent=2))
    if not all(checks.values()):terminal_block('STATIC_PREFLIGHT',[path]);raise SystemExit(2)


def live(pid):
    validate_implementation()
    static_path=RESULT/'PREFLIGHT_STATIC.json'
    if read(static_path)['status']!='PASS':raise RuntimeError('static PASS required')
    path=ROOT/f'results/spatial_vlm_refspatial_v1/{pid}/PREFLIGHT_LIVE.json'
    if path.exists():raise FileExistsError('live preflight already attempted')
    helpers.run_probe=environment.run_probe;parsers.install_parser_repair()
    run=environment.run_probe
    gate=yaml.safe_load(design.paths(pid)[2].read_text());scenes=yaml.safe_load(design.paths(pid)[0].read_text())
    topics=run(['ros2','topic','list','-t']);nodes=run(['ros2','node','list']);services=run(['ros2','service','list','-t'])
    observed={'topics':topics,'nodes':nodes,'services':services}
    checks={'required_topics':all(f'{n} [{t}]' in topics['stdout'] for n,t in helpers.REQUIRED_TOPICS.items()),
            'sim_nodes':all(n in nodes['stdout'] for n in ('/ros_gz_bridge','/robot_state_publisher','/controller_manager')),
            'reset_service':helpers.SET_POSE_SERVICE in services['stdout'],
            'no_capture':not (path.parent/'capture_attempt_01').exists(),'implementation_hashes':True}
    # Inspect simulator executable command line, excluding shell command text.
    processes=run(['ps','-eo','comm=,args='])
    world_lines=[line for line in processes['stdout'].splitlines() if line.split()[0] in ('ruby','ign','gz','gzserver') and WORLD.name in line]
    observed['world_processes']=world_lines;checks['locked_world_running']=bool(world_lines)
    if checks['required_topics'] and checks['reset_service']:
        for prefix,topic,frame in [('wrist','/wrist_camera','camera_color_optical_frame'),('base','/top_table_camera','top_table_camera_optical_frame')]:
            for label,suffix,encoding in [('rgb','/color/image_raw',{'rgb8','bgr8'}),('depth','/depth/image_raw',{'32FC1'})]:
                sample=helpers.image_probe(topic+suffix);observed[prefix+'_'+label]=sample
                checks[prefix+'_'+label]=helpers.image_matches(sample,width=640,height=480,encodings=encoding,frame_id=frame)
            info=parsers.camera_info_probe(topic+'/color/camera_info');observed[prefix+'_intrinsics']=info
            checks[prefix+'_intrinsics']=helpers.camera_info_valid(info,width=640,height=480,frame_id=frame,expected_intrinsics=gate['camera']['intrinsics'] if prefix=='wrist' else None)
            tf=run(['ros2','run','tf2_ros','tf2_echo','base_link',frame],timeout=8)
            observed[prefix+'_tf']=tf;checks[prefix+'_tf']='Translation:' in tf['stdout'] and 'Rotation:' in tf['stdout']
        labels=helpers.image_probe('/wrist_camera/evaluation_labels/labels_map');observed['labels']=labels
        checks['labels']=helpers.image_matches(labels,width=640,height=480,encodings={'rgb8'},frame_id='camera_color_optical_frame')
        joint,ok,comparison=parsers.joint_state_probe(gate['camera']['view_joint_pose']);observed['joints']={'probe':joint,'comparison':comparison};checks['fixed_pose']=ok
        reset=helpers.reset_probe(scenes);observed['reset']=reset;checks['reset']=reset['success']
        post=helpers.image_probe('/wrist_camera/depth/image_raw');observed['post_reset_depth']=post
        checks['post_reset_geometry_sensor']=helpers.image_matches(post,width=640,height=480,encodings={'32FC1'},frame_id='camera_color_optical_frame')
    else:checks['live_sensors_pose_tf_reset']=False
    checks['test_robot_sealed']=all(gate['sealed'][s] is True for s in ('gazebo_test_iid','gazebo_test_ood','robot'))
    write_new(path,{'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,'observations':observed,
                   'implementation_sha256':sha(IMPLEMENTATION),'static_sha256':sha(static_path),'capture_authorized':False})
    print(json.dumps({'status':read(path)['status'],'checks':checks},indent=2))
    if not all(checks.values()):terminal_block('LIVE_PREFLIGHT',[path],pid);raise SystemExit(2)


def authorize(pid):
    implementation=validate_implementation();output=ROOT/f'results/spatial_vlm_refspatial_v1/{pid}'
    paths=[RESULT/'PREFLIGHT_STATIC.json',output/'PREFLIGHT_LIVE.json']
    if pid==design.CALIBRATION:paths.append(ROOT/f'results/spatial_vlm_refspatial_v1/{design.PILOT}/GEOMETRY_QC.json')
    if any(read(p)['status']!='PASS' for p in paths):raise RuntimeError('predecessor PASS required')
    if (output/'capture_attempt_01').exists():raise RuntimeError('capture already exists')
    sources=dict(implementation['source_artifact_sha256']);sources.update({str(p.relative_to(ROOT)):sha(p) for p in [IMPLEMENTATION,*paths]})
    write_new(ROOT/f'protocol/{pid}_capture_compatibility_lock.json',{'status':'LOCKED_BEFORE_CAPTURE_AND_INFERENCE','protocol_id':pid,
        'workspace_root':str(ROOT),'authorization_status':'CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS',
        'parent_contract_lock_sha256':sha(design.LOCK),'implementation_lock_sha256':sha(IMPLEMENTATION),
        'source_artifact_sha256':sources,'expected_parent_families':32 if pid==design.PILOT else 128,
        'model_inventory_sha256':'NO_MODEL_INFERENCE_DURING_CAPTURE','capture_attempts_authorized':1,'test_iid_ood_access':False,'robot_access':False})


def verify_authorization(pid):
    validate_implementation();lock=read(ROOT/f'protocol/{pid}_capture_compatibility_lock.json')
    if lock['protocol_id']!=pid or lock['authorization_status']!='CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS':raise RuntimeError('capture unauthorized')
    for p,d in lock['source_artifact_sha256'].items():
        if sha(ROOT/p)!=d:raise RuntimeError('authorization hash drift: '+p)


def capture(pid):
    verify_authorization(pid);output=ROOT/f'results/spatial_vlm_refspatial_v1/{pid}/capture_attempt_01'
    if output.exists():raise FileExistsError('single capture attempt already exists')
    command=['/usr/bin/python3',str(ROOT/'ur3/ur3_perception/scripts/roborefer_pilot_capture.py'),'--ros-args','-p','use_sim_time:=true']
    for key,value in [('scene_config_file',design.paths(pid)[0]),('annotation_file',design.paths(pid)[1]),('gate_config_file',design.paths(pid)[2]),
                      ('pretrial_lock_file',ROOT/f'protocol/{pid}_capture_compatibility_lock.json'),('output_root',output),('settle_sec','1.5'),('sync_slop_sec','0.02'),('capture_timeout_sec','45.0')]:
        command+=['-p',f'{key}:={value}']
    subprocess.run(command,cwd=ROOT,env=environment.ros_environment(),check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','validate','preflight-static','preflight-live','authorize','capture']);parser.add_argument('--split',choices=['pilot','calibration'],default='pilot');args=parser.parse_args()
    pid=design.PILOT if args.split=='pilot' else design.CALIBRATION
    if args.command=='freeze':freeze()
    elif args.command=='validate':validate_implementation();print('PASS')
    elif args.command=='preflight-static':static()
    elif args.command=='preflight-live':live(pid)
    elif args.command=='authorize':authorize(pid)
    else:capture(pid)
