#!/usr/bin/env python3
"""Isolated fresh IID protocol; fixed verified sensor, no robot controller.

Reuses qualified generation/QC/materialization functions with explicit isolated
paths. Historical data is only a denylist; observations are captured afresh.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'new/outputs/pcrau_support_split_dev_20261003'
HERE=RUN/'iid_confirmation'
PROTOCOL='roborefer_dataset_v2_1_iid_confirm_200_20261003'
PREFIX='v211confirm07_family_'
DECISION='GO_FRESH_IID_FIXED_SENSOR_CAPTURE'
sys.path.insert(0,str(ROOT/'old/protocol'))
sys.path.insert(0,str(ROOT/'new/test_iid/scripts'))
from wp2_common import sha256_file, read_json, write_json


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj


def verify_model_lock():
    lock=read_json(RUN/'freeze_lock.json');threshold=read_json(RUN/'split_calibration/threshold_lock.json')
    for key,hkey in [('source_checkpoint','source_checkpoint_sha256'),('config','config_sha256'),('ranker_checkpoint','ranker_sha256')]:
        assert sha256_file(Path(lock[key]))==lock[hkey],key
    assert sha256_file(RUN/'split_calibration/selected_calibrator.json')==threshold['calibrator_sha256']
    return lock,threshold


def prepare():
    lock,threshold=verify_model_lock()
    if (HERE/'protocol/test_iid_manifest.json').exists():raise FileExistsError('Fresh protocol already sealed')
    HERE.mkdir(exist_ok=True)
    (HERE/'protocol').mkdir(exist_ok=True);(HERE/'contracts').mkdir(exist_ok=True)
    # Generator compatibility gate. The actual experimental freeze is recorded
    # independently below, and no calibration is performed on this new set.
    historical=read_json(ROOT/'new/test_iid/contracts/best_v2_freeze_lock.json')
    write_json(HERE/'contracts/best_v2_freeze_lock.json',historical)
    generation_gate={**historical,'architecture_frozen':True,'checkpoint_frozen':True,
                     'allowed_next_action':'CAPTURE_CALIBRATION','test_opened':False,
                     'compatibility_adapter_for_fresh_iid_generation':True,'experimental_model_lock':lock}
    write_json(HERE/'contracts/generation_compatibility_gate.json',generation_gate)
    g=module('fresh_generation','new/test_iid/scripts/generate_test_iid.py')
    g.PROTOCOL_ID=PROTOCOL;g.MASTER_SEED=202610031707;g.FAMILY_PREFIX=PREFIX
    g.OUTPUT_ROOT=str((HERE/'dataset').relative_to(ROOT))
    g.FREEZE_LOCK=str((HERE/'contracts/generation_compatibility_gate.json').relative_to(ROOT))
    g.configure_base()
    prior=('new/test_iid/protocol/test_iid_manifest.json','new/test_iid/protocol/test_iid_capture_plan.json')
    g.base.DENYLIST_JSON=tuple(dict.fromkeys((*g.base.DENYLIST_JSON,*prior)))
    manifest,plan=g.base.build_artifacts(ROOT);g.rewrite_for_test(manifest,plan)
    manifest['confirmation_for']='frozen support scorer at selected 7.5% budget'
    manifest['prior_test_informed_research_question']=True
    assert len(manifest['families'])==200 and len(plan['captures'])==400
    current={'manifest':manifest,'plan':plan};previous=[read_json(ROOT/f) for f in prior]
    from dataset_v2_1_development_generator import collect_values,collect_seed_values
    overlap={}
    for name,keys in [('families',{'family_id'}),('captures',{'capture_id'}),('instructions',{'instruction'}),
                      ('layouts',{'layout_instance_fingerprint_sha256','layout_fingerprint_sha256'})]:
        overlap[name]=len(collect_values(current,keys)&set().union(*(collect_values(x,keys) for x in previous)))
    overlap['seeds']=len(collect_seed_values(current)&set().union(*(collect_seed_values(x) for x in previous)))
    assert not any(overlap.values()),overlap
    write_json(HERE/'protocol/test_iid_manifest.json',manifest)
    plan['manifest_sha256']=sha256_file(HERE/'protocol/test_iid_manifest.json')
    write_json(HERE/'protocol/test_iid_capture_plan.json',plan)
    build=module('fresh_static_world','new/demo_gazebo/build_static_robot_world.py')
    build.SOURCE_WORLD=ROOT/plan['world_file'];build.TEST_IID_PLAN=HERE/'protocol/test_iid_capture_plan.json'
    world=HERE/'protocol/world_fixed_sensor.sdf';build.build(world)
    optical=build.verify_test_iid_optical_pose(world)
    frozen=[Path(lock['source_checkpoint']),Path(lock['config']),Path(lock['ranker_checkpoint']),
            RUN/'split_calibration/selected_calibrator.json',RUN/'split_calibration/threshold_lock.json',
            HERE/'protocol/test_iid_manifest.json',HERE/'protocol/test_iid_capture_plan.json',world]
    write_json(HERE/'protocol/execution_lock.json',{'protocol_id':PROTOCOL,'decision':DECISION,
        'capture_authorized':True,'training_authorized':False,'test_inference_authorized':False,
        'capture_output_root':str((HERE/'dataset').relative_to(ROOT)),
        'capture_plan_sha256':sha256_file(HERE/'protocol/test_iid_capture_plan.json'),
        'manifest_sha256':sha256_file(HERE/'protocol/test_iid_manifest.json'),
        'locked_artifact_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in frozen},
        'historical_disjointness':overlap,'fixed_sensor_optical_pose_verified':optical,
        'robot_motion_commanded':False,'robot_controller_started':False,'threshold_lock':threshold})
    print(json.dumps({'fresh_families':200,'planned_samples':1000,'captures':400,'overlap':overlap,'world':str(world)},indent=2))


def stop(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)


def capture(batch_id):
    verify_model_lock();plan=read_json(HERE/'protocol/test_iid_capture_plan.json');lock=read_json(HERE/'protocol/execution_lock.json')
    for f,h in lock['locked_artifact_sha256'].items():assert sha256_file(ROOT/f)==h,f
    batches={b['batch_id']:b for b in plan['batches']};batch=batches[batch_id]
    if batch['prerequisite_batch']:
        assert read_json(HERE/f"dataset/report_assets/checkpoints/batch_qc/{batch['prerequisite_batch']}.json")['passed']
    import rclpy
    import dataset_v2_pilot_capture as shared
    shared.PROTOCOL_ID=PROTOCOL;shared.CAPTURE_ID=re.compile(r'^v211confirm07_family_[0-9]{6}__(?:clean|occlusion)_capture$')
    from sensor_msgs.msg import JointState
    class FixedCapture(shared.PilotCapture):
        def move_camera(self,requested):
            assert requested==plan['camera_poses']['camera_v2_1_relation'],'Requested camera differs from baked sensor'
        def tf_snapshot(self):
            t=plan['camera_predictor_transform']
            return {'camera_color_optical_frame':{'parent_frame':'base_link','child_frame':'camera_color_optical_frame',**t},
                    'source':'verified generated fixed sensor; not a measured moving-robot TF'}
        def save_capture(self,capture,sensor_tuple):
            meta=super().save_capture(capture,sensor_tuple)
            path=self.capture_root/capture['capture_id']/'capture_meta.json'
            meta['capture_backend']='Gazebo Fortress fixed sensor, baked locked wrist pose, RGB-D + semantic segmentation'
            meta['joint_pose_source']='verified static URDF bake, no joint controller or motion'
            meta['robot_motion_commanded']=False;write_json(path,meta);return meta
    env=os.environ.copy();env['IGN_PARTITION']='pcrau_fresh_iid_confirmation';env['ROS_LOCALHOST_ONLY']='1'
    os.environ.update({k:env[k] for k in ['IGN_PARTITION','ROS_LOCALHOST_ONLY']})
    logs=HERE/'logs';logs.mkdir(exist_ok=True)
    bridge_args=['/wrist_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
        '/wrist_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
        '/wrist_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        '/wrist_camera/evaluation_labels/labels_map@sensor_msgs/msg/Image[ignition.msgs.Image',
        '/world/ur3_pick_place/set_pose@ros_gz_interfaces/srv/SetEntityPose', '--ros-args',
        '-r','/wrist_camera/image:=/wrist_camera/color/image_raw',
        '-r','/wrist_camera/depth_image:=/wrist_camera/depth/image_raw',
        '-r','/wrist_camera/camera_info:=/wrist_camera/color/camera_info']
    processes=[]
    with (logs/f'{batch_id}_gazebo.log').open('a') as gl,(logs/f'{batch_id}_bridge.log').open('a') as bl:
        try:
            processes.append(subprocess.Popen(['ign','gazebo','-s','-r','--headless-rendering','-v','2',str(HERE/'protocol/world_fixed_sensor.sdf')],cwd=ROOT,env=env,stdout=gl,stderr=subprocess.STDOUT,start_new_session=True))
            processes.append(subprocess.Popen(['ros2','run','ros_gz_bridge','parameter_bridge',*bridge_args],cwd=ROOT,env=env,stdout=bl,stderr=subprocess.STDOUT,start_new_session=True))
            rclpy.init();node=FixedCapture(plan,HERE/'dataset',.8)
            node.joint_state=JointState(name=shared.JOINT_NAMES,position=plan['camera_poses']['camera_v2_1_relation'])
            try:node.run([c for c in plan['captures'] if c['batch_id']==batch_id])
            finally:node.destroy_node();rclpy.shutdown()
        finally:
            for process in reversed(processes):stop(process)
            write_json(HERE/'dataset/report_assets/checkpoints/shutdown_sequence.json',{
                'ordered_shutdown_complete':all(p.poll() is not None for p in processes),
                'owned_processes_only':True,'robot_controller_started':False,'robot_motion_commanded':False})
    import dataset_v2_development_capture as inventory
    inventory.PROTOCOL_ID=PROTOCOL;inventory.EXPECTED_CAPTURES=400;inventory.CAPTURE_PATTERN=shared.CAPTURE_ID;inventory.shared.PROTOCOL_ID=PROTOCOL
    raw=inventory.inventory_raw(HERE/'dataset',plan);write_json(HERE/'dataset/raw/raw_capture_manifest.json',raw)
    qc=module('fresh_qc','old/protocol/dataset_v2_1_development_batch_qc.py');qc.PROTOCOL_ID=PROTOCOL;qc.DECISION=DECISION
    report=qc.run_qc(ROOT,HERE/'dataset',HERE/'protocol/test_iid_capture_plan.json',HERE/'protocol/execution_lock.json',HERE/'protocol/test_iid_manifest.json',batch_id)
    write_json(HERE/f'dataset/report_assets/checkpoints/batch_qc/{batch_id}.json',report)
    print(json.dumps({'batch':batch_id,'passed':report['passed'],'errors':report['errors'],'valid_captures':report['valid_capture_count']},indent=2))
    if not report['passed']:raise RuntimeError('Fresh IID capture QC failed; do not infer or retune the model')


def materialize():
    verify_model_lock()
    m=module('fresh_materialize','new/test_iid/scripts/materialize.py')
    m.TEST_ROOT=HERE;m.DATASET_ROOT=HERE/'dataset';m.PROTOCOL_ROOT=HERE/'protocol';m.PROTOCOL_ID=PROTOCOL
    m.materialize()
    p=module('fresh_prepare_inputs','new/test_iid/scripts/prepare_inference.py')
    p.TEST_ROOT=HERE;p.DATASET_ROOT=HERE/'dataset';p.PROTOCOL_ROOT=HERE/'protocol';p.PROTOCOL_ID=PROTOCOL
    print(json.dumps(p.prepare(),indent=2))


def features():
    verify_model_lock()
    b=module('fresh_features','new/test_iid/scripts/build_feature_cache.py')
    b.TEST_ROOT=HERE;b.MANIFEST_PATH=HERE/'protocol/test_iid_feature_manifest.json';b.GATE_PATH=HERE/'protocol/frozen_inference_gate.json'
    b.CACHE_ROOT=HERE/'feature_cache';b.PROTOCOL_ID=PROTOCOL
    b.EXPECTED_UNIQUE_PAIRS=read_json(b.MANIFEST_PATH)['unique_feature_pair_count']
    b.build()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','capture','materialize','features']);p.add_argument('--batch',default='canary_000');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='capture':capture(a.batch)
    elif a.stage=='materialize':materialize()
    else:features()
