#!/usr/bin/env python3
"""Append the hashed Day-8 timeout wrapper to already sealed failed-attempt locks."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WRAPPER=ROOT/'workspace/mh_pcrau_v3/day8_top_camera_capture.launch.py'
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
for lock_path in sorted((ROOT/'ketquangay/ngay_08/du_lieu/canary_v5').glob('camera_*/CAPTURE_SOURCE_LOCK.json')):
 lock=json.loads(lock_path.read_text());lock['source_artifact_sha256'][str(WRAPPER.relative_to(ROOT))]=sha(WRAPPER);lock['source_artifact_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__));lock['retry_policy']='Previous top-camera attempts produced zero accepted scenes; retry uses identical config with capture timeout 150 s.';lock_path.write_text(json.dumps(lock,indent=2)+'\n')
print(json.dumps({'status':'LOCKS_EXTENDED','count':4,'wrapper_sha256':sha(WRAPPER)},indent=2))
