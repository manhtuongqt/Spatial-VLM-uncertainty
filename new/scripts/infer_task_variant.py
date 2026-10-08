#!/usr/bin/env python3
"""One RGB-D observation -> MC task distributions -> calibrated perception action."""
import argparse,os,sys,time,json
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from pcrau.task_inference import TaskInference
from pcrau.unified_inference import load_features
from pcrau.frozen_rgbd import extract_rgbd
from pcrau.utils import atomic_json,sha256_file,seed_everything

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/bundle_release.json'))
    inp=p.add_mutually_exclusive_group(required=True);inp.add_argument('--features',type=Path);inp.add_argument('--rgb',type=Path)
    p.add_argument('--depth',type=Path);p.add_argument('--prompt',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=24082026);a=p.parse_args()
    if (a.rgb is None)!=(a.depth is None):p.error('--rgb and relative --depth required together')
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4);seed_everything(a.seed)
    if a.features:
        features=load_features(a.features);provenance={'features':str(a.features.resolve()),'sha256':sha256_file(a.features)}
    else:
        manifest=json.loads(a.bundle.read_text());bpath=Path(manifest['baseline_bundle']['path']);b=json.loads(bpath.read_text())
        desc=b['freeze_lock'];lockpath=Path(desc['path']);lockpath=lockpath if lockpath.is_absolute() else bpath.parent/lockpath
        assert sha256_file(lockpath)==desc['sha256']
        lock=json.loads(lockpath.read_text());features,provenance=extract_rgbd(a.rgb,a.depth,lock['backbone_inventory_sha256'])
    model=TaskInference.from_bundle(a.bundle);tick=time.perf_counter()
    rows,_=model.predict({k:v[None] for k,v in features.items()},[a.prompt],a.seed);torch.cuda.synchronize()
    r=rows[0];r['runtime']={'bundle_sha256':sha256_file(a.bundle),'visual_input':provenance,
        'feature_path_MC_ms':(time.perf_counter()-tick)*1000,'T':20,'seed':a.seed,'robot_motion_commanded':False}
    atomic_json(a.output,r);print(f"{r['decision']['action']} risk={r['decision']['risk']:.6f} -> {a.output}")

if __name__=='__main__':main()
