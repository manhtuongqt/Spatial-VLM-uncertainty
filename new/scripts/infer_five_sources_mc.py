#!/usr/bin/env python3
"""One feature/prompt observation: original decision plus five source MC estimates."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from pcrau.source_mc import predict_with_source_mc
from pcrau.unified_inference import UnifiedInference,load_features
from pcrau.utils import atomic_json,seed_everything,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json'))
    p.add_argument('--features',type=Path,required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=24082026)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4);seed_everything(a.seed)
    pipeline=UnifiedInference.from_bundle(a.bundle);features=load_features(a.features)
    result=predict_with_source_mc(pipeline,{k:v[None] for k,v in features.items()},[a.prompt],passes=20,seed=a.seed)
    atomic_json(a.output,{'original_runtime':result['baseline_predictions'][0],
        'source_MC':result['source_mc_rows'][0],
        'MC_statistics_fed_into_original_risk':False,'feature_sha256':sha256_file(a.features),
        'original_bundle_sha256':sha256_file(a.bundle),'deterministic_head_copy_exact':result['deterministic_copy_exact']})
    print(a.output)


if __name__=='__main__':main()
