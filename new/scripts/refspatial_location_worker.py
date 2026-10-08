#!/usr/bin/env python3
"""Isolated legacy worker: relative depth, four feature tensors, RoboRefer output.

Only runtime_manifest (RGB paths, prompts) is read; no masks or evaluator labels.
"""
import os,sys,json,time,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'new/outputs/refspatial_location_frozen_20261007'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(ROOT/'RoboRefer'))
sys.path.insert(0,str(ROOT/'RoboRefer/API/Depth_Anything_V2'))
sys.path.insert(0,str(ROOT/'old/protocol'))
import torch,cv2,numpy as np
from safetensors.torch import save_file
from wp3_feature_hook_smoke import load_model,extract_features,generate_answer,model_inventory_sha256
from pcra_u_development_common import pool_raw_feature,seed_runtime

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(4);seed_runtime(24082026,strict=False)
    p=json.loads((OUT/'protocol.json').read_text());assert digest(__file__)==p['worker_sha256']
    entries=json.loads((OUT/'runtime_manifest.json').read_text())
    for d in ('inputs','features'): (OUT/d).mkdir(exist_ok=True)
    if not (OUT/'depth_complete.json').exists():
        from depth_anything_v2.dpt import DepthAnythingV2
        weight=ROOT/'RoboRefer/models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth'
        model=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024])
        model.load_state_dict(torch.load(weight,map_location='cpu'));model=model.cuda().eval()
        meta=[]
        with torch.inference_mode():
            for n,e in enumerate(entries,1):
                raw=cv2.imread(e['original_rgb']);assert raw is not None
                depth=model.infer_image(raw,input_size=518,device='cuda')
                z=((depth-depth.min())/max(float(depth.max()-depth.min()),1e-6)*255).astype(np.uint8)
                z=np.repeat(z[:,:,None],3,axis=2)
                assert cv2.imwrite(e['native_depth'],z)
                rgb=cv2.resize(raw,(640,480),interpolation=cv2.INTER_LINEAR);z=cv2.resize(z,(640,480),interpolation=cv2.INTER_LINEAR)
                assert cv2.imwrite(e['model_rgb'],rgb) and cv2.imwrite(e['model_depth'],z)
                meta.append({'id':e['id'],'rgb_sha256':digest(e['model_rgb']),'depth_sha256':digest(e['model_depth'])})
                if n%10==0:print(f'DEPTH {n}/100',flush=True)
        (OUT/'depth_complete.json').write_text(json.dumps({'weight_sha256':digest(weight),'input_size':518,'inputs':meta},indent=2))
        del model;gc.collect();torch.cuda.empty_cache()
    expected='5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa'
    model_root=ROOT/'RoboRefer/models/RoboRefer-2B-SFT'
    assert model_inventory_sha256(model_root)==expected
    model=load_model(model_root)
    target=OUT/'backbone_predictions.jsonl'
    assert not target.exists(),'Refuse to overwrite backbone predictions.'
    with target.open('w') as f:
        for n,e in enumerate(entries,1):
            extracted=extract_features(model,Path(e['model_rgb']),Path(e['model_depth']))
            r,rt=pool_raw_feature(extracted['r0']);d,dt=pool_raw_feature(extracted['d0'])
            save_file({k:v.detach().cpu().half().contiguous() for k,v in zip(('r0','d0','r_thumb','d_thumb'),(r,d,rt,dt))},e['feature_path'])
            seed_runtime(24082026,strict=False)
            answer,ms=generate_answer(model,Path(e['model_rgb']),Path(e['model_depth']),e['backbone_prompt'],{'max_new_tokens':128})
            if e['original_size']==[640,480]:native,native_ms=answer,ms
            else:
                seed_runtime(24082026,strict=False)
                native,native_ms=generate_answer(model,Path(e['original_rgb']),Path(e['native_depth']),e['backbone_prompt'],{'max_new_tokens':128})
            f.write(json.dumps({'id':e['id'],'answer':answer,'generation_ms':ms,'native_answer':native,'native_generation_ms':native_ms,'feature_sha256':digest(e['feature_path'])})+'\n');f.flush()
            print(f'ROBOREFER {n}/100: {answer}',flush=True)
            del extracted,r,d,rt,dt
    (OUT/'backbone_complete.json').write_text(json.dumps({'samples':100,'inventory_sha256':expected,'torch':torch.__version__}))

if __name__=='__main__':main()
