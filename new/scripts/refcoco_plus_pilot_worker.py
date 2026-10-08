#!/usr/bin/env python3
"""Oracle-free RGB/depth/features/RoboRefer worker for the RefCOCO+ pilot."""
import os,sys,json,gc,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'new/outputs/refcoco_plus_val_pilot300_20261007'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(ROOT/'RoboRefer'),str(ROOT/'RoboRefer/API/Depth_Anything_V2'),str(ROOT/'old/protocol'),str(ROOT/'new/scripts')]
import torch,cv2,numpy as np
from safetensors.torch import save_file
from wp3_feature_hook_smoke import load_model,extract_features,generate_answer,model_inventory_sha256
from pcra_u_development_common import pool_raw_feature,seed_runtime
from evaluate_refspatial_location import sha,write

def rgb_answer(model,rgb,prompt):
    from llava.media import Image
    cfg=copy.deepcopy(model.default_generation_config);cfg.do_sample=False;cfg.temperature=None;cfg.top_p=None;cfg.top_k=None;cfg.max_new_tokens=128
    return str(model.generate_content([Image(str(rgb)),prompt],generation_config=cfg))

def main():
    torch.set_num_threads(4);seed_runtime(24082026,strict=False)
    p=json.loads((OUT/'protocol.json').read_text());assert sha(__file__)==p['worker_sha256']
    assert sha(OUT/'runtime_manifest.json')==p['runtime_manifest_sha256']
    entries=json.loads((OUT/'runtime_manifest.json').read_text());images={e['image_id']:e for e in entries}
    for folder in ('inputs','features'):(OUT/folder).mkdir(exist_ok=True)
    from depth_anything_v2.dpt import DepthAnythingV2
    weight=ROOT/'RoboRefer/models/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth'
    dm=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024]);dm.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));dm=dm.cuda().eval()
    provenance=[]
    with torch.inference_mode():
        for n,e in enumerate(images.values(),1):
            assert sha(e['original_rgb'])==e['rgb_sha256'];raw=cv2.imread(e['original_rgb']);assert raw is not None
            z=dm.infer_image(raw,input_size=518,device='cuda');z=((z-z.min())/max(float(z.max()-z.min()),1e-6)*255).astype(np.uint8);z=np.repeat(z[:,:,None],3,axis=2)
            rgb=cv2.resize(raw,(640,480),interpolation=cv2.INTER_LINEAR);z=cv2.resize(z,(640,480),interpolation=cv2.INTER_LINEAR)
            assert cv2.imwrite(e['model_rgb'],rgb) and cv2.imwrite(e['model_depth'],z)
            provenance.append({'image_id':e['image_id'],'rgb_sha256':sha(e['model_rgb']),'depth_sha256':sha(e['model_depth'])})
            if n%10==0:print(f'DEPTH {n}/100',flush=True)
    write(OUT/'depth_complete.json',{'weight_sha256':sha(weight),'images':provenance})
    del dm;gc.collect();torch.cuda.empty_cache()
    modelroot=ROOT/'RoboRefer/models/RoboRefer-2B-SFT';inv=model_inventory_sha256(modelroot);assert inv=='5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa'
    model=load_model(modelroot);target=OUT/'backbone_predictions.jsonl';assert not target.exists()
    done=set()
    with target.open('w') as f:
        for n,e in enumerate(entries,1):
            if e['image_id'] not in done:
                raw=extract_features(model,Path(e['model_rgb']),Path(e['model_depth']));r,rt=pool_raw_feature(raw['r0']);d,dt=pool_raw_feature(raw['d0'])
                save_file({k:v.detach().cpu().half().contiguous() for k,v in zip(('r0','d0','r_thumb','d_thumb'),(r,d,rt,dt))},e['feature_path'])
                del raw,r,rt,d,dt;done.add(e['image_id'])
            seed_runtime(24082026,strict=False)
            answer,ms=generate_answer(model,Path(e['model_rgb']),Path(e['model_depth']),e['backbone_prompt'],{'max_new_tokens':128})
            seed_runtime(24082026,strict=False);native=rgb_answer(model,e['original_rgb'],e['backbone_prompt'])
            f.write(json.dumps({'id':e['id'],'image_id':e['image_id'],'answer':answer,'native_rgb_answer':native,'matched_generation_ms':ms,'feature_sha256':sha(e['feature_path'])})+'\n');f.flush()
            if n%10==0:print(f'ROBOREFER {n}/300: RGBD={answer}; RGB={native}',flush=True)
    write(OUT/'backbone_complete.json',{'samples':300,'images':len(done),'backbone_inventory_sha256':inv,'torch':torch.__version__})

if __name__=='__main__':main()
