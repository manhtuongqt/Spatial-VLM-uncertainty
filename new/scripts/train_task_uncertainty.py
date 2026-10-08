#!/usr/bin/env python3
"""Train only task likelihood heads on existing family-locked train/dev."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,time,runpy
from pathlib import Path
from functools import lru_cache
import cv2,numpy as np,torch
from safetensors.torch import save_file,load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcrau.task_uncertainty import TaskHeads,head_losses,capture_context
from pcrau.unified_inference import UnifiedInference,load_features
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.anchor_shadow_experiment import state_digest
from pcrau.utils import atomic_json,sha256_file,seed_everything

OUT=Path('new/outputs/pcrau_task_uncertainty_20261007')

def main():
    assert not (OUT/'training_lock.json').exists()
    torch.set_num_threads(4);seed_everything(24082026)
    tests=[]
    for name,fn in runpy.run_path('new/tests/test_task_uncertainty.py').items():
        if name.startswith('test_'):fn();tests.append(name)
    supplement=json.loads((OUT/'data/supplement.json').read_text())
    protected={}
    for base in ('new/src/pcrau','new/scripts','new/tests','new/configs','new/outputs'):
        for p in Path(base).rglob('*'):
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts:protected[str(p.resolve())]=sha256_file(p)
    pipeline=UnifiedInference.from_bundle('new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json')
    initial=state_digest(pipeline.shadow.state_dict())
    lock={'protocol_sha256':sha256_file(Path('plan/S6_TASK_UNCERTAINTY_PROTOCOL_20261007.md')),
        'baseline_state':initial,'supplement_sha256':sha256_file(OUT/'data/supplement.json'),
        'train_samples':1600,'dev_samples':400,'seed':24082026,'epochs_max':15,'batch_size':20,
        'early_stop_patience':3,'learning_rate':.0003,'weight_decay':.001,'classes':supplement['classes'],
        'optimizer_scope':'three_task_heads_only','dropout':.1,'tests':tests,
        'source_hashes':{p:sha256_file(Path(p)) for p in ['new/src/pcrau/task_uncertainty.py',
            'new/scripts/train_task_uncertainty.py','new/tests/test_task_uncertainty.py']},
        'protected_files':protected}
    atomic_json(OUT/'training_lock.json',lock)
    obs={r['sample_id']:r for r in supplement['observations']}
    pairs={r['sample_id']:r for r in supplement['completion_pairs']}
    label_class={int(k):v for k,v in supplement['label_class'].items()}
    xs=np.arange(32)*20+10;ys=np.arange(24)*20+10
    datasets={}
    for split in ('train','dev'):
        ds=ArchivedPCRAUDataset(pipeline.config,split)
        contexts=[];labels=[];metadata=[]
        @lru_cache(maxsize=32)
        def feature(key):
            d=ds.features[key];p=ds.layout.feature_root/d['path'];assert sha256_file(p)==d['sha256'];return load_features(p)
        for start in range(0,len(ds),20):
            entries=ds.entries[start:start+20]
            fs=[feature(ds.sample_to_feature[e['sample_id']]) for e in entries]
            features={k:torch.stack([f[k] for f in fs]) for k in fs[0]}
            prompts=[e['feature_input']['prompt'] for e in entries]
            context,cap=capture_context(pipeline,features,prompts)
            contexts.append({k:v.detach().half().cpu() for k,v in context.items()})
            batchlabels=[]
            for e in entries:
                o=obs[e['sample_id']]
                instances=cv2.imread(o['instance_labels_supervision_only']['path'],cv2.IMREAD_UNCHANGED)
                centers=instances[np.ix_(ys,xs)]
                semantic=np.zeros((24,32),dtype=np.int64)
                for key,c in label_class.items():semantic[centers==key]=c
                dep=np.load(o['reference_metric_depth_supervision_only']['path'])[np.ix_(ys,xs)].astype('float32')
                valid=np.isfinite(dep)&(dep>.1)&(dep<5)
                dep=np.where(valid,dep,0)
                p=ds.layout.dataset_path(e['supervision']['target_mask_path'])
                mask=cv2.imread(str(p),0)>0
                if e['sample_id'] in pairs:
                    pair=pairs[e['sample_id']]
                    mask=cv2.imread(pair['reference_semantic_labels_supervision_only']['path'],cv2.IMREAD_UNCHANGED)==pair['reference_label_id_supervision_only']
                density=cv2.resize(mask.astype('float32'),(32,24),interpolation=cv2.INTER_AREA)
                batchlabels.append({'semantic':torch.from_numpy(semantic),'depth':torch.from_numpy(dep),
                    'depth_valid':torch.from_numpy(valid),'completion':torch.from_numpy(density)})
                metadata.append({'sample_id':e['sample_id'],'family_id':e['family_id'],'prompt':e['feature_input']['prompt'],
                    'completion_is_reference':e['sample_id'] in pairs})
            labels.append({k:torch.stack([r[k] for r in batchlabels]) for k in batchlabels[0]})
            if start%400==0:print(f'cache {split} {start+len(entries)}/{len(ds)}',flush=True)
        c={k:torch.cat([b[k] for b in contexts]) for k in contexts[0]}
        l={k:torch.cat([b[k] for b in labels]) for k in labels[0]}
        save_file({**{'input_'+k:v for k,v in c.items()},**{'label_'+k:v for k,v in l.items()}},str(OUT/f'{split}_task_cache.safetensors'))
        atomic_json(OUT/f'{split}_cache_metadata.json',metadata)
        datasets[split]=({k:v.to('cuda') for k,v in c.items()},{k:v.to('cuda') for k,v in l.items()},metadata)
        del contexts,labels
    seed_everything(24082026);heads=TaskHeads(len(supplement['classes'])).cuda()
    assert all(not p.requires_grad for p in pipeline.shadow.parameters())
    optimizer=torch.optim.AdamW(heads.parameters(),lr=.0003,weight_decay=.001)
    def evaluate(split):
        c,l,meta=datasets[split];heads.eval();sums={'semantic':0.,'depth':0.,'completion':0.};counts=dict.fromkeys(sums,0)
        with torch.no_grad():
            for start in range(0,len(meta),20):
                cc={k:v[start:start+20] for k,v in c.items()};ll={k:v[start:start+20] for k,v in l.items()}
                loss=head_losses(heads(cc),ll)
                ns={'semantic':ll['semantic'].numel(),'depth':int(ll['depth_valid'].sum()),
                    'completion':int((ll['completion'].flatten(1).sum(-1)>0).sum())}
                for k in sums:sums[k]+=float(loss[k])*ns[k];counts[k]+=ns[k]
        result={k:sums[k]/max(1,counts[k]) for k in sums};result['total']=sum(result.values());return result
    history=[];best=float('inf');wait=0
    c,l,meta=datasets['train'];families=sorted({r['family_id'] for r in meta});family_rows={f:[i for i,r in enumerate(meta) if r['family_id']==f] for f in families}
    for epoch in range(15):
        heads.train();order=torch.randperm(len(families)).tolist();started=time.perf_counter();losses=[]
        for start in range(0,len(families),4):
            idx=[i for j in order[start:start+4] for i in family_rows[families[j]]]
            cc={k:v[idx] for k,v in c.items()};ll={k:v[idx] for k,v in l.items()}
            optimizer.zero_grad(set_to_none=True);loss=head_losses(heads(cc),ll)
            assert torch.isfinite(loss['total']);loss['total'].backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),5)
            optimizer.step();losses.append(float(loss['total']))
        dev=evaluate('dev');row={'epoch':epoch+1,'train_batch_joint_loss':float(np.mean(losses)),'dev':dev,'seconds':time.perf_counter()-started}
        history.append(row);atomic_json(OUT/'history.json',history)
        if dev['total']<best:
            best=dev['total'];wait=0;save_file({k:v.detach().cpu().contiguous() for k,v in heads.state_dict().items()},str(OUT/'best_task_heads.safetensors'))
            atomic_json(OUT/'best_metadata.json',row)
        else:wait+=1
        print(json.dumps(row),flush=True)
        if wait>=3:break
    heads.load_state_dict(load_file(str(OUT/'best_task_heads.safetensors')));heads.eval().requires_grad_(False)
    assert state_digest(pipeline.shadow.state_dict())==initial;pipeline.shadow._check_frozen()
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    atomic_json(OUT/'training_result.json',{'status':'COMPLETE','best':json.loads((OUT/'best_metadata.json').read_text()),
        'best_reload_dev':evaluate('dev'),'neural_baseline_unchanged':True,'original_hashes_unchanged':len(protected),
        'task_checkpoint_sha256':sha256_file(OUT/'best_task_heads.safetensors'),'epochs_run':len(history)})

if __name__=='__main__':main()
