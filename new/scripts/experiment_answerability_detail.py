#!/usr/bin/env python3
"""Train-only query detail residual; strict dev answerability selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcrau.answerability_evidence import AnswerabilityEvidenceAdapter
from pcrau.checkpoint import load_model_checkpoint, save_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset, collate_samples
from pcrau.language_augmentation import TrainLanguageAugmentation, PREFIXES, augmented_id
from pcrau.engine import autocast_context, evaluate
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import weighted_answer_loss
from pcrau.utils import atomic_json, load_config, seed_everything, sha256_file, workspace_path
from experiment_answerability_head import answer_metrics, make_loader
from experiment_answerability_evidence import evidence_eligible, predict
from experiment_hard_cases_selective import capture, write_rows


def detail_capture(base, loader, data, device, config, weights, source_weights, output):
    detail=[]
    handle=base.register_forward_hook(lambda module,args,result:detail.append(result['observable_answerability_detail'].detach().cpu()))
    try:
        metrics,rows,cache=capture(base,loader,data,device,config,weights,source_weights,output)
    finally:
        handle.remove()
    cache['evidence']=torch.cat([cache['evidence'][:,:26],torch.cat(detail).float()],dim=-1)
    assert cache['evidence'].shape==(len(data),2222)
    return metrics,rows,cache


def update(adapter,optimizer,indices,cache,device,config,weights,examples,cost):
    optimizer.zero_grad(set_to_none=True)
    labels=cache['truth'][indices].to(device)
    with autocast_context(device,config['optimization']):
        delta=adapter(cache['evidence'][indices].to(device)).float()
        logits=cache['baseline_logits'][indices].to(device)+delta
        loss=weighted_answer_loss(logits,labels,weights,examples[indices].to(device))
        if cost:
            # -log(1-pFOUND), calculated stably; no gradients into frozen V2.
            false_found=-(torch.logsumexp(logits[:,1:],-1)-torch.logsumexp(logits,-1))
            penalty_weight=(labels==2).float()*2+(labels==3).float()*.5
            penalty=(false_found*penalty_weight).sum()/penalty_weight.sum().clamp_min(1)
            loss=loss+.1*penalty+.01*delta.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.)
    assert torch.isfinite(norm)
    optimizer.step()
    return float(loss.detach())


def key(metrics):
    return metrics['macro_f1'],metrics['per_class'][0]['recall'],-metrics['false_found']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--language-augmentation',action='store_true')
    args=parser.parse_args()
    root=workspace_path('new/outputs')/args.run_name
    if root.parent!=workspace_path('new/outputs'):raise ValueError('Invalid run name')
    root.mkdir(exist_ok=False)
    source=workspace_path('new/outputs/pcrau_target_v2_full_seed_24082026')
    source_config,checkpoint=source/'config.json',source/'checkpoints/best/model.safetensors'
    config=load_config(source_config)
    config['model']['export_answerability_evidence']=True
    config['model']['export_answerability_detail']=True
    base_config=deepcopy(config)
    config['model']['answerability_evidence_adapter']={'hidden_dim':64,'dropout':.3,'include_detail':True}
    config['experiment_id']='answerability_detail_development_only'
    config['experiment']={'language_augmentation':args.language_augmentation,
                          'train_language_prefixes':list(PREFIXES) if args.language_augmentation else ['']}
    atomic_json(root/'config.json',config)
    config_hash=sha256_file(root/'config.json')
    protected=[checkpoint,source_config,source/'evaluation/calibration/calibrator.json',
               workspace_path('new/hinhanh/19_answerability_confusion/fig19_answerability_confusion_test_iid.png')]
    hashes={str(p):sha256_file(p) for p in protected}
    atomic_json(root/'run.json',{'status':'DEVELOPMENT_ONLY','source_hashes':hashes,'seed':24082026,
                'protocol_sha256':sha256_file(workspace_path('new/docs/EXPERIMENT_ANSWERABILITY_LANGUAGE.md' if args.language_augmentation
                                                            else 'new/docs/EXPERIMENT_ANSWERABILITY_DETAIL.md')),
                'script_sha256':sha256_file(Path(__file__)),'calibration_accessed':False,'test_accessed':False})
    seed_everything(24082026);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train,dev=ArchivedPCRAUDataset(base_config,'train'),ArchivedPCRAUDataset(base_config,'dev')
    assert (len(train),len(dev))==(1600,400)
    assert {e['family_id'] for e in train.entries}.isdisjoint(e['family_id'] for e in dev.entries)
    train_loader,_=make_loader(train,base_config,False);dev_loader,_=make_loader(dev,base_config,False)
    _,sampler=make_loader(train,base_config,True)
    training_data=TrainLanguageAugmentation(train) if args.language_augmentation else train
    if args.language_augmentation:
        train_loader=DataLoader(training_data,batch_size=20,num_workers=0,shuffle=False,collate_fn=collate_samples)
    weights,source_weights=class_weights(train,device)
    base=PCRAUTargetV2(base_config).to(device).eval().requires_grad_(False)
    load_model_checkpoint(base,checkpoint,sha256_file(source_config))
    print('EXTRACT observable slot/node detail from frozen train/dev',flush=True)
    _,train_rows,train_cache=detail_capture(base,train_loader,training_data,device,base_config,weights,source_weights,root/'baseline_train')
    _,dev_rows,dev_cache=detail_capture(base,dev_loader,dev,device,base_config,weights,source_weights,root/'baseline_dev')
    baseline=answer_metrics(dev_cache['truth'].tolist(),dev_cache['baseline_logits'].argmax(-1).tolist())
    assert baseline['confusion_matrix']==[[140,0,3,25],[0,42,0,0],[4,0,34,15],[14,0,13,110]]
    # Reusable cache avoids repeated extraction if a later controlled variant is needed.
    torch.save({'train':{k:v for k,v in train_cache.items() if k!='features'},
                'dev':{k:v for k,v in dev_cache.items() if k!='features'},'source_v2_sha256':hashes[str(checkpoint)]},root/'detail_cache.pt')
    base_guess=train_cache['baseline_logits'].argmax(-1)
    hard_positive=(train_cache['truth']==0)&(base_guess!=0)
    hard_absent=(train_cache['truth']==2)&(base_guess==0)
    atomic_json(root/'mining.json',{'hard_found':int(hard_positive.sum()),'hard_absent':int(hard_absent.sum()),
                'used_labels':'train only','feature_dim':2222,'original_train_samples':1600,
                'augmented_presentations':len(training_data),'independent_train_families':320,
                'prefixes':list(PREFIXES) if args.language_augmentation else ['']})
    model=PCRAUTargetV2(config).to(device).eval().requires_grad_(False)
    missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
    assert not unexpected and all(n.startswith('answerability_adapter.') for n in missing)
    initial=AnswerabilityEvidenceAdapter(64,.3,2196).to(device)
    initial.fit_standardization(train_cache['evidence'])
    best_global=None;selected=None;outcomes={}
    for arm,cost in [('detail_control',False),('detail_cost',True)]:
        arm_root=root/arm;arm_root.mkdir()
        seed_everything(24082026)
        adapter=deepcopy(initial).to(device).train().requires_grad_(True)
        examples=torch.ones(len(training_data))
        if cost:
            examples[hard_positive]=2;examples[hard_absent]=3
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=3e-4,weight_decay=1e-3)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        zero,_=predict(adapter,dev_cache,device,config)
        assert zero['confusion_matrix']==baseline['confusion_matrix']
        diagnostic=None;best=None;best_state=None;step=0
        for epoch in range(25):
            adapter.train();sampler.set_epoch(epoch);total=0
            for indices in sampler:
                ids=[train_cache['lookup'][augmented_id(train.entries[i]['sample_id'],variant)]
                     for i in indices for variant in range(len(PREFIXES) if args.language_augmentation else 1)]
                total+=update(adapter,optimizer,ids,train_cache,device,config,weights,examples,cost)*len(ids);step+=1
            metrics,_=predict(adapter,dev_cache,device,config)
            passed=evidence_eligible(metrics,baseline) and metrics['confusion_matrix'][2][0]<=4 and metrics['macro_f1']>baseline['macro_f1'] and metrics['per_class'][0]['recall']>baseline['per_class'][0]['recall']
            record={'epoch':epoch,'global_step':step,'train_loss':total/len(training_data),'dev':metrics,'eligible':passed}
            with (arm_root/'history.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
            model.answerability_adapter.load_state_dict(adapter.state_dict())
            if diagnostic is None or key(metrics)>key(diagnostic['dev']):
                diagnostic=record
                save_checkpoint(arm_root/'checkpoints/diagnostic',model,optimizer,scheduler,epoch,step,metrics['macro_f1'],config_hash,record,replace=True)
            if passed and (best is None or key(metrics)>key(best['dev'])):
                best=record;best_state=deepcopy(adapter.state_dict())
                save_checkpoint(arm_root/'checkpoints/best',model,optimizer,scheduler,epoch,step,metrics['macro_f1'],config_hash,record,replace=True)
            if epoch%5==0 or epoch==24:
                print(json.dumps({'arm':arm,'epoch':epoch,'macro_f1':metrics['macro_f1'],'found_recall':metrics['per_class'][0]['recall'],
                                  'false_found':metrics['false_found'],'absent_to_found':metrics['confusion_matrix'][2][0],'eligible':passed}),flush=True)
        chosen=best or diagnostic
        chosen_path=arm_root/'checkpoints'/('best' if best else 'diagnostic')/'model.safetensors'
        load_model_checkpoint(model,chosen_path,config_hash)
        assert all(torch.equal(v.cpu(),model.state_dict()[n].cpu()) for n,v in base.state_dict().items())
        verified,rows=evaluate(model,dev_loader,device,config,weights,source_weights)
        assert verified['answerability']['confusion_matrix']==chosen['dev']['confusion_matrix']
        assert verified['grounding_accuracy']==326/335
        original={r['sample_id']:r for r in dev_rows}
        for row in rows:
            before=original[row['sample_id']]
            for name in ['source_probabilities','relation_edge_probabilities']:
                assert row[name]==before[name]
            assert row['spatial']['map_pixel_xy']==before['spatial']['map_pixel_xy']
        atomic_json(arm_root/'verified_dev_metrics.json',verified);write_rows(arm_root/'verified_dev_predictions.jsonl',rows)
        outcomes[arm]={'best_eligible':best,'diagnostic':diagnostic,'verified':chosen,'verified_checkpoint':str(chosen_path)}
        if best is not None and (best_global is None or key(best['dev'])>key(best_global['dev'])):
            selected=arm;best_global=best
    assert hashes=={str(p):sha256_file(p) for p in protected}
    result={'status':'DEVELOPMENT_ONLY','baseline':baseline,'arms':outcomes,'selected':selected,
            'source_hashes_unchanged':True,'grounding_source_edge_unchanged':True,'calibration_fit':False,'test_evaluated':False}
    atomic_json(root/'summary.json',result)
    if selected:
        chosen_path=Path(outcomes[selected]['verified_checkpoint'])
        atomic_json(root/'freeze_lock.json',{'checkpoint':str(chosen_path),'checkpoint_sha256':sha256_file(chosen_path),
                    'config':str(root/'config.json'),'config_sha256':config_hash,'selected_arm':selected,'epoch':best_global['epoch'],
                    'source_v2_sha256':hashes[str(checkpoint)],'calibration_opened_before_freeze':False,'test_accessed':False})
    lines=['# Answerability detail trên train/dev','','| Nhánh | Macro-F1 | Recall FOUND | False FOUND | ABSENT→FOUND | Qua cổng |','|---|---:|---:|---:|---:|---|']
    for name,m,passed in [('V2',baseline,True)]+[(n,v['verified']['dev'],v['best_eligible'] is not None) for n,v in outcomes.items()]:
        lines.append(f"| {name} | {m['macro_f1']:.6f} | {m['per_class'][0]['recall']:.2%} | {m['false_found']} | {m['confusion_matrix'][2][0]} | {passed} |")
    lines += ['',f"Chọn: {selected or 'Không có; giữ V2'}.",'Grounding dev326/335, source/edge và V2 parameters bất biến đã kiểm tra.',
              'Chọn trên dev nhiều vòng, chưa chứng minh calibration/test hoặc sửa hết lỗi answerability.']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(root),'selected':selected}),flush=True)


if __name__=='__main__':main()
