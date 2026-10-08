#!/usr/bin/env python3
"""Train/dev-only binary support experiment on the frozen answerability model."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcrau.acceptance_ranker import AcceptanceRanker
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import choose_policy_threshold, policy_result, probability_metrics
from pcrau.support_evidence import SUPPORT_NAMES, ranker_features
from pcrau.text import prompt_anchor_mask, relation_ids
from pcrau.utils import atomic_json, load_config, seed_everything, sha256_file, workspace_path


def write_rows(path, rows):
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')


def diagnostics(rows, scores):
    eligible=np.array([max(r['answerability_probabilities'], key=r['answerability_probabilities'].get)=='FOUND' for r in rows])
    error=np.array([r['evaluation']['error_event'] for r in rows])
    threshold=choose_policy_threshold(scores,error,[r['family_id'] for r in rows],eligible,.075,20)
    metrics, records=policy_result(rows,scores,'hard_found',threshold['threshold'])
    groups={v:sum(r['action']=='EXECUTE' and r['evaluation']['error_event']
                  for r,source in zip(records,rows) if source['variant']==v)
            for v in ['relation_counterfactual','occlusion_view_counterfactual']}
    return {'probability':probability_metrics(scores,error),
            'conditional_aurc':probability_metrics(scores[eligible],error[eligible])['aurc'],
            'threshold':threshold,'policy':metrics,'group_errors':groups}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='pcrau_support_split_dev_20261003')
    args=parser.parse_args()
    root=workspace_path('new/outputs')/args.run_name
    if root.parent!=workspace_path('new/outputs'):raise ValueError('Invalid output root')
    root.mkdir(exist_ok=False)
    source=workspace_path('new/outputs/pcrau_answerability_language_dev_20261003')
    lock=json.loads((source/'freeze_lock.json').read_text())
    assert sha256_file(Path(lock['checkpoint']))==lock['checkpoint_sha256']
    assert sha256_file(Path(lock['config']))==lock['config_sha256']
    config=load_config(lock['config']);torch.set_num_threads(4);seed_everything(24082026)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frozen=PCRAUTargetV2(config).to(device).eval().requires_grad_(False)
    load_model_checkpoint(frozen,lock['checkpoint'],lock['config_sha256'])
    protected=json.loads((source/'run.json').read_text())['source_hashes']
    protected.update({str(workspace_path(f'latex/Chapter{i}/chapter{i}.tex')):
                     sha256_file(workspace_path(f'latex/Chapter{i}/chapter{i}.tex')) for i in range(1,7)})
    atomic_json(root/'run.json',{'status':'TRAIN_DEV_ONLY','source_lock':lock,'protected_files':protected,
        'prior_test_analysis_informed_research_question':True,'test_samples_used_for_training':False,
        'calibration_opened':False,'protocol_sha256':sha256_file(workspace_path('new/docs/EXPERIMENT_SUPPORT_SPLIT_CALIBRATION.md')),
        'script_sha256':sha256_file(Path(__file__)),'epochs':25,'arms':['detail_uniform','scalar_hard','detail_hard','support_hard']})
    cache=torch.load(source/'detail_cache.pt',map_location='cpu',weights_only=True)
    assert cache['source_v2_sha256']==lock['source_v2_sha256']
    rows={}; probabilities={}; masks={}; features={}; errors={}; offsets={}
    for split in ['train','dev']:
        c=cache[split]; dataset=ArchivedPCRAUDataset(config,split)
        original={e['sample_id']:e for e in dataset.entries}
        baseline=source/f'baseline_{split}'/'predictions.jsonl'
        rows[split]=[json.loads(line) for line in baseline.open()]
        assert [r['sample_id'] for r in rows[split]]==c['sample_ids']
        logits=[]
        with torch.no_grad():
            for a,b in c['batch_ranges']:
                with autocast_context(device,config['optimization']):
                    delta=frozen.answerability_adapter(c['evidence'][a:b].to(device)).float().cpu()
                logits.append(c['baseline_logits'][a:b]+delta)
        logits=torch.cat(logits); probabilities[split]=logits.softmax(-1)
        offsets[split]=torch.logsumexp(logits[:,1:],-1)-logits[:,0]
        anchor=[];relation=[]
        for row,p in zip(rows[split],probabilities[split]):
            original_id=row['sample_id'].split('__language_prefix_')[0]
            e=original[original_id]; prompt=e.get('feature_input',e).get('prompt',e.get('prompt',''))
            anchor.append(prompt_anchor_mask(prompt,3));relation.append(relation_ids(prompt,3)[1])
            row['answerability_probabilities']={name:float(p[i]) for i,name in enumerate(config['model']['answerability_classes'])}
        masks[split]=(torch.tensor(anchor,dtype=torch.bool),torch.tensor(relation,dtype=torch.bool))
        errors[split]=torch.tensor([r['evaluation']['error_event'] for r in rows[split]],dtype=torch.float)
        features[split]={m:ranker_features(c['evidence'],probabilities[split],*masks[split],m).contiguous()
                         for m in ['scalar','detail','support']}
    assert len(rows['train'])==6400 and len(rows['dev'])==400
    assert set(cache['train']['families']).isdisjoint(cache['dev']['families'])
    assert int((probabilities['dev'].argmax(-1)==cache['dev']['truth']).sum())==349
    reference=diagnostics(rows['dev'],torch.sigmoid(offsets['dev']).numpy())
    atomic_json(root/'reference_dev.json',reference)
    print('REFERENCE '+json.dumps(reference),flush=True)
    guess=probabilities['train'].argmax(-1); safe=errors['train']==0
    positive=safe & ((guess!=0)|(probabilities['train'][:,0]<.9))
    negative=(~safe)&((guess==0)|(probabilities['train'][:,0]>.5))
    weights=torch.ones(6400);weights[positive]=2;weights[negative]=3
    focus=torch.tensor([r['variant'] in ['relation_counterfactual','occlusion_view_counterfactual'] for r in rows['train']])
    weights[focus]*=1.5
    mining=[]
    for i,row in enumerate(rows['train']):
        if '__language_prefix_' not in row['sample_id']:
            mining.append({'sample_id':row['sample_id'],'family_id':row['family_id'],'variant_evaluator_only':row['variant'],
                'truth':row['evaluation']['answerability_state'],'map_correct':row['evaluation']['map_inside_target'],
                'hard_positive':bool(positive[i]),'hard_negative':bool(negative[i]),'training_weight':float(weights[i])})
    write_rows(root/'mined_train_samples.jsonl',mining)
    atomic_json(root/'mining_summary.json',{'original_train_samples':len(mining),'language_presentations':6400,
        'independent_families':320,'hard_positive_unique':sum(r['hard_positive'] for r in mining),
        'hard_negative_unique':sum(r['hard_negative'] for r in mining),'feature_dims':{m:f.shape[-1] for m,f in features['train'].items()},
        'support_names':SUPPORT_NAMES,'hard_groups':{v:{k:sum(r[k] for r in mining if r['variant_evaluator_only']==v)
                                                   for k in ['hard_positive','hard_negative']}
                                                   for v in ['relation_counterfactual','occlusion_view_counterfactual']}})
    grouped={f:[i for i,x in enumerate(cache['train']['families']) if x==f] for f in sorted(set(cache['train']['families']))}
    families=list(grouped); global_best=None; outcomes={}; initial={}
    for arm,mode,hard in [('detail_uniform','detail',False),('scalar_hard','scalar',True),
                          ('detail_hard','detail',True),('support_hard','support',True)]:
        seed_everything(24082026)
        head=AcceptanceRanker(features['train'][mode].shape[-1],64,.3,zero_output=True).to(device)
        head.fit_standardization(features['train'][mode].to(device))
        if mode in initial:head.load_state_dict(initial[mode])
        else:initial[mode]=deepcopy(head.state_dict())
        arm_root=root/arm;arm_root.mkdir(); optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-3)
        sample_weight=weights if hard else torch.ones_like(weights)
        xtrain=features['train'][mode].to(device);xdev=features['dev'][mode].to(device)
        y=errors['train'].to(device);sw=sample_weight.to(device);offset=offsets['train'].to(device)
        best=None;diagnostic=None
        for epoch in range(25):
            head.train(); order=torch.randperm(len(families),generator=torch.Generator().manual_seed(24082026+epoch)).tolist(); total=0
            for start in range(0,len(order),4):
                ids=[i for k in order[start:start+4] for i in grouped[families[k]]]
                optimizer.zero_grad(set_to_none=True)
                delta=head(xtrain[ids]); score=offset[ids]+delta
                bce=(F.binary_cross_entropy_with_logits(score,y[ids],reduction='none')*sw[ids]).sum()/sw[ids].sum()
                bad,good=score[y[ids]>.5],score[y[ids]<=.5]
                ranking=F.softplus(good[:,None]-bad[None,:]).mean() if len(bad) and len(good) else score.sum()*0
                loss=bce+.2*ranking+.01*delta.square().mean()
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.);assert torch.isfinite(norm)
                optimizer.step();total+=float(loss.detach())*len(ids)
            head.eval()
            with torch.no_grad():raw=(offsets['dev'].to(device)+head(xdev)).cpu(); scores=raw.sigmoid().numpy()
            m=diagnostics(rows['dev'],scores)
            passed=(m['policy']['correct_acceptances']>=reference['policy']['correct_acceptances']
                and m['conditional_aurc']<reference['conditional_aurc']
                and m['policy']['accepted_absent_truth']<=reference['policy']['accepted_absent_truth']
                and all(m['group_errors'][v]<=reference['group_errors'][v] for v in m['group_errors']))
            key=(m['policy']['correct_acceptances'],-m['conditional_aurc'],-m['probability']['brier'])
            rec={'epoch':epoch,'train_loss':total/6400,'dev':m,'eligible':passed,'selection_key':key}
            with (arm_root/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            saved={'state_dict':deepcopy(head.cpu().state_dict()),'input_dim':head.input_dim,'hidden_dim':64,'dropout':.3,
                'mode':mode,'source_checkpoint_sha256':lock['checkpoint_sha256'],'config_sha256':lock['config_sha256'],
                'epoch':epoch,'training_mining':hard,'dev':m}
            head.to(device)
            if diagnostic is None or key>tuple(diagnostic['selection_key']):
                diagnostic=rec;torch.save(saved,arm_root/'diagnostic.pt')
            if passed and (best is None or key>tuple(best['selection_key'])):
                best=rec;torch.save(saved,arm_root/'best.pt')
            if epoch%5==0 or epoch==24:print(json.dumps({'arm':arm,'epoch':epoch,'dev_correct':m['policy']['correct_acceptances'],
                'dev_errors':m['policy']['errors'],'conditional_aurc':m['conditional_aurc'],'group_errors':m['group_errors'],'eligible':passed}),flush=True)
        outcomes[arm]={'best_eligible':best,'diagnostic':diagnostic}
        if best and (global_best is None or tuple(best['selection_key'])>tuple(global_best[1]['selection_key'])):global_best=(arm,best)
        del xtrain,xdev,head
    selected=global_best[0] if global_best else None
    atomic_json(root/'summary.json',{'status':'TRAIN_DEV_COMPLETE','selected':selected,'reference':reference,'arms':outcomes,
                                    'calibration_opened':False,'test_evaluated':False})
    if selected:
        checkpoint=root/selected/'best.pt'
        atomic_json(root/'freeze_lock.json',{'source_checkpoint':lock['checkpoint'],'source_checkpoint_sha256':lock['checkpoint_sha256'],
            'config':lock['config'],'config_sha256':lock['config_sha256'],'ranker_checkpoint':str(checkpoint),
            'ranker_sha256':sha256_file(checkpoint),'selected_arm':selected,'selected_epoch':global_best[1]['epoch'],
            'calibration_opened_before_freeze':False,'historical_test_not_reopened':True})
    for p,h in protected.items():assert sha256_file(Path(p))==h,p
    print('SELECTED '+str(selected),flush=True)


if __name__=='__main__':main()
