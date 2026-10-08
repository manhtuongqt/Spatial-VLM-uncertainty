#!/usr/bin/env python3
"""Frozen public Location evaluation. Separate download, inference and mask scoring.

Prepare with /usr/bin/python3; infer/report use .conda-roborefer/bin/python3.10
with user site enabled (the already calibrated torch runtime). Worker uses -s.
Primary uses the original referring prompt. Prefix-only adaptation is a secondary
diagnostic, fixed before inference. No training, threshold selection or fitting.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, re, sys, time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'new/outputs/refspatial_location_frozen_20261007'
REV = '5cb4c34a36c09962442fc5b76e2e462b75008329'
REPO = 'BAAI/RefSpatial-Bench'
BUNDLE = ROOT/'new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/bundle_release.json'
SEED = 24082026

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]

def protected():
    paths={BUNDLE,ROOT/'new/outputs/active_experimental_profile.json'}
    def walk(p):
        paths.add(p)
        obj=json.loads(p.read_text())
        def visit(v):
            if isinstance(v,dict):
                if 'path' in v and 'sha256' in v:
                    q=Path(v['path']);q=q if q.is_absolute() else p.parent/q
                    assert sha(q)==v['sha256'],q
                    paths.add(q)
                    if q.suffix=='.json' and q not in seen:
                        seen.add(q);walk(q)
                for x in v.values():visit(x)
            elif isinstance(v,list):
                for x in v:visit(x)
        visit(obj)
    seen={BUNDLE};walk(BUNDLE)
    return {str(p):sha(p) for p in sorted(paths)}

def prepare():
    import requests
    from PIL import Image
    sys.path.insert(0,str(ROOT/'new/src'))
    from pcrau.query_parser import parse_query
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Protocol already exists; use next stage.'
    def fetch(rel):
        assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
        dst=OUT/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            r=requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{rel}',timeout=60)
            r.raise_for_status();dst.write_bytes(r.content)
        return {'path':str(dst),'sha256':sha(dst),'bytes':dst.stat().st_size}
    fetch('Location/question.json')
    q=json.loads((OUT/'source/Location/question.json').read_text())
    assert len(q)==100 and len({x['id'] for x in q})==100
    files=sorted({f'Location/{x[k]}' for x in q for k in ('rgb_path','mask_path')})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        descriptors=list(pool.map(fetch,files))
    runtime=[];dimensions=Counter();scope=Counter();canonical_scope=Counter()
    for x in q:
        rgb=OUT/'source/Location'/x['rgb_path']
        with Image.open(rgb) as im:w,h=im.size
        dimensions[f'{w}x{h}']+=1
        original=x['prompt'];canonical=re.sub(r'^Please point (?:out|to)\s+','Locate ',original,count=1,flags=re.I)
        p=parse_query(original);c=parse_query(canonical)
        scope[p.predicate if p.supported else p.reason]+=1
        canonical_scope[c.predicate if c.supported else c.reason]+=1
        runtime.append({'id':x['id'],'prompt':original,'canonical_prompt':canonical,
            'backbone_prompt':original+' '+x['suffix'],'original_rgb':str(rgb),
            'original_size':[w,h],'model_rgb':str(OUT/'inputs'/f'{x["id"]:03d}_rgb.png'),
            'model_depth':str(OUT/'inputs'/f'{x["id"]:03d}_depth.png'),
            'native_depth':str(OUT/'inputs'/f'{x["id"]:03d}_native_depth.png'),
            'feature_path':str(OUT/'features'/f'{x["id"]:03d}.safetensors')})
    write(OUT/'runtime_manifest.json',runtime)
    write(OUT/'source_provenance.json',{'repo':REPO,'revision':REV,'files':descriptors,
        'question_sha256':sha(OUT/'source/Location/question.json')})
    protocol={'status':'LOCKED_BEFORE_FORWARD','samples':100,'repo':REPO,'revision':REV,
        'primary_prompt':'original question.prompt; no answer-format suffix for heatmap models',
        'secondary_prompt':'Only replace leading Please point out/to with Locate; preserve entire referent expression',
        'backbone_prompt':'original prompt + official suffix; greedy, max_new_tokens=128; both matched640 and original-size reference',
        'visual_protocol':'Generate DepthAnythingV2-Large at original RGB size; minmax uint8 grayscale RGB; then resize RGB/depth to 640x480 bilinear; same inputs for both models',
        'primary_metric':'official point-in-target-mask on all 100 original masks; map normalized coordinates back to original dimensions',
        'risk_evaluation_event':'target point outside mask; different from original composite non-FOUND OR MAP error',
        'depth_reference':'monocular relative depth, not metric ground truth',
        'MC':{'T':20,'seed':SEED,'batch_size':20},'no_training':True,'no_calibration_fit':True,
        'no_threshold_selection':True,'no_subset_selection_by_predictions':True,
        'original_dimensions':dict(dimensions),'original_parser_scope':dict(scope),
        'canonical_parser_scope':dict(canonical_scope),'protected_hashes':protected(),
        'script_sha256':sha(__file__), 'worker_sha256':sha(ROOT/'new/scripts/refspatial_location_worker.py')}
    write(OUT/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ('samples','original_dimensions','original_parser_scope','canonical_parser_scope')},ensure_ascii=False),flush=True)

def infer():
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0,str(ROOT/'new/src'))
    from pcrau.task_inference import TaskInference
    from pcrau.utils import seed_everything
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(__file__)==protocol['script_sha256']
    for p,h in protocol['protected_hashes'].items():assert sha(p)==h,p
    assert (OUT/'backbone_complete.json').exists()
    assert not (OUT/'primary_predictions.jsonl').exists()
    torch.set_num_threads(4);seed_everything(SEED)
    model=TaskInference.from_bundle(BUNDLE);entries=json.loads((OUT/'runtime_manifest.json').read_text())
    for label,key in [('primary','prompt'),('prefix_diagnostic','canonical_prompt')]:
        with (OUT/f'{label}_predictions.jsonl').open('w') as f:
            for start in range(0,100,20):
                batch=entries[start:start+20];features=[load_file(e['feature_path']) for e in batch]
                inputs={k:torch.stack([x[k].float() for x in features]) for k in features[0]}
                predictions,_=model.predict(inputs,[e[key] for e in batch],SEED)
                for e,r in zip(batch,predictions):
                    assert 'evaluation' not in r
                    f.write(json.dumps({'id':e['id'],'original_size':e['original_size'],**r},ensure_ascii=False)+'\n')
                f.flush();print(f'{label}: {start+len(batch)}/100',flush=True)
    for p,h in protocol['protected_hashes'].items():assert sha(p)==h,p
    write(OUT/'inference_complete.json',{'status':'COMPLETE','samples':100,'protected_unchanged':True,'optimizer_steps':0,'fit_calls':0})

def report():
    import numpy as np
    from PIL import Image
    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score,average_precision_score
    sys.path.insert(0,str(ROOT/'new/src'));sys.path.insert(0,str(ROOT/'RoboRefer/Evaluation'))
    from summarize_acc import text2pts
    from pcrau.calibration import expected_calibration_error
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert (OUT/'inference_complete.json').exists()
    protocol=json.loads((OUT/'protocol.json').read_text())
    q=json.loads((OUT/'source/Location/question.json').read_text())
    primary=rows(OUT/'primary_predictions.jsonl');diagnostic=rows(OUT/'prefix_diagnostic_predictions.jsonl');back=rows(OUT/'backbone_predictions.jsonl')
    assert all(len(x)==100 for x in (q,primary,diagnostic,back))
    assert [x['id'] for x in q]==[x['id'] for x in primary]==[x['id'] for x in back]
    def mask_of(x):
        m=np.asarray(Image.open(OUT/'source/Location'/x['mask_path']))
        if m.ndim==3:m=m[:,:,0]
        return m>0
    def official_score(m,pts):
        if len(pts)==0:return 0.
        h,w=m.shape;pts=np.asarray(pts,dtype=int);valid=(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
        return float(m[pts[valid,1],pts[valid,0]].sum()/len(pts))
    def point(r,m):
        x,y=r['spatial']['map_pixel_xy'];h,w=m.shape
        return [int(x/640*w),int(y/480*h)]
    paired=[]
    for x,p,d,b in zip(q,primary,diagnostic,back):
        m=mask_of(x);h,w=m.shape;bp=text2pts(b['answer'],w,h,False)
        pp,dp=point(p,m),point(d,m)
        native=text2pts(b['native_answer'],w,h,False)
        paired.append({'id':x['id'],'step':x['step'],'prompt':x['prompt'],
            'backbone_native_score':official_score(m,native),'backbone_native_points':native.tolist(),
            'backbone_score':official_score(m,bp),'backbone_points':bp.tolist(),
            'pcrau_score':official_score(m,[pp]),'pcrau_point':pp,
            'prefix_score':official_score(m,[dp]),'prefix_point':dp,
            'risk':p['decision']['risk'],'action':p['decision']['action'],
            'live44_risk':p['reference_live44_decision']['risk'],'live44_action':p['reference_live44_decision']['action'],
            'scope':p['verifiers']['P1']['scope_status'],'prefix_scope':d['verifiers']['P1']['scope_status'],
            'semantic_missing':p['task_uncertainty']['risk_evidence']['semantic_matching_missing'],
            'prefix_semantic_missing':d['task_uncertainty']['risk_evidence']['semantic_matching_missing']})
    write(OUT/'paired_evaluator.json',paired)
    def wilson(k,n):
        if not n:return None
        z=1.96;p=k/n;center=(p+z*z/(2*n))/(1+z*z/n);rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return [float(center-rad),float(center+rad)]
    score=np.array([r['pcrau_score'] for r in paired]);bs=np.array([r['backbone_score'] for r in paired]);err=1-score
    def risk_metrics(rs,risk,action):
        y=1-np.array([r['pcrau_score'] for r in rs]);p=np.array(risk);n=len(y);accepted=np.array(action)=='EXECUTE'
        order=np.argsort(p,kind='stable');curve=np.cumsum(y[order])/np.arange(1,n+1)
        c=int(accepted.sum());e=int(y[accepted].sum());pc=np.clip(p,1e-12,1-1e-12)
        return {'accepted':c,'coverage':c/n,'correct_acceptances':c-e,'errors':e,
            'selective_risk':e/c if c else None,'risk_wilson95':wilson(e,c),
            'Brier':float(np.mean((p-y)**2)),'NLL':float(-np.mean(y*np.log(pc)+(1-y)*np.log1p(-pc))),
            'ECE10':expected_calibration_error(p,y,10),'AURC':float(curve.mean()),
            'error_AUROC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'error_AUPRC':float(average_precision_score(y,p)) if len(set(y))==2 else None}
    task=risk_metrics(paired,[r['risk'] for r in paired],[r['action'] for r in paired])
    live=risk_metrics(paired,[r['live44_risk'] for r in paired],[r['live44_action'] for r in paired])
    dg=[{**r,'pcrau_score':r['prefix_score']} for r in paired]
    dm=risk_metrics(dg,[r['decision']['risk'] for r in diagnostic],[r['decision']['action'] for r in diagnostic])
    fixed=[r['id'] for r in paired if r['pcrau_score']>r['backbone_score']]
    broken=[r['id'] for r in paired if r['pcrau_score']<r['backbone_score']]
    rng=np.random.default_rng(SEED);ix=rng.integers(0,100,(10000,100));delta=(score-bs)[ix].mean(1)
    stats={}
    for key,k in [('spatial_MI','spatial_target_MI'),('semantic_MI','semantic_target_MI'),('completion_MI','completion_MI')]:
        v=np.array([r['task_uncertainty']['risk_evidence'][k] for r in primary])
        stats[key]={'mean':float(v.mean()),'error_AUROC':float(roc_auc_score(err,v)) if len(set(err))==2 else None}
    summary={'status':'COMPLETE_FROZEN_PUBLIC_LOCATION','samples':100,
        'RoboRefer_2B_SFT':{'score':float(bs.mean()),'perfect_cases':int((bs==1).sum()),'point_counts':dict(Counter(len(r['backbone_points']) for r in paired))},
        'RoboRefer_2B_SFT_native_size':{'score':float(np.mean([r['backbone_native_score'] for r in paired])),
            'perfect_cases':sum(r['backbone_native_score']==1 for r in paired)},
        'P_CRA_U_Tasks60':{'hits':int(score.sum()),'PIT':float(score.mean()),'wilson95':wilson(int(score.sum()),100),**task},
        'Live44_same_MAP':live,'prefix_only_diagnostic':{'hits':int(sum(r['prefix_score'] for r in paired)),**dm},
        'paired':{'fixed':fixed,'broken':broken,'delta_pp':float((score-bs).mean()*100),
            'sample_bootstrap_delta95_pp':(np.quantile(delta,[.025,.975])*100).tolist(),
            'bootstrap_unit':'question; descriptive, scene duplicates may invalidate independence'},
        'scope':dict(Counter(r['scope'] for r in paired)),
        'prefix_scope':dict(Counter(r['prefix_scope'] for r in paired)),
        'semantic_missing':sum(r['semantic_missing'] for r in paired),
        'prefix_semantic_missing':sum(r['prefix_semantic_missing'] for r in paired),
        'by_step':{str(s):{'n':sum(r['step']==s for r in paired),
            'backbone_score':float(np.mean([r['backbone_score'] for r in paired if r['step']==s])),
            'pcrau_score':float(np.mean([r['pcrau_score'] for r in paired if r['step']==s]))} for s in sorted({r['step'] for r in paired})},
        'uncertainty_error_detection':stats,'bundle_sha256':sha(BUNDLE),
        'risk_range':[float(min(r['risk'] for r in paired)),float(max(r['risk'] for r in paired))],
        'predicted_answerability':dict(Counter(r['decision']['predicted_answerability'] for r in primary)),
        'action_counts':dict(Counter(r['action'] for r in paired)),
        'threshold':primary[0]['decision']['threshold'],'primary_event':'point outside target mask',
        'answerability_ground_truth_available':False,'anchor_ground_truth_available':False,
        'metric_depth_ground_truth_available':False,'amodal_completion_ground_truth_available':False,
        'training_steps':0,'calibration_fit_calls':0,'threshold_tuned_on_benchmark':False}
    write(OUT/'summary.json',summary)
    # All actual images/masks below are evaluator overlays, after predictions.
    ids=(fixed[:2]+broken[:2]+[r['id'] for r in paired if r['pcrau_score']==1][:2])
    ids=list(dict.fromkeys(ids))
    for r in paired:
        if len(ids)>=6:break
        if r['id'] not in ids:ids.append(r['id'])
    fig,axs=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(hspace=.42)
    import textwrap
    for ax,i in zip(axs.flat,ids):
        j=next(k for k,x in enumerate(q) if x['id']==i);x=q[j];r=paired[j]
        im=np.asarray(Image.open(OUT/'source/Location'/x['rgb_path']).convert('RGB'));m=mask_of(x)
        ax.imshow(im);ax.contour(m,levels=[.5],colors=['lime'],linewidths=1)
        px,py=r['pcrau_point'];ax.scatter([px],[py],c='red',marker='x',s=100,label='P-CRA-U')
        if r['backbone_points']:
            z=np.asarray(r['backbone_points']);ax.scatter(z[:,0],z[:,1],c='cyan',marker='+',s=100,label='RoboRefer')
        ax.set_title(textwrap.fill(f"#{i}: {x['prompt']}",48)+f"\nP={r['pcrau_score']:.0f}; R={r['backbone_score']:.2f}; risk={r['risk']:.3f}; {r['action']}",fontsize=9)
        ax.axis('off');ax.legend(fontsize=7)
    fig.savefig(OUT/'cases.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for name,p in [('Tasks60',[r['risk'] for r in paired]),('Live44',[r['live44_risk'] for r in paired])]:
        idx=np.argsort(p,kind='stable');ax.plot(np.arange(1,101)/100,np.cumsum(err[idx])/np.arange(1,101),label=name)
    ax.axhline(.075,color='gray',ls='--',label='7.5% reference');ax.set(xlabel='Coverage',ylabel='Grounding error risk',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=160);plt.close(fig)
    lines=['# RefSpatial-Bench Location — đánh giá bundle đóng băng','',
        'Ngày 07/10/2026. Chấm đủ 100 mẫu; không train, không fit calibration, không chọn lại ngưỡng.',
        '', '## Protocol',
        f'- Dataset `{REPO}`, revision `{REV}`.',
        '- Primary: nguyên văn `prompt`. RoboRefer thêm suffix định dạng chính thức; Sidecar xuất heatmap nên không thêm suffix.',
        '- RGB và pseudo-depth Depth Anything V2 Large resize 640×480 cho cả hai. Điểm được đưa về kích thước mask gốc; masks chỉ dùng ở evaluator.',
        '- Secondary: chỉ thay tiền tố `Please point out/to` bằng `Locate`; không rút gọn các quan hệ/cụm danh từ. Không dùng để chọn primary sau khi thấy kết quả.',
        '- Risk hậu kiểm là MAP ngoài mask, khác event composite đã fit trên dữ liệu đồ án. Không có GT bốn trạng thái answerability.',
        '', '## Kết quả', '| Pipeline | PIT | Coverage | Selective grounding risk |', '|---|---:|---:|---:|',
        f'| RoboRefer-2B-SFT greedy | {100*bs.mean():.2f}% | — | — |',
        f'| RoboRefer-2B-SFT, ảnh nguyên kích thước | {100*summary["RoboRefer_2B_SFT_native_size"]["score"]:.2f}% | — | — |',
        f'| P-CRA-U Tasks60 MC, primary | {int(score.sum())}/100 | {100*task["coverage"]:.2f}% | {task["selective_risk"]} |',
        f'| Live44, cùng MAP | {int(score.sum())}/100 | {100*live["coverage"]:.2f}% | {live["selective_risk"]} |',
        f'| Prefix-only diagnostic | {summary["prefix_only_diagnostic"]["hits"]}/100 | {100*dm["coverage"]:.2f}% | {dm["selective_risk"]} |',
        '',f'Primary scope: `{summary["scope"]}`; secondary scope: `{summary["prefix_scope"]}`.',
        f'Primary semantic phrase missing: {summary["semantic_missing"]}/100; secondary: {summary["prefix_semantic_missing"]}/100.',
        '', 'Calibration/ranking (event grounding error):', '```json', json.dumps(task,indent=2), '```',
        '',f'Paired cases tốt hơn RoboRefer: {fixed}; kém hơn: {broken}.',
        f'Delta: {summary["paired"]["delta_pp"]:.2f} pp; bootstrap descriptive CI: {summary["paired"]["sample_bootstrap_delta95_pp"]}.',
        '', '## Case thật', '![Actual cases](../new/outputs/refspatial_location_frozen_20261007/cases.png)',
        'Xanh lá: mask GT hậu kiểm; đỏ: P-CRA-U; cyan: RoboRefer.',
        '![Risk coverage](../new/outputs/refspatial_location_frozen_20261007/risk_coverage.png)',
        '', '## Giới hạn và cách diễn giải',
        '- Đây là đánh giá chuyển miền sang ảnh thực/câu lệnh công khai, không phải train lại trên benchmark. Backbone đã được huấn luyện RefSpatial; không tuyên bố hoàn toàn unseen đối với cả VLM.',
        '- Không chấm anchor hit, depth MAE theo mét hoặc amodal completion vì benchmark không cung cấp các nhãn này. Không chứng minh cả năm nhóm uncertainty chỉ từ PIT.',
        '- Verifier chỉ hoạt động khi parser hỗ trợ. Unsupported vẫn có MAP/risk từ pipeline, nhưng không được diễn giải là đã kiểm chứng quan hệ.',
        '- Nếu coverage=0 thì selective risk không xác định, không phải risk=0 hoặc bảo đảm an toàn.',
        '- Nếu điểm thấp hơn IID, đó là bằng chứng giới hạn chuyển miền. Không dùng kết quả này để sửa ngưỡng rồi báo lại như test độc lập.',
        '- Public benchmark score tương thích point-in-mask; chưa là submission leaderboard. Resize/greedy/cap được ghi rõ; không đồng nhất điểm với cấu hình RFT/8B trong paper.',
        '', '## Artifacts',f'- `{OUT.relative_to(ROOT)}/summary.json`, `paired_evaluator.json`, predictions, protocol, source hashes.',
        '- Sources: https://huggingface.co/datasets/BAAI/RefSpatial-Bench ; https://github.com/Zhoues/RoboRefer/tree/main/Evaluation .']
    (ROOT/'plan/REFSPATIAL_LOCATION_EVALUATION_20261007.md').write_text('\n'.join(lines)+'\n')
    for p,h in protocol['protected_hashes'].items():assert sha(p)==h,p
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['prepare','infer','report']);args=a.parse_args()
    {'prepare':prepare,'infer':infer,'report':report}[args.stage]()
