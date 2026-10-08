#!/usr/bin/env python3
"""Image-balanced RefCOCO UNC val pilot; frozen models, point-in-box scoring.

Use .conda-roborefer/bin/python3.10 -B (calibrated user-site torch enabled).
Worker uses the same interpreter with -s/PYTHONNOUSERSITE=1.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,io
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'new/outputs/refcoco_val_pilot300_20261007'
SOURCE=ROOT/'new/outputs/refcoco_val_frozen_20261007/source/val.parquet'
REPO='jxu124/refcoco-benchmark'
REV='2f2f892835bbf44b4db81f1c4ad6d46a3e9e4359'
SOURCE_SHA='9f2f0057259be527249e0d260511ace5869b9055c23b65eb951d01e6e6920424'
BUNDLE=ROOT/'new/outputs/pcrau_task_uncertainty_20261007/evaluation_r3/bundle_release.json'
SEED=24082026
sys.path.insert(0,str(ROOT/'new/scripts'))
from evaluate_refspatial_location import protected,sha,write,rows

def key(kind,value):return hashlib.sha256(f'{SEED}:{kind}:{value}'.encode()).hexdigest()

def prompt_for(phrase):
    s=phrase.strip().rstrip('.!?')
    if s.lower().startswith(('the ','a ','an ')):return 'Locate '+s+'.'
    return 'Locate the '+s+'.'

def prepare():
    import pyarrow.parquet as pq
    from PIL import Image
    sys.path.insert(0,str(ROOT/'new/src'));from pcrau.query_parser import parse_query
    assert sha(SOURCE)==SOURCE_SHA
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    dataset=pq.read_table(SOURCE).to_pylist()
    nref=sum(len(x['ref_list']) for x in dataset)
    nquery=sum(len(r['ref_info']['sentences']) for x in dataset for r in x['ref_list'])
    assert len(dataset)==1500 and nref==3811 and nquery==10834
    assert all(r['ref_info']['split']=='val' for x in dataset for r in x['ref_list'])
    eligible=[x for x in dataset if sum(len(r['ref_info']['sentences']) for r in x['ref_list'])>=3]
    selected=sorted(eligible,key=lambda x:key('image',x['image_info']['id']))[:100]
    suffix='Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], where the coordinates are between 0 and 1, indicating the normalized pixel location of one point satisfying the conditions above.'
    runtime=[];evaluation=[];scope=Counter();refs=set();imgmeta=[]
    for x in selected:
        info=x['image_info'];iid=info['id']
        assert Path(info['file_name']).name==info['file_name'] and info['file_name'].startswith('COCO_train2014_')
        rgb=OUT/'source_images'/info['file_name'];rgb.parent.mkdir(exist_ok=True)
        rgb.write_bytes(x['image']['bytes'])
        with Image.open(rgb) as im:w,h=im.size
        assert [w,h]==[info['width'],info['height']]
        candidates=[]
        for r in x['ref_list']:
            ri,ann=r['ref_info'],r['ann_info'];assert ri['ann_id']==ann['id'] and ann['image_id']==iid
            for s in ri['sentences']:candidates.append((r,s))
        for r,s in sorted(candidates,key=lambda v:key('sentence',v[1]['sent_id']))[:3]:
            ri,ann=r['ref_info'],r['ann_info'];sid=s['sent_id'];phrase=s['raw'];prompt=prompt_for(phrase)
            parsed=parse_query(prompt);scope[parsed.predicate if parsed.supported else parsed.reason]+=1;refs.add(ri['ref_id'])
            runtime.append({'id':sid,'image_id':iid,'phrase':phrase,'prompt':prompt,'backbone_prompt':prompt+' '+suffix,
                'original_size':[w,h],'original_rgb':str(rgb),'rgb_sha256':sha(rgb),
                'model_rgb':str(OUT/'inputs'/f'{iid}_rgb.png'),'model_depth':str(OUT/'inputs'/f'{iid}_depth.png'),
                'feature_path':str(OUT/'features'/f'{iid}.safetensors')})
            evaluation.append({'id':sid,'image_id':iid,'ref_id':ri['ref_id'],'ann_id':ann['id'],
                'bbox_xywh':ann['bbox'],'category_id':ri['category_id'],'phrase':phrase,'image_size':[w,h]})
        imgmeta.append({'image_id':iid,'path':str(rgb),'sha256':sha(rgb),'size':[w,h]})
    assert len(runtime)==300 and len({x['id'] for x in runtime})==300
    write(OUT/'runtime_manifest.json',runtime);write(OUT/'evaluator_manifest.json',evaluation)
    protocol={'status':'LOCKED_BEFORE_INFERENCE','dataset':'RefCOCO UNC val','pilot':True,
        'source_repo':REPO,'source_revision':REV,'source_path':str(SOURCE),'source_sha256':SOURCE_SHA,
        'full_val':{'images':1500,'references':3811,'expressions':10834},
        'selection':'SHA256(seed:image:image_id) lowest 100 images with >=3 sentences; per selected image SHA256(seed:sentence:sent_id) lowest 3 expressions; no model predictions used',
        'eligible_images':len(eligible),'excluded_images_with_less_than_3_sentences':1500-len(eligible),
        'samples':300,'images':100,'references':len(refs),'seed':SEED,'parser_scope_before_forward':dict(scope),
        'prompt_rule':'Locate <phrase> if phrase has initial the/a/an; otherwise Locate the <phrase>; preserve raw referent wording; remove terminal .!? and add period',
        'baseline_native':'RoboRefer-2B-SFT RGB-only original-size, greedy max_new_tokens128',
        'baseline_matched':'Same RoboRefer-2B-SFT RGB-D with same 640x480 inputs as Sidecar, greedy max_new_tokens128',
        'depth_rule':'DepthAnythingV2 Large, input_size518 on original RGB; minmax uint8 repeat3; resize RGB/depth bilinear to640x480',
        'MC':{'T':20,'batch_size':20,'seed':SEED},
        'metric':'point-in-GT-bbox; original xywh; inclusive rectangle edges but image bounds [0,w),[0,h); one point expected, no valid point = failure',
        'risk_event':'MAP outside GT bbox, different from original composite event',
        'CI':'10,000 bootstrap image-id groups, micro-average over 3 expressions/image',
        'runtime_label_input':False,'train_steps':0,'fit_calls':0,'threshold_selection':False,
        'protected_hashes':protected(),'script_sha256':sha(__file__),
        'worker_sha256':sha(ROOT/'new/scripts/refcoco_pilot_worker.py'),
        'runtime_manifest_sha256':sha(OUT/'runtime_manifest.json'),'evaluator_manifest_sha256':sha(OUT/'evaluator_manifest.json'),
        'image_inventory':imgmeta}
    write(OUT/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ('samples','images','references','eligible_images','parser_scope_before_forward')},ensure_ascii=False))

def infer():
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0,str(ROOT/'new/src'))
    from pcrau.task_inference import TaskInference
    from pcrau.utils import seed_everything
    protocol=json.loads((OUT/'protocol.json').read_text());assert sha(__file__)==protocol['script_sha256']
    for p,h in protocol['protected_hashes'].items():assert sha(p)==h,p
    assert sha(OUT/'runtime_manifest.json')==protocol['runtime_manifest_sha256']
    assert (OUT/'backbone_complete.json').exists();assert not (OUT/'predictions.jsonl').exists()
    torch.set_num_threads(4);seed_everything(SEED);m=TaskInference.from_bundle(BUNDLE)
    entries=json.loads((OUT/'runtime_manifest.json').read_text())
    with (OUT/'predictions.jsonl').open('w') as f:
        for start in range(0,len(entries),20):
            b=entries[start:start+20];features=[load_file(e['feature_path']) for e in b]
            inputs={k:torch.stack([x[k].float() for x in features]) for k in features[0]}
            pred,_=m.predict(inputs,[e['prompt'] for e in b],SEED)
            for e,r in zip(b,pred):
                assert 'evaluation' not in r
                f.write(json.dumps({'id':e['id'],'image_id':e['image_id'],'original_size':e['original_size'],**r},ensure_ascii=False)+'\n')
            f.flush();print(f'SIDECAR {min(start+20,len(entries))}/{len(entries)}',flush=True)
    for p,h in protocol['protected_hashes'].items():assert sha(p)==h,p
    write(OUT/'inference_complete.json',{'samples':len(entries),'no_training':True,'no_calibration_fit':True})

def point_inside(point,box,size):
    x,y=point;bx,by,bw,bh=box;w,h=size
    return bool(0<=x<w and 0<=y<h and bx<=x<=bx+bw and by<=y<=by+bh)

def report():
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    from PIL import Image
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.patches import Rectangle
    sys.path.insert(0,str(ROOT/'new/src'));sys.path.insert(0,str(ROOT/'RoboRefer/Evaluation'))
    from summarize_acc import text2pts
    from pcrau.calibration import expected_calibration_error
    assert (OUT/'inference_complete.json').exists()
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'evaluator_manifest.json')==protocol['evaluator_manifest_sha256']
    ev=json.loads((OUT/'evaluator_manifest.json').read_text());runtime=json.loads((OUT/'runtime_manifest.json').read_text())
    preds=rows(OUT/'predictions.jsonl');back=rows(OUT/'backbone_predictions.jsonl')
    assert len(preds)==len(back)==len(ev)==300 and [r['id'] for r in preds]==[r['id'] for r in back]==[r['id'] for r in ev]
    paired=[]
    for e,r,b in zip(ev,preds,back):
        w,h=e['image_size'];px,py=r['spatial']['map_pixel_xy'];p=[int(px/640*w),int(py/480*h)]
        def decode(text):
            pts=text2pts(text,w,h,False).tolist()
            if len(pts)!=1:return pts,False
            return pts,point_inside(pts[0],e['bbox_xywh'],[w,h])
        bp,bhit=decode(b['answer']);npnt,nhit=decode(b['native_rgb_answer'])
        paired.append({**e,'pcrau_point':p,'pcrau_hit':point_inside(p,e['bbox_xywh'],[w,h]),
            'backbone_matched_points':bp,'backbone_matched_hit':bhit,'backbone_native_rgb_points':npnt,'backbone_native_rgb_hit':nhit,
            'risk':r['decision']['risk'],'action':r['decision']['action'],
            'live44_risk':r['reference_live44_decision']['risk'],'live44_action':r['reference_live44_decision']['action'],
            'scope':r['verifiers']['P1']['scope_status'],
            'semantic_phrase_missing':r['task_uncertainty']['risk_evidence']['semantic_matching_missing']})
    write(OUT/'paired_evaluator.json',paired)
    hit=np.array([r['pcrau_hit'] for r in paired],dtype=float);err=1-hit;bh=np.array([r['backbone_matched_hit'] for r in paired],dtype=float);nh=np.array([r['backbone_native_rgb_hit'] for r in paired],dtype=float)
    def rm(risks,actions):
        p=np.array(risks);accepted=np.array(actions)=='EXECUTE';c=int(accepted.sum());e=int(err[accepted].sum());pc=np.clip(p,1e-12,1-1e-12);idx=np.argsort(p,kind='stable')
        return {'accepted':c,'coverage':c/300,'errors':e,'correct_acceptances':c-e,'selective_grounding_risk':e/c if c else None,
            'Brier':float(np.mean((p-err)**2)),'NLL':float(-np.mean(err*np.log(pc)+(1-err)*np.log1p(-pc))),
            'ECE10':expected_calibration_error(p,err),'AURC':float(np.mean(np.cumsum(err[idx])/np.arange(1,301))),
            'error_AUROC':float(roc_auc_score(err,p)) if len(set(err))==2 else None,'error_AUPRC':float(average_precision_score(err,p)) if len(set(err))==2 else None}
    ids=sorted({r['image_id'] for r in paired});groups=[[i for i,r in enumerate(paired) if r['image_id']==iid] for iid in ids];assert len(groups)==100 and all(len(g)==3 for g in groups)
    rng=np.random.default_rng(SEED);draw=rng.integers(0,100,(10000,100));indices=np.asarray(groups)[draw].reshape(10000,300)
    delta=(hit-bh)[indices].mean(1);deltan=(hit-nh)[indices].mean(1)
    fixed=[r['id'] for r in paired if r['pcrau_hit'] and not r['backbone_matched_hit']];broken=[r['id'] for r in paired if not r['pcrau_hit'] and r['backbone_matched_hit']]
    tasks=rm([r['risk'] for r in paired],[r['action'] for r in paired]);live=rm([r['live44_risk'] for r in paired],[r['live44_action'] for r in paired])
    summary={'status':'COMPLETE_REFCOCO_UNC_VAL_PILOT_NOT_FULL_SPLIT','samples':300,'images':100,'references':protocol['references'],
        'full_val_expressions':10834,'bundle_sha256':sha(BUNDLE),'threshold':preds[0]['decision']['threshold'],
        'P_CRA_U_Tasks60':{'hits':int(hit.sum()),'point_in_box':float(hit.mean()),**tasks},
        'Live44_same_MAP':live,'RoboRefer_RGBD_matched640':{'hits':int(bh.sum()),'point_in_box':float(bh.mean())},
        'RoboRefer_RGB_native':{'hits':int(nh.sum()),'point_in_box':float(nh.mean())},
        'scope':dict(Counter(r['scope'] for r in paired)),'semantic_phrase_missing':sum(r['semantic_phrase_missing'] for r in paired),
        'answerability_predictions':dict(Counter(r['decision']['predicted_answerability'] for r in preds)),
        'actions':dict(Counter(r['action'] for r in paired)),
        'paired':{'fixed_ids':fixed,'broken_ids':broken,'delta_matched_pp':float(100*(hit-bh).mean()),
            'delta_matched95_pp':(100*np.quantile(delta,[.025,.975])).tolist(),'delta_native_pp':float(100*(hit-nh).mean()),
            'delta_native95_pp':(100*np.quantile(deltan,[.025,.975])).tolist(),
            'pcrau_PIT95_percent':(100*np.quantile(hit[indices].mean(1),[.025,.975])).tolist(),'CI_unit':'image_id; 10000 cluster bootstrap'},
        'risk_event':'MAP outside bbox; not original composite event','train_steps':0,'fit_calls':0,'threshold_tuned_on_pilot':False,
        'risk_range':[float(min(r['risk'] for r in paired)),float(max(r['risk'] for r in paired))],
        'backbone_point_counts':dict(Counter(len(r['backbone_matched_points']) for r in paired)),
        'comparison_to_paper_exact_reproduction':False}
    # Point membership implementation independently cross-checked at every sample.
    for r in paired:
        x,y=r['pcrau_point'];bx,by,bw,bhh=r['bbox_xywh'];w,h=r['image_size']
        assert r['pcrau_hit']==bool((np.array([x,y])>=np.array([bx,by])).all() and (np.array([x,y])<=np.array([bx+bw,by+bhh])).all() and 0<=x<w and 0<=y<h)
    write(OUT/'summary.json',summary)
    accepted=[r['id'] for r in paired if r['action']=='EXECUTE' and not r['pcrau_hit']]
    chosen=list(dict.fromkeys(fixed[:2]+broken[:2]+accepted[:1]+[r['id'] for r in paired if r['pcrau_hit']][:1]))
    for r in paired:
        if len(chosen)>=6:break
        if r['id'] not in chosen:chosen.append(r['id'])
    fig,axs=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(hspace=.45)
    ridx={r['id']:i for i,r in enumerate(paired)}
    for ax,sid in zip(axs.flat,chosen):
        i=ridx[sid];r=paired[i];im=np.asarray(Image.open(runtime[i]['original_rgb']).convert('RGB'))
        ax.imshow(im);bx,by,bw,bhh=r['bbox_xywh'];ax.add_patch(Rectangle((bx,by),bw,bhh,fill=False,edgecolor='lime',lw=1.5))
        px,py=r['pcrau_point'];ax.scatter([px],[py],marker='x',c='red',s=90,label='P-CRA-U')
        pts=np.asarray(r['backbone_matched_points'])
        if len(pts):ax.scatter(pts[:,0],pts[:,1],marker='+',c='cyan',s=90,label='RoboRefer RGB-D')
        ax.set_title(textwrap.fill(f"sent {sid}: {r['phrase']}",48)+f"\nP={int(r['pcrau_hit'])}; R={int(r['backbone_matched_hit'])}; risk={r['risk']:.3f}; {r['action']}",fontsize=9);ax.axis('off');ax.legend(fontsize=7)
    fig.savefig(OUT/'cases.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for name,keyname in [('Tasks60','risk'),('Live44','live44_risk')]:
        idx=np.argsort([r[keyname] for r in paired],kind='stable');ax.plot(np.arange(1,301)/300,np.cumsum(err[idx])/np.arange(1,301),label=name)
    ax.set(xlabel='Coverage',ylabel='Point-outside-box risk',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=160);plt.close(fig)
    pc=summary['P_CRA_U_Tasks60'];sr='không xác định' if pc['selective_grounding_risk'] is None else f"{100*pc['selective_grounding_risk']:.2f}% ({pc['errors']}/{pc['accepted']})"
    text=f'''# RefCOCO UNC val — pilot 300 câu\n\nNgày 07/10/2026. **Subset pilot, không phải toàn bộ val hoặc submission leaderboard.**\n\n## Dữ liệu và protocol\n\n- Nguồn mirror `{REPO}`, revision `{REV}`; parquet SHA256 `{SOURCE_SHA}`, đã đối chiếu LFS.\n- Full UNC val: 1.500 ảnh, 3.811 references, 10.834 câu. Pilot 100 ảnh × 3 câu = 300 câu, {protocol['references']} references.\n- Sampling khóa trước forward: rank SHA256(seed:image:id) chọn 100 ảnh trong {protocol['eligible_images']} ảnh có ≥3 câu; rank SHA256(seed:sentence:id) chọn 3 câu mỗi ảnh. {protocol['excluded_images_with_less_than_3_sentences']} ảnh có <3 câu không đủ điều kiện pilot. Không lọc theo model score. Đây là sampling cân bằng ảnh, không random uniform 300 câu trong toàn val.\n- Giữ nguyên nội dung raw phrase; thêm lệnh Locate và mạo từ the khi cần. Không rút gọn quan hệ/dùng class GT để viết prompt.\n- P-CRA-U dùng RGB và pseudo-depth 640×480. RoboRefer có đối chứng RGB-D cùng đầu vào và RGB-only ở ảnh gốc. Greedy max_new_tokens=128.\n- MAP được chuyển về hệ tọa độ ảnh gốc, chấm point-in-box từ bbox xywh GT. Bbox/nhãn chỉ vào evaluator sau khi lưu predictions; không IoU, không PIT theo mask.\n- Frozen Tasks60 bundle, threshold {summary['threshold']}; T=20; seed {SEED}. Không train/fit/chọn ngưỡng bằng pilot.\n\n## Kết quả\n\n| Pipeline | Point-in-box | Coverage | Selective grounding risk |\n|---|---:|---:|---:|\n| P-CRA-U Tasks60 MC | {int(hit.sum())}/300 ({100*hit.mean():.2f}%) | {100*pc['coverage']:.2f}% | {sr} |\n| RoboRefer RGB-D, matched640 | {int(bh.sum())}/300 ({100*bh.mean():.2f}%) | — | — |\n| RoboRefer RGB-only, ảnh gốc | {int(nh.sum())}/300 ({100*nh.mean():.2f}%) | — | — |\n\nRisk metrics (event MAP ngoài bbox, khác event composite đã fit):\n\n```json\n{json.dumps(tasks,indent=2)}\n```\n\nScope: `{summary['scope']}`. Semantic phrase matching missing: {summary['semantic_phrase_missing']}/300, không đồng nghĩa semantic classification accuracy=0%.\n\nPaired so matched640: fixed {len(fixed)}, broken {len(broken)}; delta {summary['paired']['delta_matched_pp']:.2f} pp; CI theo ảnh {summary['paired']['delta_matched95_pp']} pp. PIT P-CRA-U CI {summary['paired']['pcrau_PIT95_percent']}%.\n\n## Case thật\n\n![Cases](../new/outputs/refcoco_val_pilot300_20261007/cases.png)\n\nXanh lá: bbox GT hậu kiểm; đỏ: P-CRA-U; cyan: RoboRefer RGB-D matched640.\n\n![Risk coverage](../new/outputs/refcoco_val_pilot300_20261007/risk_coverage.png)\n\n## Diễn giải và giới hạn\n\n- Pilot đo grounding 2D trên ảnh thực. Verifier chỉ hoạt động nếu parser hỗ trợ; không coi unsupported/direct là đã kiểm chứng quan hệ.\n- Box bao gồm cả nền/vật khác nên point-in-box có thể đạt khi điểm ngoài mask của target. Không so trực tiếp với 6/100 point-in-mask trên RefSpatial.\n- Ngay cả nếu point-in-box cao, chưa xác nhận depth theo mét, anchor binding, amodal completion hoặc cả năm nhóm uncertainty.\n- Không có nhãn answerability bốn trạng thái để chấm accuracy/macro-F1 hoặc event composite gốc. Risk metrics là kiểm tra transfer sang event grounding box.\n- Đây không phải tái hiện chính xác điểm bảng paper: subset cân bằng ảnh, prompt/decoding/RGB-D được ghi rõ. Không lấy số của 300 câu như điểm toàn val; không dùng benchmark đã xem để tune rồi gọi test độc lập.\n- Checkpoint, calibrator, threshold và active profile giữ nguyên; không phát lệnh robot.\n\n## Artifacts\n\n`new/outputs/refcoco_val_pilot300_20261007/`: protocol, runtime/evaluator manifests tách biệt, predictions, paired_evaluator.json, summary.json, figures, logs.\n\nNguồn: https://github.com/lichengunc/refer ; https://huggingface.co/datasets/jxu124/refcoco-benchmark ; https://arxiv.org/html/2506.04308v2#S4.SS3 .\n'''
    (ROOT/'plan/REFCOCO_VAL_PILOT300_20261007.md').write_text(text)
    for p,h in protocol['protected_hashes'].items():assert sha(p)==h,p
    write(OUT/'FINAL_STATUS.json',{'status':summary['status'],'samples':300,'protected_unchanged':True,'point_membership_crosscheck':True,'robot_motion_commanded':False})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer','report']);a=p.parse_args();{'prepare':prepare,'infer':infer,'report':report}[a.stage]()
