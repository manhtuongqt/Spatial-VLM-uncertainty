#!/usr/bin/env python3
"""Image-balanced RefCOCO+ UNC val pilot; frozen models, point-in-box scoring.

Use .conda-roborefer/bin/python3.10 -B (calibrated user-site torch enabled).
Worker uses the same interpreter with -s/PYTHONNOUSERSITE=1.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,io
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'new/outputs/refcoco_plus_val_pilot300_20261007'
SOURCE=OUT/'source/val.parquet'
REPO='jxu124/refcoco-benchmark'
REV='2f2f892835bbf44b4db81f1c4ad6d46a3e9e4359'
SOURCE_SHA='c7949870f569f09cba78286fd29d979e8e29fe5af1e69b30d1a709d88125738b'
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
    assert len(dataset)==1500 and nref==3805 and nquery==10758
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
    protocol={'status':'LOCKED_BEFORE_INFERENCE','dataset':'RefCOCO+ UNC val','pilot':True,
        'source_repo':REPO,'source_revision':REV,'source_path':str(SOURCE),'source_sha256':SOURCE_SHA,
        'full_val':{'images':1500,'references':3805,'expressions':10758},
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
        'primary_adapter_sha256':sha(ROOT/'new/src/pcrau/roborefer_primary.py'),
        'primary_risk_status':'PENDING_CALIBRATION',
        'worker_sha256':sha(ROOT/'new/scripts/refcoco_plus_pilot_worker.py'),
        'runtime_manifest_sha256':sha(OUT/'runtime_manifest.json'),'evaluator_manifest_sha256':sha(OUT/'evaluator_manifest.json'),
        'image_inventory':imgmeta}
    
    for directory in ('refcoco_val_pilot300_20261007','pcrau_roborefer_primary_pilot300_20261007'):
        for name in ('protocol.json','predictions.jsonl','summary.json','FINAL_STATUS.json'):
            source=ROOT/'new/outputs'/directory/name
            protocol['protected_hashes'][str(source)]=sha(source)
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
    from pcrau.roborefer_primary import compose_prediction, generation_prompt
    assert sha(ROOT/'new/src/pcrau/roborefer_primary.py')==protocol['primary_adapter_sha256']
    saved=rows(OUT/'predictions.jsonl');back=rows(OUT/'backbone_predictions.jsonl')
    assert len(saved)==len(back)==len(entries)==300
    with (OUT/'primary_predictions.jsonl').open('w') as f:
        for e,s,b in zip(entries,saved,back):
            assert e['id']==s['id']==b['id']
            assert e['backbone_prompt']==generation_prompt(e['prompt'])
            assert sha(e['feature_path'])==b['feature_sha256']
            r=compose_prediction(e['prompt'],b['answer'],image_size=e['original_size'],sidecar_observation=s)
            f.write(json.dumps({'id':e['id'],'image_id':e['image_id'],**r},ensure_ascii=False)+'\n')
    write(OUT/'inference_complete.json',{'samples':len(entries),'no_training':True,'no_calibration_fit':True,
        'mode':'NEW_GENERATIONS_AND_TASKS60_FORWARD_PLUS_PRIMARY_ADAPTER',
        'primary_predictions_sha256':sha(OUT/'primary_predictions.jsonl')})


def point_inside(point,box,size):
    if point is None:return False
    x,y=point;bx,by,bw,bh=box;w,h=size
    return bool(0<=x<w and 0<=y<h and bx<=x<=bx+bw and by<=y<=by+bh)


def report():
    import numpy as np
    from PIL import Image
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import textwrap
    sys.path.insert(0,str(ROOT/'RoboRefer/Evaluation'))
    from summarize_acc import text2pts
    protocol=json.loads((OUT/'protocol.json').read_text())
    complete=json.loads((OUT/'inference_complete.json').read_text())
    assert sha(__file__)==protocol['script_sha256']
    assert sha(OUT/'evaluator_manifest.json')==protocol['evaluator_manifest_sha256']
    assert sha(OUT/'primary_predictions.jsonl')==complete['primary_predictions_sha256']
    evaluation=json.loads((OUT/'evaluator_manifest.json').read_text())
    runtime=json.loads((OUT/'runtime_manifest.json').read_text())
    primary,sidecar,back=[rows(OUT/name) for name in ('primary_predictions.jsonl','predictions.jsonl','backbone_predictions.jsonl')]
    assert len(evaluation)==len(primary)==len(sidecar)==len(back)==300
    paired=[];decoder_matches=0;counts=Counter()
    for e,r,s,b in zip(evaluation,primary,sidecar,back):
        assert e['id']==r['id']==s['id']==b['id']
        w,h=e['image_size'];point=r['target']['pixel_xy'];counts[r['target']['status']]+=1
        official=text2pts(b['answer'],w,h,False).tolist()
        equal=(official==[point]) if point is not None else (len(official)!=1 or not point_inside(official[0],[0,0,w-1,h-1],[w,h]))
        decoder_matches+=int(equal)
        sx,sy=s['spatial']['map_pixel_xy'];sp=[int(sx/640*w),int(sy/480*h)]
        native=text2pts(b['native_rgb_answer'],w,h,False).tolist()
        paired.append({**e,'primary_point':point,'primary_hit':point_inside(point,e['bbox_xywh'],[w,h]),
            'sidecar_point':sp,'sidecar_hit':point_inside(sp,e['bbox_xywh'],[w,h]),
            'native_rgb_points':native,'native_rgb_hit':len(native)==1 and point_inside(native[0],e['bbox_xywh'],[w,h]),
            'primary_scope':r['geometry']['scope_status'],'primary_risk':r['decision']['risk'],
            'primary_action':r['decision']['action'],
            'sidecar_risk_diagnostic':s['decision']['risk'],'sidecar_action_diagnostic':s['decision']['action']})
    p=np.array([r['primary_hit'] for r in paired],dtype=float)
    s=np.array([r['sidecar_hit'] for r in paired],dtype=float)
    native=np.array([r['native_rgb_hit'] for r in paired],dtype=float)
    images=sorted({r['image_id'] for r in paired});assert len(images)==100
    groups=np.array([[i for i,r in enumerate(paired) if r['image_id']==iid] for iid in images]);assert groups.shape==(100,3)
    rng=np.random.default_rng(SEED);indices=groups[rng.integers(0,100,(10000,100))].reshape(10000,300)
    def metric(hits):
        return {'hits':int(hits.sum()),'samples':300,'point_in_box':float(hits.mean()),
            'CI95_percent':(100*np.quantile(hits[indices].mean(1),[.025,.975])).tolist()}
    fixed=[r['id'] for r in paired if r['primary_hit'] and not r['sidecar_hit']]
    broken=[r['id'] for r in paired if not r['primary_hit'] and r['sidecar_hit']]
    old=json.loads((ROOT/'new/outputs/refcoco_val_pilot300_20261007/runtime_manifest.json').read_text())
    oldimages={r['image_id'] for r in old};common=set(images)&oldimages
    summary={'status':'COMPLETE_REFCOCO_PLUS_UNC_VAL_PILOT_100_IMAGES_NOT_FULL_VAL',
        'samples':300,'images':100,'references':protocol['references'],'full_val':protocol['full_val'],
        'primary_RoboRefer_RGBD':metric(p),'Sidecar_Tasks60':metric(s),'RoboRefer_RGB_native':metric(native),
        'paired_primary_vs_sidecar':{'fixed_ids':fixed,'broken_ids':broken,'delta_pp':float(100*(p-s).mean()),
            'delta95_pp':(100*np.quantile((p-s)[indices].mean(1),[.025,.975])).tolist()},
        'paired_primary_vs_native':{'delta_pp':float(100*(p-native).mean()),
            'delta95_pp':(100*np.quantile((p-native)[indices].mean(1),[.025,.975])).tolist()},
        'primary_point_status':dict(counts),'official_decoder_matches':decoder_matches,
        'primary_scope':dict(Counter(r['primary_scope'] for r in paired)),
        'semantic_phrase_missing_diagnostic':sum(x['task_uncertainty']['risk_evidence']['semantic_matching_missing'] for x in sidecar),
        'primary_calibrated_risk':None,'primary_selective_coverage':None,
        'old_RefCOCO_pilot_common_image_ids':sorted(common),'common_images_count':len(common),
        'GT_input':False,'train_steps':0,'fit_calls':0,'threshold_tuned_on_pilot':False,
        'RGBD_preprocess':'DepthAnythingV2 Large original RGB then minmax uint8; RGB/depth bilinear640x480',
        'new_generations':300,'new_native_RGB_control_generations':300,
        'exact_paper_table_reproduction':False,'robot_motion_commanded':False}
    write(OUT/'paired_evaluator.json',paired);write(OUT/'summary.json',summary)
    failures=[r for r in paired if not r['primary_hit']]
    write(OUT/'primary_errors.json',failures)
    chosen=list(dict.fromkeys([r['id'] for r in failures[:3]]+fixed[:2]+broken[:1]))
    for r in paired:
        if len(chosen)>=6:break
        if r['id'] not in chosen:chosen.append(r['id'])
    lookup={r['id']:i for i,r in enumerate(paired)}
    fig,axes=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(hspace=.4)
    for ax,sid in zip(axes.flat,chosen):
        i=lookup[sid];r=paired[i];ax.imshow(Image.open(runtime[i]['original_rgb']).convert('RGB'))
        bx,by,bw,bh=r['bbox_xywh'];ax.add_patch(Rectangle((bx,by),bw,bh,fill=False,edgecolor='lime',lw=2))
        ax.scatter(*r['sidecar_point'],marker='x',c='red',s=90,label='Sidecar')
        if r['primary_point'] is not None:ax.scatter(*r['primary_point'],marker='+',c='cyan',s=100,label='RoboRefer primary')
        ax.set_title(textwrap.fill(f"{sid}: {r['phrase']}",48)+
            f"\nprimary={int(r['primary_hit'])}; Sidecar={int(r['sidecar_hit'])}; risk=pending",fontsize=9)
        ax.axis('off');ax.legend(fontsize=7)
    fig.savefig(OUT/'cases.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Preserve all predictions, bundles, original pilots, and source scripts.
    for path,digest in protocol['protected_hashes'].items():assert sha(path)==digest,path
    assert sha(OUT/'runtime_manifest.json')==protocol['runtime_manifest_sha256']
    assert sha(ROOT/'new/src/pcrau/roborefer_primary.py')==protocol['primary_adapter_sha256']
    write(OUT/'FINAL_STATUS.json',{'status':summary['status'],'protected_unchanged':True,
        'primary_predictions_sha256':sha(OUT/'primary_predictions.jsonl'),
        'evaluator_sha256':sha(OUT/'paired_evaluator.json'),'robot_motion_commanded':False})
    def result(label,m):return f"| {label} | {m['hits']}/300 | {100*m['point_in_box']:.2f}% | [{m['CI95_percent'][0]:.2f}; {m['CI95_percent'][1]:.2f}]% |"
    table='\n'.join([result('RoboRefer RGB-D primary',summary['primary_RoboRefer_RGBD']),
        result('Sidecar Tasks60, MAP riêng',summary['Sidecar_Tasks60']),
        result('RoboRefer RGB-only, ảnh gốc',summary['RoboRefer_RGB_native'])])
    oldsummary=json.loads((ROOT/'new/outputs/refcoco_val_pilot300_20261007/summary.json').read_text())
    txt=f'''# RefCOCO+ UNC val — 100 ảnh / 300 câu

Ngày 07/10/2026. Pilot theo ảnh, không toàn bộ val.

## Protocol

- Source `jxu124/refcoco-benchmark`, revision `{REV}`, file `data/refcoco_plus_unc_val-00000-of-00001-2c68b2dca7cc4d2b.parquet`, SHA256 `{SOURCE_SHA}` đã đối chiếu LFS.
- Full val {protocol['full_val']['images']} ảnh, {protocol['full_val']['references']} references, {protocol['full_val']['expressions']} câu; chỉ chạy100ảnh/300câu/{protocol['references']}references.
- Sampling khóa trước forward: seed{SEED}, SHA256 rank100ảnh trong {protocol['eligible_images']}ảnh có≥3câu; rank3câu/ảnh. Loại {protocol['excluded_images_with_less_than_3_sentences']}ảnh thiếu3câu theo eligibility, không theo model score.
- Giữ raw phrase, chỉ thêm Locate/mạo từ và dấu chấm như pilot RefCOCO. Không dùng category/bbox để viết câu hoặc chọn prediction.
- Primary frozen RoboRefer-2B-SFT RGB-D640×480, DepthAnythingV2 Large infer từ RGB gốc, greedy max_new_tokens128. Native RGB-only là đối chứng riêng. Không so hai nhánh rồi chọn theo nhãn.
- Sidecar Tasks60 chạy forward + MC20 trên cùng features/câu; primary adapter giữ điểm RoboRefer. Primary risk/threshold null; decision REVIEW_UNCALIBRATED khi valid point.
- Metric point-in-GT-bbox xywh trong ảnh gốc; không IoU hoặc point-in-mask. GT chỉ đọc trong report sau lưu predictions. CI bootstrap10000 theo100image groups.
- {len(common)}/100ảnh trùng imageID với pilot RefCOCO cũ; câu/annotation là RefCOCO+. Không xem hai pilot là hai bộ cảnh độc lập hoặc cộng mẫu như600cảnh.
- Không train/fit/tune threshold; không đổi active profile/bundle cũ hoặc robot motion.

## Kết quả

| Pipeline / nguồn target | Đúng | Point-in-box | 95% CI theo ảnh |
|---|---:|---:|---:|
{table}

Primary so Sidecar: sửa{len(fixed)}, làm sai{len(broken)}; delta {summary['paired_primary_vs_sidecar']['delta_pp']:+.2f}pp, CI {summary['paired_primary_vs_sidecar']['delta95_pp']}pp.
Official decoder crosscheck {decoder_matches}/300. Status primary `{dict(counts)}`.
Scope geometry `{summary['primary_scope']}`. Semantic matching missing diagnostic {summary['semantic_phrase_missing_diagnostic']}/300, không là accuracy semantic.

## Case thật và lỗi

![Case thật](../new/outputs/refcoco_plus_val_pilot300_20261007/cases.png)

Xanh lá: GT bbox chỉ dùng hậu kiểm; cyan: primary; đỏ: Sidecar. Hình gồm các lỗi primary và ca được sửa/làm sai khi đổi nguồn target.
Lưu toàn bộ {len(failures)} lỗi primary trong `primary_errors.json`; không chỉ chọn ảnh đẹp.

## Đánh giá và giới hạn

- Điểm primary là năng lực grounding kế thừa từ RoboRefer. Sidecar không thay/veto điểm này; chưa chứng minh uncertainty cải thiện grounding.
- Pilot RefCOCO cũ primary {oldsummary['RoboRefer_RGBD_matched640']['hits']}/300 ({100*oldsummary['RoboRefer_RGBD_matched640']['point_in_box']:.2f}%). RefCOCO+ mới {int(p.sum())}/300 ({100*p.mean():.2f}%). Hai kết quả khác câu/targets; không diễn giải chênh lệch như paired model improvement.
- Risk/coverage primary chưa được hiệu chuẩn. Không lấy risk Sidecar đánh giá như risk của RoboRefer, không báo answerability accuracy vì benchmark không có nhãn4states.
- Scope verifier hạn chế trái/phải với cú pháp đã hỗ trợ; không gọi mô tả unsupported là đã kiểm chứng. Benchmark này chưa xác nhận năm task uncertainty, depth mét, occlusion amodal hoặc an toàn robot.
- Không phải tái hiện bảng paper theo full split hoặc leaderboard. Model/prompt/RGB-D preprocessing được ghi rõ; pilot không dùng để train/fit/tune.

## Artifacts

`new/outputs/refcoco_plus_val_pilot300_20261007/`: source, protocol, runtime/evaluator manifests, inputs/features, backbone_predictions, predictions Sidecar, primary_predictions, paired_evaluator, summary, primary_errors, cases và final status.
Runner `new/scripts/evaluate_refcoco_plus_pilot.py`; worker `new/scripts/refcoco_plus_pilot_worker.py`.

Nguồn: [REFER tác giả](https://github.com/lichengunc/refer), [mirror dataset](https://huggingface.co/datasets/jxu124/refcoco-benchmark), [RoboRefer point-in-box protocol](https://arxiv.org/html/2506.04308v2#S4.SS3).
'''
    (ROOT/'plan/REFCOCO_PLUS_VAL_PILOT300_20261007.md').write_text(txt)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','infer','report'])
    a=parser.parse_args();{'prepare':prepare,'infer':infer,'report':report}[a.stage]()
