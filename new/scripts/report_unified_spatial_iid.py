#!/usr/bin/env python3
"""Read-only reports and real IID cases from a frozen live reevaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.utils import read_json,sha256_file,atomic_json


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('new/outputs/pcrau_unified_spatial_iid_20261007'))
    args=parser.parse_args();root=args.root.resolve()
    inputs={str(p):sha256_file(p) for p in root.rglob('*') if p.is_file()}
    report=root/'report';report.mkdir(exist_ok=False)
    summary=read_json(root/'summary.json')
    runtime={r['sample_id']:r for r in rows(root/'runtime_predictions.jsonl')}
    evaluated={r['sample_id']:r for r in rows(root/'evaluator_predictions.jsonl')}
    baseline={r['sample_id']:r for r in rows(root/'baseline_evaluator_predictions.jsonl')}
    anchors={r['sample_id']:r for r in rows(root/'anchor_evaluation.jsonl')}
    changes=rows(root/'paired_decision_changes.jsonl')
    with (report/'comparison.csv').open('w',newline='') as f:
        fields=['model','threshold','accepted','correct','errors','coverage','selective_risk',
                'valid_recall','answerability_accuracy','macro_f1','brier','nll','ece10','aurc']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,metrics in summary['metrics'].items():
            p=metrics['policy'];prob=metrics['probability'];a=metrics['answerability']
            writer.writerow(dict(model=name,threshold=summary[name+'_threshold'],accepted=p['accepted'],
                correct=p['correct_acceptances'],errors=p['errors'],coverage=p['coverage'],
                selective_risk=p['empirical_risk'],valid_recall=p['correct_acceptances']/p['correct_found_total'],
                answerability_accuracy=a['accuracy'],macro_f1=a['macro_f1'],brier=prob['brier'],
                nll=prob['nll'],ece10=prob['ece_10'],aurc=prob['aurc']))
    # Display-only risk–coverage; frozen thresholds are never reselected.
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for name,rr,color in [('baseline',list(baseline.values()),'#2563eb'),('live',list(evaluated.values()),'#d97706')]:
        risk=np.asarray([r['decision']['risk'] for r in rr])
        errors=np.asarray([r['evaluation']['error_event'] for r in rr])
        eligible=np.asarray([r['decision']['predicted_answerability']=='FOUND' for r in rr])
        indices=np.flatnonzero(eligible);order=indices[np.argsort(risk[indices],kind='stable')]
        count=np.arange(1,len(order)+1)
        ax.plot(count/len(rr)*100,np.cumsum(errors[order])/count*100,label=name,color=color)
        p=summary['metrics'][name]['policy'];ax.scatter([p['coverage']*100],[p['empirical_risk']*100],s=65,color=color,zorder=5)
    ax.axhline(7.5,color='black',ls='--',lw=1,label='Calibration budget 7.5%')
    ax.set(xlabel='Coverage (%)',ylabel='Empirical selective risk (%)',
           title='Old IID reevaluation: current live evidence; dots = frozen profiles')
    ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(report/'risk_coverage.png',dpi=160);fig.savefig(report/'risk_coverage.pdf');plt.close(fig)
    lock=read_json(root/'frozen/original_freeze_lock.json');cfg=read_json(Path(lock['config']['path']))
    ds=ArchivedPCRAUDataset(cfg,'test_iid',profile='test_iid');entries={e['sample_id']:e for e in ds.entries}
    labels={'new_correct_accept':'Thêm lượt nhận đúng','removed_correct_accept':'Mất lượt nhận đúng',
            'new_error_accept':'Thêm lượt nhận sai','removed_error_accept':'Loại lượt nhận sai',
            'same_acceptance':'Đổi hành động từ chối'}
    selected=[(labels[r['acceptance_category']],r['sample_id']) for r in changes]
    def panel(selected,name):
        cols=min(4,len(selected));height=math.ceil(len(selected)/cols)
        fig,axes=plt.subplots(height,cols,figsize=(4.3*cols,5.3*height),squeeze=False)
        table=['| Sample | Nhóm | Query | Margin px / compatibility | Baseline → live | Truth / anchor hậu kiểm |',
               '|---|---|---|---|---|---|']
        for i,(label,sid) in enumerate(selected):
            r=runtime[sid];ev=evaluated[sid]['evaluation'];b=baseline[sid];v=r['verifiers']['P1'];q=v['query']
            e=entries[sid];rgbpath=ds.layout.dataset_path(e['feature_input']['rgb_path'])
            assert sha256_file(rgbpath)==e['feature_input']['rgb_sha256']
            rgb=cv2.imread(str(rgbpath));assert rgb is not None and rgb.shape[:2]==(480,640)
            ax=axes.flat[i];ax.imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
            tx,ty=v['target_peak_xy'];ax.scatter([tx],[ty],c='#00cfff',marker='x',s=95,lw=2)
            if v['anchor_peak_xy'] is not None:
                xx,yy=v['anchor_peak_xy'];ax.scatter([xx],[yy],facecolors='none',edgecolors='#ffb100',s=100,lw=2)
            short=sid.removeprefix('v211iid_family_');ax.set_title(f'{label}\n{short}',loc='left',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
            t=q['target']['text'] if q['target'] else 'unparsed'
            a=q['anchors'][0]['text'] if q['anchors'] else '—';relation=q['predicate'] or 'unsupported'
            text=f'{t} / {relation} / {a}\n'
            if v['signed_margin_px'] is not None:
                text+=f"margin={v['signed_margin_px']} px; compatibility={v['pair_compatibility']:.3f}\n"
            else:text+=v['scope_status']+'\n'
            text+=f"B: {b['decision']['risk']:.4f} {b['decision']['action']}\nL: {r['decision']['risk']:.4f} {r['decision']['action']}"
            ax.set_xlabel(text,fontsize=8)
            hindsight=ev['answerability_state']
            if sid in anchors:hindsight+=f"; anchor={anchors[sid]['P1_hit']}; pixels={anchors[sid]['pixels']}"
            compat='—' if v['pair_compatibility'] is None else f"{v['pair_compatibility']:.6f}"
            table.append(f"| `{sid}` | {label} | {t} → {relation} → {a} | {v['signed_margin_px']} / {compat} | {b['decision']['action']} → {r['decision']['action']} | {hindsight} |")
            atomic_json(report/f'{name}_case_{i+1:02d}.json',{'sample_id':sid,'label':label,
                'rgb_path':str(rgbpath),'rgb_sha256':sha256_file(rgbpath),'runtime_prediction':r,
                'baseline_decision':b['decision'],'evaluator_only':ev,
                'anchor_evaluator_only':anchors.get(sid)})
        for ax in axes.flat[len(selected):]:ax.set_visible(False)
        fig.suptitle('P-CRA-U: frozen live bundle — IID cũ',x=.02,ha='left',fontsize=15)
        fig.text(.02,.02,'Cyan ×: target MAP · orange circle: anchor MAP · binding/presence UNVERIFIED · B=baseline, L=live',fontsize=9)
        fig.subplots_adjust(left=.025,right=.985,top=.88,bottom=.17,wspace=.07,hspace=.70)
        fig.savefig(report/f'{name}.png',dpi=160,facecolor='white')
        fig.savefig(report/f'{name}.pdf',facecolor='white');plt.close(fig)
        (report/f'{name}_TABLE.md').write_text('# Case thật — IID đã quan sát\n\nNhóm và masks chỉ dùng hậu kiểm; không chọn threshold/model bằng những case này.\n\n'+'\n'.join(table)+'\n')
    if selected:panel(selected,'changed_decisions')
    # Honest scope/binding limitations from saved predictions, not tuned cases.
    supported=[r for r in evaluated.values() if r['verifiers']['P1']['scope_status']=='SUPPORTED_DIAGNOSTIC']
    candidates=[('Anchor sai, geometry tương thích',lambda r:anchors[r['sample_id']]['P1_hit'] is False and
                    r['verifiers']['P1']['features']['peak_compatible']==1.),
        ('Geometry âm nhưng vẫn nhận',lambda r:r['decision']['action']=='EXECUTE' and
                    r['verifiers']['P1']['features']['peak_compatible']==0.),
        ('Anchor rỗng vẫn có peak',lambda r:anchors[r['sample_id']]['pixels']==0),
        ('Geometry đúng, thiếu evidence',lambda r:r['evaluation']['answerability_state']=='INSUFFICIENT_EVIDENCE' and
                    r['verifiers']['P1']['features']['peak_compatible']==1. and r['decision']['action']!='EXECUTE')]
    failures=[]
    for label,fn in candidates:
        match=next((r for r in supported if fn(r)),None)
        if match:failures.append((label,match['sample_id']))
    if failures:panel(failures,'binding_and_scope_limits')
    assert all(sha256_file(Path(p))==h for p,h in inputs.items())
    atomic_json(report/'provenance.json',{'original_run_files_sha256':inputs,
        'script_sha256':sha256_file(Path(__file__)),'changed_case_ids':[s for _,s in selected],
        'limitation_case_ids':[s for _,s in failures],'original_files_unchanged':True,
        'fit_model_or_threshold_selection':False,'images_from_existing_IID_RGB':True})
    print(json.dumps({'report':str(report),'changed_cases':len(selected),'limitation_cases':len(failures),
                      'inputs_preserved':len(inputs)},ensure_ascii=False))


if __name__=='__main__':main()
