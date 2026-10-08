#!/usr/bin/env python3
"""Read-only five-source MC results, real examples and uncertainty plots."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcrau.dataset import SOURCE_CLASSES
from pcrau.utils import read_json,sha256_file,atomic_json


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def error_auc(values,error):
    positive=values[error];negative=values[~error]
    if not len(positive) or not len(negative):return None
    delta=positive[:,None]-negative[None,:]
    return float(((delta>0)+.5*(delta==0)).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('new/outputs/pcrau_source_mc_train_dev_20261007'))
    a=p.parse_args();root=a.root.resolve();report=root/'report'
    protected={str(p):sha256_file(p) for p in root.rglob('*') if p.is_file()};report.mkdir(exist_ok=False)
    summary=read_json(root/'summary.json');ev=rows(root/'dev/evaluator_sources.jsonl');analysis={}
    with (report/'dev_per_source.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source','positive_labels','det_F1','MC_F1','det_Brier','MC_Brier',
            'mean_MI','max_MI','MI_error_AUROC','entropy_error_AUROC','weak_label'])
        writer.writeheader()
        for k,name in enumerate(SOURCE_CLASSES):
            m=summary['splits']['dev']['per_source'][name]
            mean=np.asarray([r['sources'][name]['source_score_mc_mean'] for r in ev])
            mi=np.asarray([r['sources'][name]['mutual_information_normalized'] for r in ev])
            pe=np.asarray([r['sources'][name]['predictive_entropy_normalized'] for r in ev])
            y=np.asarray([r['evaluation']['source_multihot'][k] for r in ev])
            error=(mean>=.5)!=y
            analysis[name]={'MI_error_AUROC':error_auc(mi,error),'entropy_error_AUROC':error_auc(pe,error),
                            'errors':int(error.sum()),'is_dev_descriptive':True}
            writer.writerow({'source':name,'positive_labels':m['MC_mean']['positive_labels'],
                'det_F1':m['deterministic']['f1'],'MC_F1':m['MC_mean']['f1'],
                'det_Brier':m['deterministic']['brier'],'MC_Brier':m['MC_mean']['brier'],
                'mean_MI':m['MI_mean'],'max_MI':m['MI_max'],
                'MI_error_AUROC':analysis[name]['MI_error_AUROC'],
                'entropy_error_AUROC':analysis[name]['entropy_error_AUROC'],'weak_label':name=='spatial'})
    selected=[next(r for r in ev if r['sample_id']==sid) for sid in
        ('v211dev_family_000001__clean','v211dev_family_000001__occlusion_view_counterfactual')]
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for col,r in enumerate(selected):
        vals=[r['sources'][k]['source_score_mc_mean'] for k in SOURCE_CLASSES]
        std=[r['sources'][k]['source_score_mc_std'] for k in SOURCE_CLASSES]
        mi=[r['sources'][k]['mutual_information_normalized'] for k in SOURCE_CLASSES]
        axes[0,col].bar(SOURCE_CLASSES,vals,yerr=std,capsize=3,color='#2563eb')
        axes[0,col].set_ylim(0,1.05);axes[0,col].set_ylabel('MC mean source score ± std')
        axes[0,col].set_title(r['sample_id'].removeprefix('v211dev_family_'),fontsize=11)
        axes[1,col].bar(SOURCE_CLASSES,mi,color='#d97706')
        axes[1,col].set_ylabel('Normalized MI: model disagreement')
        axes[1,col].set_ylim(0,max(.006,max(mi)*1.3))
        for ax in axes[:,col]:ax.tick_params(axis='x',labelrotation=20);ax.grid(axis='y',alpha=.15)
        atomic_json(report/f'case_{col+1:02d}.json',r)
    fig.suptitle('20 source-head MC dropout draws: source score and uncertainty are different quantities',fontsize=12)
    fig.savefig(report/'source_scores_and_MI.png',dpi=160);fig.savefig(report/'source_scores_and_MI.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(16,3.6),constrained_layout=True)
    for k,name in enumerate(SOURCE_CLASSES):
        mean=np.asarray([r['sources'][name]['source_score_mc_mean'] for r in ev])
        mi=np.asarray([r['sources'][name]['mutual_information_normalized'] for r in ev])
        y=np.asarray([r['evaluation']['source_multihot'][k] for r in ev]);error=(mean>=.5)!=y
        ax=axes[k];ax.scatter(mean[~error],mi[~error],s=10,alpha=.5,color='#2563eb',label='correct')
        ax.scatter(mean[error],mi[error],s=15,alpha=.8,color='#dc2626',label='wrong')
        ax.set_title(name+(' (weak)' if name=='spatial' else ''));ax.set_xlabel('MC mean source score');ax.grid(alpha=.15)
        if k==0:ax.set_ylabel('Normalized MI');ax.legend(fontsize=8)
    fig.suptitle('Dev: MC disagreement can remain low on wrong source predictions',fontsize=13)
    fig.savefig(report/'dev_uncertainty_errors.png',dpi=160);fig.savefig(report/'dev_uncertainty_errors.pdf');plt.close(fig)
    atomic_json(report/'source_error_detection.json',analysis)
    table=['| Nguồn | MC mean | Std | Predictive entropy | MI | Nhãn hậu kiểm |',
           '|---|---:|---:|---:|---:|---|']
    r=selected[1]
    for k,name in enumerate(SOURCE_CLASSES):
        x=r['sources'][name]
        table.append(f"| {name} | {x['source_score_mc_mean']:.6f} | {x['source_score_mc_std']:.6f} | {x['predictive_entropy_normalized']:.6f} | {x['mutual_information_normalized']:.6f} | {int(r['evaluation']['source_multihot'][k])} |")
    (report/'REAL_CASE.md').write_text('# Kết quả MC thật\n\nSample `'+r['sample_id']+'`, T=20, source-head-only, p=0,1.\n\n'+'\n'.join(table)+'\n\nNhãn chỉ hậu kiểm; spatial là AMBIGUOUS proxy. Source score chưa hiệu chuẩn theo nguồn.\n')
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    atomic_json(report/'provenance.json',{'original_files_sha256':protected,
        'original_artifacts_unchanged':True,'report_script_sha256':sha256_file(Path(__file__)),
        'actual_case_ids':[r['sample_id'] for r in selected],'no_fit_or_source_selection':True})
    print(json.dumps({'report':str(report),'error_detection':analysis},ensure_ascii=False))


if __name__=='__main__':main()
