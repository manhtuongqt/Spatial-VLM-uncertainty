#!/usr/bin/env python3
"""Read-only scientific reports for the actual task-MC bundle and real images."""
import json,csv,sys
from pathlib import Path
import numpy as np,cv2
from scipy.stats import rankdata,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcrau.task_inference import apply_task_calibrator
from pcrau.utils import atomic_json,sha256_file

ROOT=Path('new/outputs/pcrau_task_uncertainty_20261007');OUT=ROOT/'evaluation_r3'
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def auc(score,error):
    s=np.asarray(score);e=np.asarray(error,bool);n=e.sum();m=(~e).sum()
    return float((rankdata(s)[e].sum()-n*(n+1)/2)/(n*m)) if n and m else None

def main():
    assert (OUT/'RUN_STATUS.json').exists();report=OUT/'report_r2';report.mkdir(exist_ok=False)
    protected={str(p.resolve()):sha256_file(p) for p in OUT.rglob('*') if p.is_file() and report not in p.parents}
    cal=json.loads((OUT/'calibrator.json').read_text());profile=json.loads((OUT/'profile.json').read_text())
    observed=rows(ROOT/'evaluation_r2/dev/observable_predictions.jsonl')
    dev=apply_task_calibrator(observed,cal,profile);task_eval=rows(ROOT/'evaluation_r2/dev/task_evaluation.jsonl')
    lookup={r['sample_id']:r for r in dev};ev={r['sample_id']:r for r in task_eval}
    diagnostics={}
    for name,select,score,error in [
        ('semantic_identity',lambda e:True,lambda r:r['task_uncertainty']['semantic']['target']['MI'],lambda e:not e['semantic_identity_correct']),
        ('spatial_target',lambda e:e['target_exists'],lambda r:r['task_uncertainty']['spatial']['target']['MI'],lambda e:not e['target_hit']),
        ('relation',lambda e:'relation_MI' in e,lambda r:r['task_uncertainty']['relation']['MI'],lambda e:not e['relation_prediction_correct'])]:
        selected=[(lookup[e['sample_id']],e) for e in task_eval if select(e)]
        s=[score(r) for r,e in selected];err=[error(e) for r,e in selected]
        diagnostics[name]={'samples':len(s),'errors':sum(err),'MI_error_detection_AUROC':auc(s,err),
            'mean_MI_correct':float(np.mean([v for v,w in zip(s,err) if not w])) if any(not w for w in err) else None,
            'mean_MI_wrong':float(np.mean([v for v,w in zip(s,err) if w])) if any(err) else None}
    selected=[(lookup[e['sample_id']],e) for e in task_eval if e['depth_valid']]
    rho=spearmanr([r['task_uncertainty']['depth']['total_predictive_variance_m2'] for r,e in selected],
                  [abs(e['depth_error_m']) for r,e in selected])
    diagnostics['depth']={'variance_abs_error_Spearman':float(rho.statistic),'descriptive_p':float(rho.pvalue),
        'nominal_interval_coverage':.95,'actual_dev_coverage':float(np.mean([e['depth_interval_hit'] for r,e in selected])),
        'mean_interval_width_m':float(np.mean([r['task_uncertainty']['depth']['upper_95_m']-r['task_uncertainty']['depth']['lower_95_m'] for r,e in selected])),
        'family_independence_assumed_for_p':False,'p_not_used_as_confirmatory_claim':True}
    diagnostics['evaluation_status']='descriptive_dev_already_used_for_checkpoint_selection'
    atomic_json(report/'uncertainty_diagnostics.json',diagnostics)
    supplement=json.loads((ROOT/'data/supplement.json').read_text());preview=json.loads((ROOT/'data/preview_cases.json').read_text())
    chosen=[p for p in preview if p['split']=='dev'];cases=[]
    for p in chosen:
        r=lookup[p['sample_id']];e=ev[p['sample_id']];t=r['task_uncertainty']
        image=cv2.cvtColor(cv2.imread(p['occluded_rgb']['path']),cv2.COLOR_BGR2RGB)
        fig,ax=plt.subplots(2,3,figsize=(16,9))
        ax[0,0].imshow(image);x,y=r['spatial']['map_pixel_xy'];ax[0,0].scatter([x],[y],c='cyan',marker='x',s=100)
        anchor=r['verifiers']['P1']['anchor_peak_xy']
        if anchor:ax[0,0].scatter([anchor[0]],[anchor[1]],c='yellow',marker='x',s=100)
        ax[0,0].set_title('Real RGB: cyan target / yellow P1 anchor');ax[0,0].axis('off')
        prob=np.asarray(t['semantic']['target']['mean_distribution']);ids=np.argsort(prob)[-5:][::-1]
        ax[0,1].barh(np.arange(5),prob[ids]);ax[0,1].set_yticks(np.arange(5),[t['semantic']['classes'][i] for i in ids]);ax[0,1].invert_yaxis();ax[0,1].set_xlim(0,1)
        ax[0,1].set_title(f"Node semantic identity / MI={t['semantic']['target']['MI']:.5f}")
        rel=t['relation'];ax[0,2].axis('off')
        q=r['verifiers']['P1']['query']
        noun=q['target']['text'] if q['target'] else 'parser unsupported'
        anchorname=q['anchors'][0]['text'] if q['anchors'] else 'none / outside scope'
        compatibility=f"{rel['compatibility_mean']:.4f}" if rel['supported'] else 'not evaluated'
        relmi=f"{rel['MI']:.5f}" if rel['supported'] else 'not evaluated'
        ax[0,2].text(.02,.95,f"Parsed target: {noun}\nRelation: {q['predicate']}\nAnchor: {anchorname}\n\nCompatibility mean: {compatibility}\nRelation MI: {relmi}\nBinding/presence: UNVERIFIED\n\nRisk60: {r['decision']['risk']:.4f}\nAction: {r['decision']['action']}\nVariant: {e.get('variant','')} / evaluator only",va='top',fontsize=11)
        target=np.asarray(t['spatial']['target']['mean_distribution']);im=ax[1,0].imshow(target,cmap='magma',extent=(0,640,480,0));fig.colorbar(im,ax=ax[1,0],fraction=.04)
        ax[1,0].set_title(f"MC target map / MI={t['spatial']['target']['MI']:.5f}")
        comp=np.asarray(t['occlusion_completion']['mean_distribution']);im=ax[1,1].imshow(comp,cmap='magma',extent=(0,640,480,0));fig.colorbar(im,ax=ax[1,1],fraction=.04)
        lab=p['reference_label_id_supervision_only'];A=cv2.imread(p['reference_semantic_labels_supervision_only']['path'],cv2.IMREAD_UNCHANGED)==lab
        ax[1,1].contour(A,levels=[.5],colors=['lime'],linewidths=.7)
        ax[1,1].set_title(f"Reference completion / MI={t['occlusion_completion']['MI']:.5f}\nGreen contour: evaluator reference only")
        dep=t['depth'];ax[1,2].errorbar([0],[dep['mean_m']],yerr=[[dep['mean_m']-dep['lower_95_m']],[dep['upper_95_m']-dep['mean_m']]],fmt='o',capsize=6,label='Model mixture 95% interval')
        ax[1,2].axhline(e['depth_reference_m'],c='red',ls='--',label='Reference at predicted pixel')
        ax[1,2].set_xlim(-.5,.5);ax[1,2].set_xticks([]);ax[1,2].set_ylabel('Depth (m)');ax[1,2].legend(fontsize=8)
        ax[1,2].set_title(f"Depth mean={dep['mean_m']:.4f} m\nMC variance={dep['MC_mean_variance_m2']:.6f} m²")
        fig.suptitle(p['sample_id']+' — actual outputs, no hand-adjusted confidence',fontsize=13);fig.tight_layout(rect=(0,0,1,.96))
        path=report/(p['family_id']+'_dashboard.png');fig.savefig(path,dpi=140);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
        atomic_json(report/(p['family_id']+'_case.json'),{'prediction':r,'evaluator':e,'supplement':p});cases.append(path.name)
    changes=rows(OUT/'test_iid/changed_decisions.jsonl');atomic_json(report/'paired_changed_cases.json',changes)
    summary=json.loads((OUT/'summary.json').read_text());iid=summary['splits']['test_iid']
    fig,ax=plt.subplots(1,2,figsize=(11,4));labels=['live44','tasks60']
    for i,k in enumerate(('reference_live44','primary')):
        p=iid[k]['policy'];ax[0].bar(i,p['correct_acceptances'],label='Correct' if i==0 else None,color='seagreen');ax[0].bar(i,p['errors'],bottom=p['correct_acceptances'],label='Errors' if i==0 else None,color='tomato')
        ax[1].bar(i,iid[k]['probability']['brier'],color=['steelblue','darkorange'][i])
        ax[0].text(i,p['accepted']+3,f"{p['correct_acceptances']}+{p['errors']}\nrisk={p['empirical_risk']:.2%}",ha='center')
    ax[0].set_xticks([0,1],labels);ax[0].set_ylabel('Accepted observations');ax[0].set_ylim(0,450);ax[0].legend();ax[0].set_title('Old IID reevaluation / frozen thresholds')
    ax[1].set_xticks([0,1],labels);ax[1].set_ylabel('Brier score');ax[1].set_title('Composite perceptual error probability')
    fig.tight_layout();fig.savefig(report/'IID_comparison.png',dpi=150);fig.savefig(report/'IID_comparison.pdf');plt.close(fig)
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    atomic_json(report/'provenance.json',{'original_files_sha256':protected,'unchanged':True,
        'source_sha256':sha256_file(Path(__file__)),'actual_dashboard_cases':cases,'new_training_or_fit':False})
    print(json.dumps({'report':str(report),'cases':cases,'diagnostics':diagnostics},ensure_ascii=False))

if __name__=='__main__':main()
