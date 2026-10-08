#!/usr/bin/env python3
"""Report frozen cached-evidence reevaluation; no model fitting or execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('new/outputs/pcrau_horizontal_verifier_iid_cached_20261005'))
    args=parser.parse_args();root=args.run.resolve()
    summary=json.loads((root/'summary.json').read_text());inputs={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
    report=root/'report';report.mkdir(exist_ok=False)
    with (report/'comparison.csv').open('w',newline='') as f:
        fields=['model','threshold','cal_oof_accepted','cal_oof_errors','iid_accepted','iid_correct','iid_errors','iid_risk','brier','nll','ece','aurc']
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for name,result in summary['iid'].items():
            c=summary['calibration_profiles'][name];m=result['policy'];p=result['probability']
            writer.writerow(dict(model=name,threshold=c['threshold_fit']['threshold'],cal_oof_accepted=c['crossfit_calibration']['accepted'],
                cal_oof_errors=c['crossfit_calibration']['errors'],iid_accepted=m['accepted'],iid_correct=m['correct_acceptances'],
                iid_errors=m['errors'],iid_risk=m['empirical_risk'],brier=p['brier'],nll=p['nll'],ece=p['ece_10'],aurc=p['aurc']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(x) for x in (root/'test_iid/predictions.jsonl').read_text().splitlines()]
    risk=np.load(root/'test_iid/risk_scores.npz')
    eligible=np.asarray([max(r['answerability_probabilities'],key=r['answerability_probabilities'].get)=='FOUND' for r in rows])
    errors=np.asarray([r['evaluation']['error_event'] for r in rows])
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for name,color in [('B33','#2563eb'),('P1_A41','#64748b'),('C1_G44','#16a34a'),('P1_G44','#d97706')]:
        inds=np.flatnonzero(eligible);order=inds[np.argsort(risk[name][inds],kind='stable')]
        count=np.arange(1,len(order)+1);curve=np.cumsum(errors[order])/count
        ax.plot(100*count/len(rows),100*curve,label=name,color=color,linewidth=1.5)
        m=summary['iid'][name]['policy'];ax.scatter([100*m['coverage']],[100*m['empirical_risk']],s=60,color=color,zorder=5)
    ax.axhline(7.5,linestyle='--',color='black',linewidth=1,label='Calibration budget 7.5%')
    ax.set(xlabel='Coverage (%)',ylabel='Empirical selective risk (%)',title='Old IID: hard_found risk–coverage; dots = frozen profiles')
    ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylim(bottom=0)
    fig.savefig(report/'risk_coverage.png',dpi=160);plt.close(fig)
    (report/'provenance.json').write_text(json.dumps({'original_run_files_sha256':inputs,
        'source_sha256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        'no_inference_fit_or_threshold_selection':True},indent=2)+'\n')
    assert all(sha(p)==d for p,d in inputs.items())
    print(json.dumps({'report':str(report),'original_files_unchanged':len(inputs)}))


if __name__=='__main__':main()
