#!/usr/bin/env python3
"""Read-only paired reporting from a completed C1/P1/P2 pilot; no inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcrau.anchor_peak_pilot import aggregate, family_bootstrap, locked_gates, paired_changes


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('new/outputs/pcrau_s1_anchor_peak_pilot_v2_20261005'))
    args=parser.parse_args();root=args.run.resolve()
    read=lambda path:json.loads(path.read_text())
    summary=read(root/'summary.json');provenance=read(root/'provenance.json')
    assert summary['status']=='PILOT_COMPLETE'
    files=[p for p in root.rglob('*') if p.is_file()]
    source_hashes={str(p):sha(p) for p in files}
    rows=[json.loads(x) for x in (root/'evaluator_rows.jsonl').read_text().splitlines()]
    for split in ('train','dev'):
        rs=[r for r in rows if r['split']==split]
        assert aggregate(rs)==summary['groups'][split]
        for a,b in (('P1','C1'),('P1','P2')):
            assert paired_changes(rs,a,b)==summary['paired_changes'][split][f'{a}_vs_{b}']
            assert family_bootstrap(rs,a,b)==summary['family_bootstrap'][split][f'{a}_vs_{b}']
    assert locked_gates([r for r in rows if r['split']=='dev'],True,
        all(summary['latency'][a]['gate_pass'] for a in ('C1','P1','P2')))==summary['gates']
    destination=root/'paired_report';destination.mkdir(exist_ok=False)
    def write_csv(path, data, fields):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fields);writer.writeheader()
            for row in data:writer.writerow({k:row[k] for k in fields})
    family=[json.loads(x) for x in (root/'family_paired_deltas.jsonl').read_text().splitlines()]
    write_csv(destination/'family_paired_deltas.csv',family,
        ['split','family_id','contrast','fixed_count','broken_count','net_hits','swap_delta','found_both_delta'])
    empty=[]
    for r in rows:
        if r['anchor_pixels']:continue
        for a,m in r['models'].items():
            empty.append({'sample_id':r['sample_id'],'split':r['split'],'arm':a,
                **{k:m[k] for k in ('max_logit','mean_zero_bce','sigmoid_max')},
                'delta_sigmoid_vs_M0':m['sigmoid_max']-r['models']['M0']['sigmoid_max'],
                'delta_sigmoid_vs_C1':m['sigmoid_max']-r['models']['C1']['sigmoid_max']})
    write_csv(destination/'empty_cases.csv',empty,list(empty[0]))
    (destination/'remaining_dev_misses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows
        if r['split']=='dev' and r['anchor_pixels'] and not r['models']['P1']['anchor_hit']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=read(root/'epoch_history.json')['records']
    fig,axs=plt.subplots(1,3,figsize=(13,3.9),constrained_layout=True)
    for a,color in (('C1','#2563eb'),('P1','#d97706'),('P2','#16a34a')):
        hs=[r for r in history if r['arm']==a];epochs=[r['epoch'] for r in hs]
        values=([r['dev']['models'][a]['anchor_hits'] for r in hs],
            [r['dev']['models'][a]['empty_mean_sigmoid_max'] for r in hs],
            [r['mean_train_losses']['total'] for r in hs])
        selected=summary['arms'][a]['selected_epoch'];idx=epochs.index(selected)
        for ax,vs in zip(axs,values):
            ax.plot(epochs,vs,'o-',color=color,label=a,markersize=3)
            ax.scatter([selected],[vs[idx]],marker='*',s=110,color=color,zorder=5)
    titles=['Dev anchor hits / 61','Dev mean empty sigmoid-max (3 cases)','Mean train objective (objectives differ)']
    axs[0].axhline(49,color='black',linestyle='--',linewidth=1,label='gate 49')
    for ax,title in zip(axs,titles):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('C1/P1/P2: one seed, dev-selected checkpoints (*)',fontsize=12)
    fig.savefig(destination/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# C1/P1/P2 paired report — 05/10/2026','',
        '**Pilot complete; upgrade gates FAIL. No verifier/calibration/IID.**','',
        '| Arm | Selected epoch | Epochs / steps | Anchor /61 | Swap /15 | FOUND-both /16 |',
        '|---|---:|---:|---:|---:|---:|']
    for a in ('M0','C1','P1','P2'):
        m=summary['groups']['dev']['models'][a];info=summary['arms'].get(a)
        lines.append(f"| {a} | {info['selected_epoch'] if info else '—'} | {str(info['epochs'])+' / '+str(info['optimizer_steps']) if info else '—'} | {m['anchor_hits']} | {m['swap_both_hits']} | {m['found_both_hits']} |")
    lines+=['','P1−C1: no dev correction or regression; no added binding benefit demonstrated.',
        'P1−P2: +4 hits, +2 pairs; all four corrected cases concern lemon in two families.',
        'P1 remains below49/61. P1 improves3/5 train-empty peaks vsC1, worsens2/5;',
        'P1 still increases3/5 vsM0. On dev empties P1 improvesall3 vsM0 but only1/3 vsC1.',
        '', '| Gate | Pass |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in summary['gates']['gates'].items()]
    lines+=['','![Learning curves](learning_curves.png)','',
        'Raw train objectives differ between C1 and P1/P2; their magnitudes are not directly comparable.',
        'Family bootstrap is conditional on dev-selected checkpoints. P1−P2 hit delta CI includes0;',
        'P1−C1 observed hit/pair labels coincide, so bootstrap CI[0,0] does not prove population equivalence.',
        '', 'See [family deltas](family_paired_deltas.csv), [each empty case](empty_cases.csv),',
        '[13 remaining misses](remaining_dev_misses.jsonl), [full summary](../summary.json)',
        'and [Vietnamese report](../../../../plan/S1_ANCHOR_PEAK_PILOT_V2_20261005.md).']
    (destination/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    assert all(sha(p)==digest for p,digest in source_hashes.items())
    assert all(sha(p)==digest for p,digest in provenance['protected_sha256'].items())
    (destination/'provenance.json').write_text(json.dumps({'original_run_files_sha256':source_hashes,
        'source_sha256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        'checks':{'metrics_gates_bootstrap_reproduced':True,'original_artifacts_unchanged':True,
            'protected_files_unchanged':True,'no_inference_or_training':True}},indent=2)+'\n')
    print(json.dumps({'report':str(destination),'original_files_unchanged':len(source_hashes),'gate_pass':False}))


if __name__=='__main__':main()
