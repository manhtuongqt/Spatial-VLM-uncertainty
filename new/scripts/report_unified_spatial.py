#!/usr/bin/env python3
"""Read frozen live predictions; write real dev cases without altering evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.utils import read_json, sha256_file, atomic_json


def rows(path): return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--refresh-report', action='store_true', help='Regenerate only this script\'s report files')
    args = parser.parse_args(); root = args.root.resolve()
    report = root/'report'
    protected = {str(p): sha256_file(p) for p in root.rglob('*') if p.is_file() and report not in p.parents}
    report.mkdir(exist_ok=args.refresh_report)
    prediction = {r['sample_id']: r for r in rows(root/'dev/runtime_predictions.jsonl')}
    evaluator = {r['sample_id']: r['evaluation'] for r in rows(root/'dev/evaluator_join.jsonl')}
    anchors = {r['sample_id']: r for r in rows(root/'dev/anchor_evaluation.jsonl')}
    lock = read_json(root/'frozen/freeze_lock.json')
    cfg = read_json(Path(lock['config']['path'])); ds = ArchivedPCRAUDataset(cfg, 'dev')
    entries = {e['sample_id']:e for e in ds.entries}
    selected = [('Correct pair', 'v211dev_family_000001__clean'),
                ('Swapped query', 'v211dev_family_000001__relation_counterfactual'),
                ('Geometry good, evidence weak', 'v211dev_family_000001__depth_corruption'),
                ('Wrong anchor, positive margin', 'v211dev_family_000383__relation_counterfactual'),
                ('Negative margin, false accept', 'v211dev_family_000042__clean'),
                ('Empty anchor, peak still exists', 'v211dev_family_000392__clean')]
    for status, label in [('DIRECT_BYPASS','Direct bypass'),('PARSE_UNSUPPORTED','Outside scope')]:
        selected.append((label, next(r['sample_id'] for r in prediction.values()
                                    if r['verifiers']['P1']['scope_status']==status)))
    cases = []; table = ['| Case dev | Target → quan hệ → anchor | MAP target / anchor | Margin px / compatibility | Risk / action | Hậu kiểm |',
                         '|---|---|---|---|---|---|']
    fig, axes = plt.subplots(2,4,figsize=(17,10.5))
    for index, (label, sid) in enumerate(selected):
        r = prediction[sid]; v = r['verifiers']['P1']; q = v['query']; e = entries[sid]
        rgbpath = ds.layout.dataset_path(e['feature_input']['rgb_path'])
        assert sha256_file(rgbpath)==e['feature_input']['rgb_sha256']
        rgb = cv2.cvtColor(cv2.imread(str(rgbpath)), cv2.COLOR_BGR2RGB)
        ax = axes.flat[index]; ax.imshow(rgb)
        tx,ty = v['target_peak_xy']; ax.scatter([tx],[ty],marker='x',c='#00cfff',s=95,linewidths=2)
        if v['anchor_peak_xy'] is not None:
            xx,yy = v['anchor_peak_xy']; ax.scatter([xx],[yy],marker='o',facecolors='none',edgecolors='#ffb100',s=100,linewidths=2)
        short = sid.removeprefix('v211dev_family_'); decision = r['decision']
        ax.set_title(f'({chr(97+index)}) {label}\n{short}', fontsize=9, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
        relation = q['predicate'] or 'unsupported'
        target = q['target']['text'] if q['target'] else 'unparsed'
        anchor = q['anchors'][0]['text'] if q['anchors'] else '—'
        footer = f'{target} / {relation} / {anchor}\n'
        if v['signed_margin_px'] is not None:
            footer += f"margin={v['signed_margin_px']} px; compatibility={v['pair_compatibility']:.3f}\n"
        else: footer += v['scope_status']+'\n'
        footer += f"risk={decision['risk']:.4f}; {decision['action']}"
        ax.set_xlabel(footer, fontsize=8)
        case = {'label':label,'sample_id':sid,'rgb_path':str(rgbpath),'rgb_sha256':sha256_file(rgbpath),
                'prediction':r,'evaluator_only':evaluator[sid],'anchor_evaluator_only':anchors.get(sid)}
        atomic_json(report/f'case_{index+1:02d}.json',case); cases.append(case)
        hindsight = evaluator[sid]['answerability_state']
        if sid in anchors: hindsight += '; anchor='+str(anchors[sid]['hit'])+'; pixels='+str(anchors[sid]['pixels'])
        table.append(f"| `{short}` | {target} → {relation} → {anchor} | {v['target_peak_xy']} / {v['anchor_peak_xy']} | {v['signed_margin_px']} / {v['pair_compatibility'] if v['pair_compatibility'] is None else format(v['pair_compatibility'],'.6f')} | {decision['risk']:.6f} / {decision['action']} | {hindsight} |")
    fig.suptitle('P-CRA-U spatial variant — live evidence on real dev RGB',fontsize=15,x=.02,ha='left')
    fig.text(.02,.015,'Cyan ×: predicted target MAP · orange circle: predicted anchor MAP · binding/presence unverified · perception decisions only',fontsize=9)
    fig.subplots_adjust(left=.025,right=.985,top=.88,bottom=.14,wspace=.06,hspace=.60)
    fig.savefig(report/'dev_cases.png',dpi=160,facecolor='white')
    fig.savefig(report/'dev_cases.pdf',facecolor='white'); plt.close(fig)
    (report/'CASE_TABLE.md').write_text('# Case thật trên dev\n\nMasks/nhãn chỉ dùng hậu kiểm. Geometry compatibility không xác nhận identity.\n\n'+'\n'.join(table)+'\n')
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    atomic_json(report/'provenance.json',{'read_only_inputs':protected,'case_ids':[s for _,s in selected],
                'all_input_hashes_unchanged':True,'script_sha256':sha256_file(Path(__file__))})
    print('REPORT',report)


if __name__=='__main__': main()
