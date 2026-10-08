"""Read-only paired analysis of frozen perception decisions; never fits models."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .dataset import ANSWER_CLASSES
from .metrics import confusion_metrics
from .selective_experiment import policy_result, predicted_answer, probability_metrics


def model_metrics(evaluated, risk, threshold):
    policy, _ = policy_result(evaluated, risk, 'hard_found', threshold)
    truth = [ANSWER_CLASSES.index(r['evaluation']['answerability_state']) for r in evaluated]
    predicted = [ANSWER_CLASSES.index(predicted_answer(r)) for r in evaluated]
    present = [r for r in evaluated if r['evaluation']['target_exists']]
    found = [r for r in evaluated if r['evaluation']['answerability_state']=='FOUND']
    error = np.asarray([r['evaluation']['error_event'] for r in evaluated])
    return {'policy': policy, 'probability': probability_metrics(risk, error),
            'answerability': confusion_metrics(truth, predicted, 4),
            'grounding': {'target_present': len(present),
                'target_present_hits': sum(r['evaluation']['map_inside_target'] for r in present),
                'truth_FOUND': len(found),
                'FOUND_MAP_hits': sum(r['evaluation']['map_inside_target'] for r in found),
                'composite_error_prevalence': float(error.mean())}}


def paired_changes(primary, reference):
    """Each arm carries its own error event; equality is checked by caller."""
    if [r['sample_id'] for r in primary] != [r['sample_id'] for r in reference]:
        raise ValueError('Paired rows must have matching ordered sample IDs')
    changes = []
    counts = {k:0 for k in ('new_correct_accept','removed_correct_accept',
                             'new_error_accept','removed_error_accept')}
    for a,b in zip(primary,reference):
        if a['family_id']!=b['family_id']:
            raise ValueError('Paired family IDs differ')
        accept_a = a['decision']['action']=='EXECUTE'
        accept_b = b['decision']['action']=='EXECUTE'
        error_a, error_b = a['evaluation']['error_event'],b['evaluation']['error_event']
        if accept_a!=accept_b:
            category = ('new_error_accept' if error_a else 'new_correct_accept') if accept_a else (
                'removed_error_accept' if error_b else 'removed_correct_accept')
            counts[category]+=1
        else: category = 'same_acceptance'
        if a['decision']['action']!=b['decision']['action']:
            changes.append({'sample_id':a['sample_id'],'family_id':a['family_id'],
                'variant':a['variant'],'truth':a['evaluation']['answerability_state'],
                'primary_action':a['decision']['action'],'reference_action':b['decision']['action'],
                'primary_risk':a['decision']['risk'],'reference_risk':b['decision']['risk'],
                'primary_error':error_a,'reference_error':error_b,'acceptance_category':category,
                'scope':a['verifiers']['P1']['scope_status']})
    return {'counts':counts,'changed_actions':len(changes),
            'changed_acceptance':sum(counts.values())},changes


def paired_family_bootstrap(primary, reference, seed=24082026, resamples=5000):
    if [r['sample_id'] for r in primary] != [r['sample_id'] for r in reference]:
        raise ValueError('Mismatched paired samples')
    groups = defaultdict(list)
    for i,(a,b) in enumerate(zip(primary,reference)):
        if a['family_id']!=b['family_id']: raise ValueError('Mismatched family')
        groups[a['family_id']].append(i)
    if not groups: raise ValueError('No paired families')
    tables=[]
    for family in sorted(groups):
        ids=groups[family]; values=[]
        for rows in (primary, reference):
            accept=np.asarray([rows[i]['decision']['action']=='EXECUTE' for i in ids])
            error=np.asarray([rows[i]['evaluation']['error_event'] for i in ids])
            risk=np.asarray([rows[i]['decision']['risk'] for i in ids])
            clipped=risk.clip(1e-7,1-1e-7)
            values.extend([len(ids),accept.sum(),(accept & error).sum(),(accept & ~error).sum(),
                (~error).sum(),np.square(risk-error).sum(),
                -(error*np.log(clipped)+(1-error)*np.log(1-clipped)).sum()])
        tables.append(values)
    table=np.asarray(tables,dtype=float)
    picks=np.random.default_rng(seed).integers(0,len(groups),size=(resamples,len(groups)))
    sampled=table[picks].sum(1); point=table.sum(0)
    metrics={}
    for name,num,den in [('coverage',1,0),('correct_acceptance_rate',3,0),
                         ('selective_risk',2,1),('valid_recall',3,4),('brier',5,0),('nll',6,0)]:
        valid=(sampled[:,den]>0)&(sampled[:,den+7]>0)
        delta=sampled[valid,num]/sampled[valid,den]-sampled[valid,num+7]/sampled[valid,den+7]
        metrics[name]={'point':float(point[num]/point[den]-point[num+7]/point[den+7])
            if point[den]>0 and point[den+7]>0 else None,
            'ci95':np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,
            'valid_resamples':int(valid.sum())}
    return {'families':len(groups),'resamples':resamples,'seed':seed,'metrics':metrics,
            'fixed_models_profiles':True,'includes_fit_or_selection_uncertainty':False}
