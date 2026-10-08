"""Frozen verifier evidence calibration; evaluation fields used only in fit."""
from __future__ import annotations

import hashlib
import numpy as np

from .calibration import FEATURE_NAMES,fit_logistic,predict_risk
from .horizontal_verifier import ANCHOR_FEATURES,EXTRA_FEATURES,VERSION
from .selective_experiment import EXTRA_RISK_FEATURES,apply_risk,fit_risk,probability_metrics,vector

MODELS={'B33':('base',None),'P1_A41':('anchor','P1'),
    'M0_G44':('geometry','M0'),'C1_G44':('geometry','C1'),'P1_G44':('geometry','P1')}


def feature_names(model):
    kind,_=MODELS[model]
    return FEATURE_NAMES+EXTRA_RISK_FEATURES+([] if kind=='base' else
        ['verifier_'+k for k in (ANCHOR_FEATURES if kind=='anchor' else EXTRA_FEATURES)])


def risk_vector(row,model):
    kind,arm=MODELS[model]
    base=vector(row,True)
    if kind=='base':return base
    trace=row['verifiers'][arm]
    if trace['version']!=VERSION or trace['binding_status']!='UNVERIFIED' or trace['target_map_changed']:
        raise ValueError('Verifier schema/claim drift')
    names=ANCHOR_FEATURES if kind=='anchor' else EXTRA_FEATURES
    result=np.concatenate([base,np.asarray([trace['features'][k] for k in names])])
    if not np.isfinite(result).all():raise ValueError('Nonfinite observable risk evidence')
    return result


def fit_verifier_risk(rows,model,bundle_sha256):
    if model=='B33':
        calibrator,oof=fit_risk(rows,True,.01)
    else:
        families=[r['family_id'] for r in rows]
        unique=sorted(set(families),key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        if len(unique)<5:raise ValueError('At least five calibration families required')
        assignment={f:i%5 for i,f in enumerate(unique)}
        x=np.stack([risk_vector(r,model) for r in rows])
        y=np.asarray([r['evaluation']['error_event'] for r in rows],dtype=float)
        oof=np.full(len(rows),np.nan)
        for fold in range(5):
            held=np.asarray([assignment[f]==fold for f in families]);train=~held
            mean,scale=x[train].mean(0),x[train].std(0);scale[scale<1e-8]=1.
            if len(np.unique(y[train]))==1:oof[held]=y[train].mean()
            else:
                coefficients,intercept=fit_logistic((x[train]-mean)/scale,y[train],.01)
                oof[held]=predict_risk((x[held]-mean)/scale,coefficients,intercept)
        if not np.isfinite(oof).all():raise ValueError('Invalid family crossfit risks')
        mean,scale=x.mean(0),x.std(0);scale[scale<1e-8]=1.
        coefficients,intercept=fit_logistic((x-mean)/scale,y,.01)
        calibrator={'method':'logistic_verifier_family_crossfit_fit_all','l2':.01,
            'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coefficients.tolist(),
            'intercept':intercept,'fit_samples':len(rows),'fit_families':len(unique),
            'family_folds':assignment,'crossfit_metrics':probability_metrics(oof,y),
            'feature_names':feature_names(model)}
    calibrator.update(model=model,bundle_sha256=bundle_sha256,primary=model=='P1_G44')
    return calibrator,oof


def apply_verifier_risk(rows,calibrator):
    if any(r['verifier_bundle_sha256']!=calibrator['bundle_sha256'] for r in rows):
        raise ValueError('Verifier bundle does not match calibrator')
    model=calibrator['model']
    if calibrator['feature_names']!=feature_names(model):raise ValueError('Feature schema drift')
    if model=='B33':return apply_risk(rows,calibrator)
    x=np.stack([risk_vector(r,model) for r in rows])
    mean,scale,coefficients=[np.asarray(calibrator[k]) for k in ('mean','scale','coefficients')]
    if (scale<=0).any() or not np.isfinite(x).all():raise ValueError('Invalid standardization/input')
    return predict_risk((x-mean)/scale,coefficients,calibrator['intercept'])
