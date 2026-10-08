"""Live baseline + MC task distributions + separately calibrated error risk."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.special import ndtr
from .task_uncertainty import VERSION,TaskHeads,TaskMC,capture_context,categorical_statistics,bernoulli_statistics,depth_statistics
from .unified_inference import UnifiedInference,export_observable,attach_decisions
from .verifier_risk import risk_vector,feature_names
from .selective_experiment import experimental_action,predicted_answer
from .calibration import predict_risk
from .utils import sha256_file

TASK_FEATURES=['semantic_target_entropy','semantic_target_MI','semantic_target_phrase_score','semantic_matching_missing',
 'semantic_anchor_entropy','semantic_anchor_MI','semantic_anchor_phrase_score','spatial_target_MI','spatial_anchor_MI',
 'relation_predictive_entropy','relation_MI','depth_conditional_variance_m2','depth_MC_mean_variance_m2',
 'completion_predictive_entropy','completion_MI','completion_unobserved_mass']
RISK_FEATURES=feature_names('P1_G44')+TASK_FEATURES

def mixture_interval(means,variances):
    m=np.asarray(means,dtype=float);sd=np.sqrt(variances)
    bounds=[]
    for q in (.025,.975):
        lo=(m-9*sd).min(0);hi=(m+9*sd).max(0)
        for _ in range(55):
            mid=(lo+hi)/2;cdf=ndtr((mid[None]-m)/sd).mean(0)
            lo=np.where(cdf<q,mid,lo);hi=np.where(cdf>=q,mid,hi)
        bounds.append((lo+hi)/2)
    return bounds

def summarize_samples(raw,base_rows,queries,classes,visible_logits):
    stats={k:categorical_statistics(raw[k]) for k in ('target','anchor','semantic_target','semantic_anchor','completion')}
    depth=depth_statistics(raw['depth_mean'],raw['depth_variance']);lower,upper=mixture_interval(raw['depth_mean'],raw['depth_variance'])
    xs=np.arange(32)*20+10;kernel=xs[:,None]-xs[None,:]>12
    rel=np.full((20,len(base_rows)),.5)
    for i,q in enumerate(queries):
        if q.supported and q.predicate in ('left_of','right_of'):
            K=kernel if q.predicate=='right_of' else kernel.T
            t=raw['target'][:,i].astype(np.float64).reshape(20,24,32).sum(1)
            a=raw['anchor'][:,i].astype(np.float64).reshape(20,24,32).sum(1)
            t/=t.sum(-1,keepdims=True);a/=a.sum(-1,keepdims=True)
            mass=np.einsum('ti,ij,tj->t',t,K,a)
            if (mass< -1e-12).any() or (mass>1+1e-12).any():raise ValueError('Invalid geometric integration')
            rel[:,i]=mass.clip(0,1)  # Machine-rounding endpoint correction only.
    relation=bernoulli_statistics(rel)
    visible=1/(1+np.exp(-np.clip(visible_logits,-80,80)))
    results=[]
    def cat(k,i,grid=False):
        s=stats[k];r={f:float(s[f][i]) for f in ('predictive_entropy','expected_entropy','MI')}
        r['mean_distribution']=s['mean'][i].reshape(24,32).tolist() if grid else s['mean'][i].tolist()
        return r
    for i,(base,q) in enumerate(zip(base_rows,queries)):
        anchor_valid=q.supported and bool(q.anchors)
        target_name=q.target.text if q.supported else None
        anchor_name=q.anchors[0].text if anchor_valid else None
        ti=classes.index(target_name) if target_name in classes else None
        ai=classes.index(anchor_name) if anchor_name in classes else None
        semantic={'scope':'closed_set_identity_at_deterministic_predicted_node_not_instance_certification',
            'classes':classes,'target':{**cat('semantic_target',i),'requested_phrase':target_name,
                'requested_class_probability':float(stats['semantic_target']['mean'][i,ti]) if ti is not None else None,
                'predicted_class':classes[int(stats['semantic_target']['mean'][i].argmax())]},
            'anchor':{**cat('semantic_anchor',i),'requested_phrase':anchor_name,
                'requested_class_probability':float(stats['semantic_anchor']['mean'][i,ai]) if ai is not None else None,
                'predicted_class':classes[int(stats['semantic_anchor']['mean'][i].argmax())]} if anchor_valid else None}
        relation_valid=q.supported and q.predicate in ('left_of','right_of')
        completion=cat('completion',i,True)
        completion['target']='reference_visible_support_location_not_full_amodal_shape'
        completion['unobserved_mass']=float((stats['completion']['mean'][i]*(1-visible[i])).sum())
        pred=int(stats['completion']['mean'][i].argmax());completion['map_xy']=[int(20*(pred%32)+10),int(20*(pred//32)+10)]
        d={k:float(v[i]) for k,v in depth.items()};d.update(lower_95_m=float(lower[i]),upper_95_m=float(upper[i]),
             interval_method='Gaussian_mixture_quantiles',reference='simulated_metric_depth',position=base['spatial']['map_pixel_xy'])
        tasks={'version':VERSION,'passes':20,'sampling_scope':'frozen_text_backbone; trained_fusion_dropout_and_three_task_head_dropouts',
            'semantic':semantic,'spatial':{'target':cat('target',i,True),'anchor':cat('anchor',i,True) if anchor_valid else None},
            'relation':{'supported':relation_valid,'compatibility_mean':float(rel[:,i].mean()) if relation_valid else None,
                **{k:float(relation[k][i]) if relation_valid else None for k in ('predictive_entropy','expected_entropy','MI')},
                'binding_presence':'UNVERIFIED','joint_distribution':'product_of_predicted_marginals_approximation'},
            'depth':d,'occlusion_completion':completion,'five_causal_sources_decomposed':False}
        f=[stats['semantic_target']['predictive_entropy'][i],stats['semantic_target']['MI'][i],
           stats['semantic_target']['mean'][i,ti] if ti is not None else 0,float(ti is None),
           stats['semantic_anchor']['predictive_entropy'][i] if anchor_valid else 0,
           stats['semantic_anchor']['MI'][i] if anchor_valid else 0,
           stats['semantic_anchor']['mean'][i,ai] if ai is not None else 0,
           stats['target']['MI'][i],stats['anchor']['MI'][i] if anchor_valid else 0,
           relation['predictive_entropy'][i] if relation_valid else 0,relation['MI'][i] if relation_valid else 0,
           depth['conditional_variance_m2'][i],depth['MC_mean_variance_m2'][i],
           stats['completion']['predictive_entropy'][i],stats['completion']['MI'][i],completion['unobserved_mass']]
        tasks['risk_evidence']=dict(zip(TASK_FEATURES,map(float,f)))
        results.append({**base,'task_uncertainty':tasks})
    return results,{**raw,'relation':rel}

def task_vector(row):
    x=np.concatenate([risk_vector(row,'P1_G44'),[row['task_uncertainty']['risk_evidence'][k] for k in TASK_FEATURES]])
    if x.shape!=(60,) or not np.isfinite(x).all():raise ValueError('Invalid live task risk vector')
    return x

def apply_task_calibrator(rows,calibrator,profile):
    if calibrator['feature_names']!=RISK_FEATURES:raise ValueError('Task risk schema mismatch')
    x=np.stack([task_vector(r) for r in rows]);mean,scale,coef=[np.asarray(calibrator[k]) for k in ('mean','scale','coefficients')]
    risks=predict_risk((x-mean)/scale,coef,calibrator['intercept']);result=[]
    for r,v,xx in zip(rows,risks,x):
        contrib=(xx-mean)/scale*coef
        tau=profile['threshold']
        action=experimental_action(r,float(v),'hard_found',tau if tau is not None else -1.)
        result.append({**r,'reference_live44_decision':r['decision'],
            'decision':{**r['decision'],'risk':float(v),'threshold':tau,'action':action,'risk_producer':VERSION},
            'task_risk_trace':{'feature_names':RISK_FEATURES,'feature_values':xx.tolist(),
                'standardized_logit_contributions':contrib.tolist(),'intercept':calibrator['intercept'],
                'MC_task_evidence_logit_contribution':float(contrib[44:].sum()),'target_MAP_changed':False,
                'contribution_is_causal_attribution':False}})
    return result

class TaskInference:
    def __init__(self,pipeline,heads,classes,calibrator=None,profile=None):
        self.pipeline=pipeline;self.heads=heads.eval().requires_grad_(False);self.classes=classes
        self.sampler=TaskMC(pipeline,self.heads);self.calibrator=calibrator;self.profile=profile

    @classmethod
    def from_bundle(cls,path):
        path=Path(path);manifest=json.loads(path.read_text())
        def check(d):
            p=Path(d['path']);assert sha256_file(p)==d['sha256'],p;return p
        for d in manifest['source_files'].values():check(d)
        pipeline=UnifiedInference.from_bundle(check(manifest['baseline_bundle']))
        heads=TaskHeads(len(manifest['classes'])).to(pipeline.device)
        heads.load_state_dict(load_file(str(check(manifest['task_checkpoint']))))
        cal=json.loads(check(manifest['calibrator']).read_text());profile=json.loads(check(manifest['profile']).read_text())
        return cls(pipeline,heads,manifest['classes'],cal,profile)

    @torch.no_grad()
    def observe(self,features,prompts,seed=24082026):
        context,capture=capture_context(self.pipeline,features,prompts)
        # Auxiliary training cache stores this representation in FP16.
        context={k:v.half().float() for k,v in context.items()}
        batch,cap,anchor,_=capture
        baseline=attach_decisions(export_observable(cap.baseline,batch,prompts,anchor,self.pipeline.bundle_sha256),self.pipeline.calibrator,self.pipeline.profile)
        raw=self.sampler.sample(context,capture,20,seed)
        rows,samples=summarize_samples(raw,baseline,cap.queries,self.classes,cap.baseline['target_logits'].flatten(1).float().cpu().numpy())
        for r in rows:r['task_uncertainty']['seed']=seed
        return rows,samples

    def predict(self,features,prompts,seed=24082026):
        if self.calibrator is None:raise RuntimeError('Decisions require the frozen task calibrator')
        rows,samples=self.observe(features,prompts,seed)
        return apply_task_calibrator(rows,self.calibrator,self.profile),samples
