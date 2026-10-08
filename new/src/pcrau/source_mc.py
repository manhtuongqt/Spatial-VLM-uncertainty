"""Five-label MC source scores and prediction-disagreement statistics.

Labels give source semantics; dropout gives a conditional sampling distribution.
No calibration, causal attribution, aleatoric identification or full-VLM posterior.
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
from torch import nn

from .dataset import SOURCE_CLASSES

VERSION='pcrau_five_source_head_mc_v1'
FIELDS=('source_score_mc_mean','source_score_mc_std','predictive_entropy_normalized',
        'expected_entropy_normalized','mutual_information_normalized')


def source_statistics(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=3 or p.shape[0]<2 or p.shape[1]<1 or p.shape[2]!=5:
        raise ValueError('Expected T>=2, B>=1, five Bernoulli probabilities')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Invalid source probabilities')
    def entropy(x):
        a=np.clip(x,1e-300,1);b=np.clip(1-x,1e-300,1)
        return -(x*np.log(a)+(1-x)*np.log(b))/math.log(2)
    mean=p.mean(0);pe=entropy(mean);ee=entropy(p).mean(0);mi=pe-ee
    if (mi < -1e-10).any():raise ValueError('Entropy concavity was violated')
    return dict(zip(FIELDS,(mean,p.std(0,ddof=0),pe,ee,np.maximum(mi,0))))


class FiveSourceHeadMC:
    """Read-only source-head copy; only its learned dropout is made stochastic."""
    def __init__(self,source_head:nn.Module,amp_dtype=None):
        if any(m.training for m in source_head.modules()):
            raise ValueError('Original source head must be in eval')
        if not isinstance(source_head[0],nn.LayerNorm):
            raise ValueError('Expected the selected LayerNorm → MLP source head')
        linear=[m for m in source_head.modules() if isinstance(m,nn.Linear)]
        dropout=[m for m in source_head.modules() if isinstance(m,nn.Dropout)]
        if not linear or linear[-1].out_features!=5 or len(dropout)!=1 or not 0<dropout[0].p<1:
            raise ValueError('Expected five-output source head with one learned dropout')
        self.head=copy.deepcopy(source_head).eval().requires_grad_(False)
        for p in self.head.parameters():p.grad=None
        self.original=source_head
        self.device=next(self.head.parameters()).device
        self.width=int(source_head[0].normalized_shape[0])
        self.dropout_p=float(dropout[0].p)
        self.amp_dtype=amp_dtype

    def _input(self,context):
        if not isinstance(context,torch.Tensor) or context.ndim!=2 or context.shape[1]!=self.width:
            raise ValueError('Expected observed source-head input [B,context_width]')
        if context.shape[0]<1 or not context.is_floating_point() or not bool(torch.isfinite(context).all()):
            raise ValueError('Invalid source context')
        if context.device!=self.device:raise ValueError('Source context device mismatch')
        if any(m.training for m in self.original.modules()):raise RuntimeError('Original head mode changed')
        return context.detach()

    @torch.no_grad()
    def deterministic_logits(self,context):
        context=self._input(context);self.head.eval()
        with torch.autocast(device_type=self.device.type,enabled=self.amp_dtype is not None,
                            dtype=self.amp_dtype or torch.bfloat16):
            return self.head(context).detach().float()

    @torch.no_grad()
    def sample(self,context,passes=20,seed=24082026):
        context=self._input(context)
        if not isinstance(passes,int) or passes<2:raise ValueError('At least two MC draws required')
        devices=[self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
        result=[]
        try:
            self.head.eval()
            for module in self.head.modules():
                if isinstance(module,nn.Dropout):module.train(True)
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(self.device):torch.cuda.manual_seed(seed)
                with torch.autocast(device_type=self.device.type,enabled=self.amp_dtype is not None,
                                    dtype=self.amp_dtype or torch.bfloat16):
                    for _ in range(passes):result.append(self.head(context).float().sigmoid().cpu().numpy())
        finally:self.head.eval()
        samples=np.stack(result)
        return {'samples':samples,'statistics':source_statistics(samples),'passes':passes,
                'seed':seed,'dropout_p':self.dropout_p,'version':VERSION}


def source_row(result,index):
    return {'version':VERSION,'method':'MC_dropout_model_averaged_five_source_head',
        'sampling_scope':'source_head_only_conditioned_on_fixed_global_feature',
        'passes':result['passes'],'seed':result['seed'],'dropout_p':result['dropout_p'],
        'source_order':SOURCE_CLASSES,
        'sources':{name:{field:float(result['statistics'][field][index,k]) for field in FIELDS}
                   for k,name in enumerate(SOURCE_CLASSES)},
        'source_names_come_from_supervised_labels':True,'spatial_label_is_AMBIGUOUS_proxy':True,
        'MC_mean_is_calibrated_source_probability':False,
        'expected_entropy_is_validated_aleatoric_uncertainty':False,
        'MI_interpretation':'conditional_source_label_model_disagreement_proxy',
        'causal_source_decomposition':False,'full_VLM_epistemic_uncertainty':False}


@torch.no_grad()
def predict_with_source_mc(pipeline,features,prompts,sampler=None,passes=20,seed=24082026):
    """Companion output; leaves original predictions/risk/decision untouched."""
    captured={}
    head=pipeline.shadow.baseline.source_head
    def before(module,args):captured['context']=args[0].detach().clone()
    def after(module,args,output):captured['logits']=output.detach().float().clone()
    hooks=[head.register_forward_pre_hook(before),head.register_forward_hook(after)]
    try:baseline=pipeline.predict(features,prompts)
    finally:
        for h in hooks:h.remove()
    if sampler is None:
        optimization=pipeline.config['optimization']
        amp=torch.bfloat16 if pipeline.device.type=='cuda' and optimization['amp'] else None
        sampler=FiveSourceHeadMC(head,amp)
    zero=sampler.deterministic_logits(captured['context'])
    if not torch.equal(zero,captured['logits']):
        raise RuntimeError('Read-only source-head copy changed deterministic logits')
    result=sampler.sample(captured['context'],passes,seed)
    pipeline.shadow._check_frozen()
    return {'baseline_predictions':baseline,'source_mc_rows':[source_row(result,i) for i in range(len(prompts))],
            'context':captured['context'].cpu(),'deterministic_logits':captured['logits'].cpu(),
            'mc_result':result,'deterministic_copy_exact':True}
