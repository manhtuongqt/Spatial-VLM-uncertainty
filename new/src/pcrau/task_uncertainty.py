"""Opt-in task distributions; source labels and oracle masks are never inputs."""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn

VERSION='pcrau_task_uncertainty_s6_v1'
HEAD_INPUT_KEYS={'rgb_projection','depth_projection','fused','target_phrase','target_logits'}

def categorical_statistics(p):
    a=np.asarray(p,dtype=np.float64)
    if a.ndim<3 or a.shape[0]<2 or a.shape[-1]<2 or not np.isfinite(a).all():
        raise ValueError('Expected T,B,...,K categorical distributions')
    if (a<0).any() or not np.allclose(a.sum(-1),1,atol=1e-5,rtol=0):raise ValueError('Unnormalized categorical distributions')
    h=lambda x:-(x*np.log(np.maximum(x,1e-300))).sum(-1)/math.log(a.shape[-1])
    mean=a.mean(0);pe=h(mean);ee=h(a).mean(0);mi=pe-ee
    if (mi< -1e-8).any():raise ValueError('Entropy concavity failed')
    return {'mean':mean,'std':a.std(0),'predictive_entropy':pe,'expected_entropy':ee,'MI':np.maximum(mi,0)}

def bernoulli_statistics(p):
    a=np.asarray(p,dtype=np.float64)
    return categorical_statistics(np.stack([1-a,a],-1))

def depth_statistics(mean,variance):
    m=np.asarray(mean,dtype=np.float64);v=np.asarray(variance,dtype=np.float64)
    if m.shape!=v.shape or m.shape[0]<2 or not np.isfinite(m).all() or not np.isfinite(v).all() or (v<=0).any():
        raise ValueError('Invalid Gaussian mixture parameters')
    model=m.var(0);conditional=v.mean(0)
    return {'mean_m':m.mean(0),'conditional_variance_m2':conditional,
            'MC_mean_variance_m2':model,'total_predictive_variance_m2':conditional+model}

class TaskHeads(nn.Module):
    def __init__(self,classes=22,dropout=.1):
        super().__init__()
        def mlp(n,out):return nn.Sequential(nn.LayerNorm(n),nn.Linear(n,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,out))
        self.semantic=mlp(128,classes)
        self.depth=mlp(256,2)
        self.completion=mlp(259,1)
        nn.init.zeros_(self.completion[-1].weight);nn.init.zeros_(self.completion[-1].bias)

    def forward(self,context):
        if set(context)!=HEAD_INPUT_KEYS:raise ValueError('Task heads reject annotations/metadata')
        rgb,dep,fused=context['rgb_projection'],context['depth_projection'],context['fused']
        if rgb.shape!=dep.shape or rgb.ndim!=4 or rgb.shape[1:]!=(24,32,128):raise ValueError('Invalid observable projections')
        sem=self.semantic(rgb.float())
        dv=self.depth(torch.cat([rgb.float(),dep.float()],-1))
        mu=torch.nn.functional.softplus(dv[...,0])+.05
        var=torch.nn.functional.softplus(dv[...,1])+1e-4
        n=len(rgb);phrase=context['target_phrase'].float()[:,None,None,:].expand(n,24,32,128)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,24,device=rgb.device),torch.linspace(-1,1,32,device=rgb.device),indexing='ij')
        xy=torch.stack([xx,yy],-1)[None].expand(n,-1,-1,-1)
        inp=torch.cat([fused.float(),phrase,context['target_logits'].float()[...,None],xy],-1)
        completion=context['target_logits'].float()+self.completion(inp).squeeze(-1)
        return {'semantic_logits':sem,'depth_mean':mu,'depth_variance':var,'completion_logits':completion}

def head_losses(output,labels):
    sem=torch.nn.functional.cross_entropy(output['semantic_logits'].permute(0,3,1,2),labels['semantic'])
    valid=labels['depth_valid'];truth=labels['depth']
    v=output['depth_variance'];mu=output['depth_mean']
    depth=(.5*((truth-mu).square()/v+v.log()+math.log(2*math.pi)))[valid].mean()
    mask=labels['completion'];mass=mask.flatten(1).sum(-1);good=mass>0
    q=mask.flatten(1)/mass[:,None].clamp_min(1e-12)
    ce=-(q*output['completion_logits'].flatten(1).log_softmax(-1)).sum(-1)
    completion=ce[good].mean() if good.any() else output['completion_logits'].sum()*0
    return {'semantic':sem,'depth':depth,'completion':completion,'total':sem+depth+completion}

@torch.no_grad()
def capture_context(pipeline,features,prompts):
    from .unified_inference import make_batch
    from .engine import autocast_context
    batch=make_batch(features,prompts,pipeline.config['model'],pipeline.device)
    found={}
    def graph_hook(m,args,out):found.update(target_query=out['target'].detach(),relation_context=out['relation_context'].detach())
    hook=pipeline.shadow.baseline.query_graph.register_forward_hook(graph_hook)
    try:
        with autocast_context(pipeline.device,pipeline.config['optimization']):cap=pipeline.shadow.capture(batch,prompts);anchor=pipeline.shadow.predict(cap)
    finally:hook.remove()
    masks=torch.tensor([q.token_masks()['target_token_mask'] for q in cap.queries],device=pipeline.device,dtype=torch.float32)
    # Unsupported phrases use full observable text; flags remain explicit.
    masks=torch.where((masks.sum(-1)>0)[:,None],masks,cap.token_mask.float())
    phrase=torch.einsum('bl,bld->bd',masks,cap.text.float())/masks.sum(-1,keepdim=True).clamp_min(1)
    with autocast_context(pipeline.device,pipeline.config['optimization']):
        rgb=pipeline.shadow.baseline.fusion.rgb(batch['r0']);dep=pipeline.shadow.baseline.fusion.depth(batch['d0'])
    return {'rgb_projection':rgb.float(),'depth_projection':dep.float(),'fused':cap.fused.float(),
            'target_phrase':phrase,'target_logits':cap.baseline['target_logits'].float()},(batch,cap,anchor,found)

class TaskMC:
    """Samples learned fusion dropout and task-head dropout; original bundle stays eval."""
    def __init__(self,pipeline,heads):
        self.pipeline=pipeline
        self.fusion=copy.deepcopy(pipeline.shadow.baseline.fusion).eval().requires_grad_(False)
        self.heads=copy.deepcopy(heads).eval().requires_grad_(False)
        self.device=pipeline.device

    @torch.no_grad()
    def sample(self,context,capture,passes=20,seed=24082026):
        from .engine import autocast_context
        if passes<2:raise ValueError('Need multiple MC samples')
        batch,cap,detanchor,found=capture
        pipeline=self.pipeline;baseline=pipeline.shadow.baseline
        pipeline.shadow._check_frozen()
        # P1 query is fixed: MC only changes the visual fused representation.
        weights=cap.phrase_masks.float()
        pooled=torch.einsum('bsl,bld->bsd',weights,cap.text.float())/weights.sum(-1,keepdim=True).clamp_min(1)
        delta=pipeline.shadow.residual(pooled)*cap.active_slots[...,None]
        queries=cap.old_queries+delta.to(cap.old_queries.dtype)
        points=cap.baseline['target_logits'].flatten(1).argmax(-1)
        ap=detanchor[:,0].flatten(1).argmax(-1)
        samples={k:[] for k in ('target','anchor','semantic_target','semantic_anchor','depth_mean','depth_variance','completion')}
        device_index=self.device.index if self.device.index is not None else torch.cuda.current_device()
        try:
            for net in (self.fusion,self.heads):
                net.eval()
                for m in net.modules():
                    if isinstance(m,nn.Dropout):m.train(True)
            with torch.random.fork_rng(devices=[device_index]):
                torch.random.default_generator.manual_seed(seed);torch.cuda.manual_seed(seed)
                for _ in range(passes):
                    with autocast_context(self.device,pipeline.config['optimization']):
                        fused,_=self.fusion(batch['r0'],batch['d0'],found['relation_context'])
                        target=baseline.target_head(fused,found['target_query'])
                        anchor=baseline.anchor_head(fused,queries)[:,0]
                    # Head likelihood evaluation in FP32, as during auxiliary training.
                    o=self.heads({**context,'fused':fused.float(),'target_logits':target.float()})
                    st=o['semantic_logits'].flatten(1,2).softmax(-1)
                    idx=torch.arange(len(points),device=self.device)
                    for k,v in [('target',target.flatten(1).float().softmax(-1)),('anchor',anchor.flatten(1).float().softmax(-1)),
                        ('semantic_target',st[idx,points]),('semantic_anchor',st[idx,ap]),
                        ('depth_mean',o['depth_mean'].flatten(1)[idx,points]),
                        ('depth_variance',o['depth_variance'].flatten(1)[idx,points]),
                        ('completion',o['completion_logits'].flatten(1).softmax(-1))]:samples[k].append(v.cpu().numpy())
        finally:
            self.fusion.eval();self.heads.eval()
        pipeline.shadow._check_frozen()
        return {k:np.stack(v) for k,v in samples.items()}
