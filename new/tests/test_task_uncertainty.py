import numpy as np
import torch
from pcrau.task_uncertainty import TaskHeads,head_losses,categorical_statistics,bernoulli_statistics,depth_statistics

def test_known_uncertainties():
    stable=np.full((20,2,2),.5)
    s=categorical_statistics(stable)
    assert np.allclose(s['predictive_entropy'],1) and np.allclose(s['MI'],0)
    alternating=np.asarray([[[1.,0.]],[[0.,1.]]])
    assert np.allclose(categorical_statistics(alternating)['MI'],1)
    assert np.allclose(bernoulli_statistics(np.full((20,1),.95))['MI'],0)
    d=depth_statistics(np.array([[1.],[3.]]),np.array([[.2],[.4]]))
    assert np.allclose(d['MC_mean_variance_m2'],1) and np.allclose(d['total_predictive_variance_m2'],1.3)

def test_likelihood_gradient_and_boundary():
    torch.manual_seed(9);h=TaskHeads()
    ctx={k:torch.randn(2,24,32,128) for k in ('rgb_projection','depth_projection','fused')}
    ctx.update(target_phrase=torch.randn(2,128),target_logits=torch.randn(2,24,32))
    labels={'semantic':torch.zeros(2,24,32,dtype=torch.long),'depth':torch.ones(2,24,32),
            'depth_valid':torch.ones(2,24,32,dtype=torch.bool),'completion':torch.zeros(2,24,32)}
    labels['completion'][0,8:12,14:18]=1
    o=h(ctx);loss=head_losses(o,labels);assert all(torch.isfinite(v) for v in loss.values())
    loss['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
    assert (o['depth_variance']>0).all()
    try:h({**ctx,'target_mask':labels['completion']})
    except ValueError:pass
    else:raise AssertionError('Oracle boundary accepted a mask')

def test_invalid_probability_rejected():
    for p in (np.ones((20,1,2)),np.full((20,1,2),np.nan),np.ones((1,1,2))):
        try:categorical_statistics(p)
        except ValueError:pass
        else:raise AssertionError('Invalid distribution accepted')
