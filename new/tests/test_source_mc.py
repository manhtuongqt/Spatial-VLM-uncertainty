"""Controlled MC tests distinguish source score from uncertainty."""
import copy
import numpy as np
import torch
from torch import nn
from pcrau.source_mc import FiveSourceHeadMC,source_statistics


def test_known_entropy_disagreement_cases():
    stable=np.full((20,1,5),.5);s=source_statistics(stable)
    assert np.allclose(s['predictive_entropy_normalized'],1)
    assert np.allclose(s['expected_entropy_normalized'],1)
    assert np.allclose(s['mutual_information_normalized'],0)
    split=np.array([np.zeros((1,5)),np.ones((1,5))])
    s=source_statistics(split)
    assert np.allclose(s['source_score_mc_mean'],.5)
    assert np.allclose(s['mutual_information_normalized'],1)
    assert np.allclose(s['expected_entropy_normalized'],0)


def test_high_source_score_does_not_mean_high_MC_disagreement():
    s=source_statistics(np.full((20,1,5),.95))
    assert np.allclose(s['source_score_mc_mean'],.95)
    assert np.allclose(s['mutual_information_normalized'],0)
    assert s['source_score_mc_mean'].sum()>1 # Multi-label, not softmax.


def test_MC_mean_is_mean_probabilities_not_sigmoid_mean_logits():
    logits=np.array([[[0.]*5],[[4.]*5]])
    p=1/(1+np.exp(-logits));s=source_statistics(p)
    assert np.allclose(s['source_score_mc_mean'],p.mean(0))
    assert not np.allclose(s['source_score_mc_mean'],1/(1+np.exp(-logits.mean(0))))


def test_sampler_replay_modes_weights_and_rng_preserved():
    torch.manual_seed(42)
    head=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,5)).eval()
    before=copy.deepcopy(head.state_dict());sampler=FiveSourceHeadMC(head)
    x=torch.randn(4,8);rng=torch.random.get_rng_state().clone()
    a=sampler.sample(x,20,7);b=sampler.sample(x,20,7);c=sampler.sample(x,20,8)
    assert np.array_equal(a['samples'],b['samples']) and not np.array_equal(a['samples'],c['samples'])
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in before.items())
    assert all(not m.training for m in head.modules()) and all(not m.training for m in sampler.head.modules())
    assert all(p.grad is None and not p.requires_grad for p in sampler.head.parameters())
    assert torch.equal(sampler.deterministic_logits(x),head(x))


def test_invalid_samples_and_context_rejected():
    for p in (np.zeros((1,1,5)),np.full((2,1,5),float('nan')),np.full((2,1,5),2.)):
        try:source_statistics(p)
        except ValueError:continue
        raise AssertionError('Invalid MC probabilities accepted')
    h=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,5),nn.Dropout(.1)).eval();sampler=FiveSourceHeadMC(h)
    try:sampler.sample({'mask':torch.ones(1)},20)
    except ValueError:return
    raise AssertionError('Annotation channel accepted')
