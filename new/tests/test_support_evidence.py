import torch
from pcrau.support_evidence import SUPPORT_NAMES, ranker_features, support_features


def test_support_is_finite_and_inactive_slots_masked():
    e=torch.randn(2,2222);p=torch.rand(2,4).softmax(-1)
    mask=torch.zeros(2,3,dtype=torch.bool)
    s=support_features(e,p,mask,mask)
    assert s.shape==(2,len(SUPPORT_NAMES)) and torch.isfinite(s).all()
    e2=e.clone();e2[:,26+896:26+1664]=10000;e2[:,26+1792:26+2176]=-10000;e2[:,26+2181:]=10000
    torch.testing.assert_close(s,support_features(e2,p,mask,mask))
    assert ranker_features(e,p,mask,mask,'scalar').shape==(2,30)
    assert ranker_features(e,p,mask,mask,'detail').shape==(2,2226)
    assert ranker_features(e,p,mask,mask,'support').shape==(2,2268)


def test_support_rejects_wrong_dimensions():
    try:support_features(torch.randn(2,794),torch.ones(2,4),torch.ones(2,3),torch.ones(2,3))
    except ValueError:return
    raise AssertionError('Invalid/oracle-extended evidence dimensions accepted')
