import unittest
import torch

from pcrau.acceptance_ranker import AcceptanceRanker, acceptance_loss


def test_pairwise_loss_prefers_bad_above_good():
    labels = torch.tensor([0., 1.])
    correct = torch.tensor([-2., 2.], requires_grad=True)
    reversed_scores = torch.tensor([2., -2.])
    assert acceptance_loss(correct, labels, .1) < acceptance_loss(reversed_scores, labels, .1)
    acceptance_loss(correct, labels, .1).backward()
    assert correct.grad[0] > 0 and correct.grad[1] < 0


def test_single_class_batches_have_finite_gradients():
    for value in (0., 1.):
        scores = torch.zeros(3, requires_grad=True)
        loss = acceptance_loss(scores, torch.full((3,), value), .1)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()


def test_standardization_serialization_and_invalid_inputs():
    torch.manual_seed(3)
    head = AcceptanceRanker(4).eval()
    train = torch.randn(20, 4)
    head.fit_standardization(train)
    torch.testing.assert_close(head.feature_mean, train.mean(0))
    copy = AcceptanceRanker(4).eval()
    copy.load_state_dict(head.state_dict())
    torch.testing.assert_close(copy(train), head(train))
    with unittest.TestCase().assertRaises(ValueError):
        head.fit_standardization(torch.empty(0, 4))
    with unittest.TestCase().assertRaises(ValueError):
        head.fit_standardization(torch.full((2, 4), float('nan')))


def test_zero_residual_preserves_base_scores():
    head = AcceptanceRanker(4, zero_output=True).eval()
    features = torch.randn(10, 4)
    base = torch.randn(10)
    assert torch.equal(head(features), torch.zeros(10))
    torch.testing.assert_close((base+head(features)).sigmoid(), base.sigmoid(), rtol=0, atol=0)


def test_wrapper_preserves_all_v2_outputs_and_oracle_boundary():
    from pcrau.acceptance_ranker import FrozenAcceptanceScorer
    from pcrau.dataset import ArchivedPCRAUDataset, collate_samples, move_model_batch
    from pcrau.model import PCRAUTargetV2
    from pcrau.utils import load_config
    config = load_config()
    data = ArchivedPCRAUDataset(config, 'train')
    batch = collate_samples([data[0]])
    inputs = move_model_batch(batch, torch.device('cpu'))
    base = PCRAUTargetV2(config).eval()
    ranker = AcceptanceRanker(26+config['model']['hidden_dim']*6, zero_output=True)
    with torch.no_grad():
        original = base(inputs)
        wrapper = FrozenAcceptanceScorer(base, ranker, 'ranker-hash', 'v2-hash')
        scored = wrapper(inputs)
    for name, tensor in original.items():
        torch.testing.assert_close(scored[name], tensor, rtol=0, atol=0)
    expected = torch.logsumexp(original['answerability_logits'].float()[:,1:],dim=-1)-original['answerability_logits'].float()[:,0]
    torch.testing.assert_close(scored['ranker_error_logit'], expected, rtol=0, atol=0)
    assert len(base.answer_head._forward_pre_hooks) == 0
    unsafe = {**inputs, 'target_full': batch['target_full']}
    with unittest.TestCase().assertRaises(ValueError):
        wrapper(unsafe)
    assert len(base.answer_head._forward_pre_hooks) == 0


def test_load_wrapper_rejects_different_v2_hash():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from pcrau.acceptance_ranker import load_acceptance_scorer
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'mismatch.pt'
        torch.save({'residual':True,'source_v2_sha256':'other-v2'},path)
        with unittest.TestCase().assertRaises(ValueError):
            load_acceptance_scorer(torch.nn.Linear(1,1),path,'expected-v2')
