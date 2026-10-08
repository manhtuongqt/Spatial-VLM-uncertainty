from __future__ import annotations

from copy import deepcopy

import torch

from pcrau.answerability_evidence import AnswerabilityEvidenceAdapter, EVIDENCE_NAMES, observable_answerability_evidence
from pcrau.model import PCRAUTargetV2
from pcrau.utils import load_config


def toy_inputs():
    config = deepcopy(load_config())
    config["model"].update(feature_dim=8, hidden_dim=16, text_layers=1, fusion_layers=1,
                           grid_height=3, grid_width=4, max_tokens=8)
    batch = {
        "r0": torch.randn(2, 3, 4, 8), "d0": torch.randn(2, 3, 4, 8),
        "r_thumb": torch.randn(2, 8), "d_thumb": torch.randn(2, 8),
        "token_ids": torch.ones(2, 8, dtype=torch.long), "token_mask": torch.ones(2, 8, dtype=torch.bool),
        "relation_ids": torch.zeros(2, 3, dtype=torch.long), "relation_mask": torch.ones(2, 3, dtype=torch.bool),
        "anchor_mask": torch.zeros(2, 3, dtype=torch.bool),
    }
    return config, batch


def test_evidence_is_finite_without_anchors_and_rejects_annotations():
    config, batch = toy_inputs()
    model = PCRAUTargetV2(config).eval()
    with torch.no_grad():
        output = model(batch)
        evidence = observable_answerability_evidence(output, batch)
    assert evidence.shape == (2, len(EVIDENCE_NAMES))
    assert torch.isfinite(evidence).all()
    for name in ("anchor_sigmoid_max_mean", "anchor_sigmoid_max_min", "edge_probability_mean", "edge_probability_min"):
        assert torch.equal(evidence[:, EVIDENCE_NAMES.index(name)], torch.zeros(2))
    for key in ("target_full", "answer_target", "variant"):
        unsafe = dict(batch)
        unsafe[key] = torch.ones(2)
        try:
            observable_answerability_evidence(output, unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Annotation/metadata allowed into evidence: {key}")


def test_zero_adapter_matches_v2_and_update_preserves_original_outputs():
    config, batch = toy_inputs()
    base = PCRAUTargetV2(config).eval().requires_grad_(False)
    config["model"]["answerability_evidence_adapter"] = {"hidden_dim": 8, "dropout": 0.0}
    candidate = PCRAUTargetV2(config).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(name.startswith("answerability_adapter.") for name in missing)
    candidate.requires_grad_(False)
    candidate.answerability_adapter.requires_grad_(True)
    with torch.no_grad():
        reference = base(batch)
        evidence = observable_answerability_evidence(reference, batch)
    candidate.answerability_adapter.fit_standardization(evidence)
    initial = candidate(batch)
    assert torch.equal(initial["answerability_logits"], reference["answerability_logits"])
    optimizer = torch.optim.AdamW(candidate.answerability_adapter.parameters(), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(initial["answerability_logits"], torch.tensor([0, 2]))
    loss.backward()
    assert all(parameter.grad is None for name, parameter in candidate.named_parameters() if not name.startswith("answerability_adapter."))
    optimizer.step()
    after = candidate(batch)
    assert not torch.equal(after["answerability_logits"], reference["answerability_logits"])
    for name in ("target_logits", "interior_logits", "anchor_logits", "source_logits", "relation_edge_logits"):
        assert torch.equal(after[name], reference[name])
    assert all(torch.equal(value, candidate.state_dict()[name]) for name, value in base.state_dict().items())


def test_standardization_is_stored_and_independent_of_inference_batch():
    adapter = AnswerabilityEvidenceAdapter(8, 0.0).eval()
    train = torch.randn(20, len(EVIDENCE_NAMES))
    adapter.fit_standardization(train)
    with torch.no_grad():
        adapter.net[-1].weight.fill_(0.05)
    before = {name: value.clone() for name, value in adapter.state_dict().items()}
    observation = train[:1]
    single = adapter(observation)
    combined = adapter(torch.cat([observation, train[1:]]))[:1]
    torch.testing.assert_close(single, combined)
    assert all(torch.equal(value, adapter.state_dict()[name]) for name, value in before.items())


def test_context_adapter_uses_only_frozen_global_feature_and_preserves_base():
    config, batch = toy_inputs()
    base = PCRAUTargetV2(config).eval().requires_grad_(False)
    config["model"]["answerability_evidence_adapter"] = {"hidden_dim": 8, "dropout": 0.0, "include_context": True}
    candidate = PCRAUTargetV2(config).eval().requires_grad_(False)
    candidate.load_state_dict(base.state_dict(), strict=False)
    captured = []
    hook = base.answer_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.no_grad():
        output = base(batch)
        joined = torch.cat([observable_answerability_evidence(output, batch), captured[0].float()], dim=-1)
        candidate.answerability_adapter.fit_standardization(joined)
        assert joined.shape[1] == 26 + 6*config["model"]["hidden_dim"]
        assert torch.equal(candidate(batch)["answerability_logits"], output["answerability_logits"])
        candidate.answerability_adapter.net[-1].weight.fill_(0.03)
        expected = output["answerability_logits"] + candidate.answerability_adapter(joined)
        torch.testing.assert_close(candidate(batch)["answerability_logits"], expected)
    hook.remove()
# Detail adapter must retain frozen V2 behavior at initialization.
def test_detail_adapter_zero_preserves_predictions_and_uses_observable_features():
    from copy import deepcopy
    from pcrau.dataset import ArchivedPCRAUDataset, collate_samples, move_model_batch
    from pcrau.model import PCRAUTargetV2
    from pcrau.utils import load_config
    config = load_config()
    config['model']['export_answerability_detail'] = True
    config['model']['export_answerability_evidence'] = True
    data = ArchivedPCRAUDataset(config, 'train')
    inputs = move_model_batch(collate_samples([data[0]]), torch.device('cpu'))
    base = PCRAUTargetV2(config).eval()
    candidate_config = deepcopy(config)
    candidate_config['model']['answerability_evidence_adapter'] = {'hidden_dim':64, 'dropout':.3, 'include_detail':True}
    candidate = PCRAUTargetV2(candidate_config).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(name.startswith('answerability_adapter.') for name in missing)
    with torch.no_grad():
        before, after = base(inputs), candidate(inputs)
    assert before['observable_answerability_detail'].shape == (1,2196)
    assert candidate.answerability_adapter.input_dim == 2222
    for name,value in before.items():
        torch.testing.assert_close(after[name],value,rtol=0,atol=0)
    unsafe={**inputs,'target_full':torch.zeros(1,480,640)}
    try:
        candidate(unsafe)
    except ValueError as error:
        assert 'forbidden' in str(error)
    else:
        raise AssertionError('Detail adapter accepted evaluator input')
