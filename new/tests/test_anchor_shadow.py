"""Synthetic contract checks; no archive, checkpoint, optimizer steps or test data."""
import torch

from pcrau.anchor_shadow import AnchorShadow
from pcrau.anchor_shadow_experiment import PROTOCOL, ShadowPilotRunner, anchor_shadow_loss, family_batches
from pcrau.model import PCRAUTargetV2
from pcrau.text import tokenize, relation_ids, prompt_anchor_mask


def setup():
    cfg = {"model": {"hidden_dim": 128, "feature_dim": 8, "max_tokens": 72, "vocab_size": 8192,
        "max_anchors": 3, "max_relations": 3, "attention_heads": 4, "text_layers": 1,
        "fusion_layers": 1, "dropout": 0.1, "grid_height": 2, "grid_width": 3,
        "relation_classes": ["direct", "left_of", "right_of", "front_of", "behind", "nearer_than", "farther_than", "between_in_depth", "nearer_than_both"],
        "answerability_classes": ["FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE"],
        "source_classes": ["semantic", "relation", "spatial", "depth", "occlusion"],
        "export_answerability_detail": True, "export_answerability_evidence": True,
        "answerability_evidence_adapter": {"include_detail": True, "hidden_dim": 64, "dropout": .3}}}
    torch.manual_seed(123)
    model = PCRAUTargetV2(cfg).eval()
    prompts = ["locate the apple that is right of the purple cube.", "locate the apple.", "locate the apple that is behind the cube."]
    batch = {"r0": torch.randn(3, 2, 3, 8), "d0": torch.randn(3, 2, 3, 8),
             "r_thumb": torch.randn(3, 8), "d_thumb": torch.randn(3, 8)}
    fields = {key: [] for key in ("token_ids", "token_mask", "relation_ids", "relation_mask", "anchor_mask")}
    for p in prompts:
        ids, mask = tokenize(p, 72, 8192); rid, rm = relation_ids(p, 3)
        for key, value in zip(fields, (ids, mask, rid, rm, prompt_anchor_mask(p, 3))):
            fields[key].append(value)
    batch.update({k: torch.tensor(v, dtype=torch.long if k.endswith("ids") else torch.bool) for k, v in fields.items()})
    return model, batch, prompts


def test_zero_initialization_and_exact_baseline_outputs():
    model, batch, prompts = setup()
    m1, m2 = AnchorShadow(model, "phrase"), AnchorShadow(model, "whole_text")
    assert sum(p.numel() for p in m1.parameters() if p.requires_grad) == 33024
    assert all(torch.equal(a, b) for a, b in zip(m1.residual.parameters(), m2.residual.parameters()))
    with torch.no_grad():
        plain = model(batch); cap = m1.capture(batch, prompts)
        a, b = m1.predict(cap), m2.predict(cap)
    assert all(torch.equal(v, cap.baseline[k]) for k, v in plain.items())
    torch.testing.assert_close(a, plain["anchor_logits"], rtol=0, atol=1e-6)
    assert torch.equal(a, b)
    assert cap.scope_status == ("SUPPORTED_SHADOW", "DIRECT_BYPASS", "PARSE_UNSUPPORTED")
    m1.train(); assert m1.residual.training and not any(m.training for m in model.modules())


def test_backward_only_reaches_residual_and_w1_after_zero_init():
    model, batch, prompts = setup(); branch = AnchorShadow(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cap = branch.capture(batch, prompts)
    target = torch.zeros_like(cap.baseline["anchor_logits"]); target[0, 0, 0, 0] = 1
    loss = anchor_shadow_loss(branch.predict(cap), target, cap.active_slots)["total"]
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert branch.residual[2].weight.grad.abs().sum() > 0
    assert branch.residual[0].weight.grad.abs().sum() == 0  # Expected with zero W2.
    branch.zero_grad(set_to_none=True)
    with torch.no_grad(): branch.residual[2].weight.fill_(0.001)
    loss = anchor_shadow_loss(branch.predict(cap), target, cap.active_slots)["total"]
    loss.backward()
    assert branch.residual[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in branch.residual.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_nonzero_branch_changes_active_only_and_preserves_downstream():
    model, batch, prompts = setup(); branch = AnchorShadow(model)
    with torch.no_grad(): branch.residual[2].bias.fill_(0.3)
    out = branch(batch, prompts); new = out["anchor_logits_shadow"]; old = out["baseline"]["anchor_logits"]
    assert not torch.equal(new[0, 0], old[0, 0])
    assert torch.equal(new[1:], old[1:]) and torch.equal(new[:, 1:], old[:, 1:])
    with torch.no_grad(): plain = model(batch)
    assert all(torch.equal(plain[k], v) for k, v in out["baseline"].items())


def test_oracle_and_mismatched_prompt_rejected():
    model, batch, prompts = setup(); branch = AnchorShadow(model)
    bad = dict(batch, target_heatmap=torch.zeros(3, 2, 3))
    for inputs, text in [(bad, prompts), (batch, [prompts[1], prompts[0], prompts[2]])]:
        try: branch(inputs, text)
        except ValueError: pass
        else: raise AssertionError("Forbidden or misbound input was accepted")


def test_capture_cleanup_on_failure_and_ordinary_tensors_under_inference_mode():
    model, batch, prompts = setup(); branch = AnchorShadow(model)
    def fail(*args): raise RuntimeError("Injected fusion hook failure")
    handle = model.fusion.register_forward_hook(fail)
    counts = {k: len(m._forward_hooks) for k, m in model.named_modules()}
    try: branch.capture(batch, prompts)
    except RuntimeError as e: assert "Injected" in str(e)
    else: raise AssertionError("Expected injected exception")
    assert counts == {k: len(m._forward_hooks) for k, m in model.named_modules()}
    assert not branch._capturing
    handle.remove()
    with torch.inference_mode(): cap = branch.capture(batch, prompts)
    assert not cap.text.is_inference() and not cap.fused.is_inference() and not cap.active_slots.is_inference()
    branch.predict(cap)[0, 0].sum().backward()
    assert branch.residual[2].weight.grad.abs().sum() > 0


def test_empty_mask_loss_and_inactive_slots():
    logits = torch.zeros(3, 3, 2, 3, requires_grad=True); targets = torch.zeros_like(logits)
    targets[0, 0, 0, 0] = .5; targets[2, 1] = 1  # The latter slot is inactive.
    active = torch.zeros(3, 3, dtype=torch.bool); active[0, 0] = active[1, 0] = True
    losses = anchor_shadow_loss(logits, targets, active)
    assert losses["visible_count"] == losses["empty_count"] == 1
    torch.testing.assert_close(losses["empty_bce"], torch.tensor(2.).log())
    losses["total"].backward()
    assert (logits.grad[1, 0] > 0).all() and logits.grad[0, 0, 0, 0] < 0
    assert logits.grad[2].abs().sum() == 0 and logits.grad[:, 1:].abs().sum() == 0
    empty = anchor_shadow_loss(logits, targets, torch.zeros_like(active))
    assert empty["total"].item() == 0 and torch.isfinite(empty["total"])


def test_family_batch_schedule_and_optimizer_gate():
    presentations = [{"family_id": str(f), "entry": {"split": "train"}} for f in range(8) for _ in range(16)]
    a = family_batches(presentations, 0); b = family_batches(presentations, 0)
    assert a == b and len(a) == 2 and all(len(batch) == 64 for batch in a)
    assert sorted(i for batch in a for i in batch) == list(range(128))
    assert not {presentations[i]["family_id"] for i in a[0]} & {presentations[i]["family_id"] for i in a[1]}
    for bad in [-1, PROTOCOL.max_epochs]:
        try: family_batches(presentations, bad)
        except ValueError: pass
        else: raise AssertionError("Budget exceeded")
    model, _, _ = setup(); runner = ShadowPilotRunner(AnchorShadow(model))
    assert runner.optimizer is None and runner.optimizer_steps == 0
    try: runner.start_optimizer({})
    except ValueError: pass
    else: raise AssertionError("Failed preflight accepted")


def test_capture_preserves_amp_head_mode_for_later_prediction():
    model, batch, prompts = setup(); branch = AnchorShadow(model)
    with torch.autocast("cpu", dtype=torch.bfloat16): cap = branch.capture(batch, prompts)
    assert cap.head_autocast_dtype == torch.bfloat16
    with torch.no_grad(): new = branch.predict(cap)
    assert torch.equal(new, cap.baseline["anchor_logits"])


def test_shadow_head_reuses_sliced_query_stride():
    model, batch, prompts = setup(); branch = AnchorShadow(model)
    cap = branch.capture(batch, prompts)
    assert cap.old_queries.stride()[0] == 7 * 128
    seen = []
    handle = model.anchor_head.register_forward_pre_hook(lambda m, args: seen.append(args[1].stride()))
    try: branch.predict(cap)[0, 0].sum().backward()
    finally: handle.remove()
    assert seen == [cap.old_queries.stride()]
    assert branch.residual[2].weight.grad.abs().sum() > 0


def test_runner_dev_selection_ties_and_pending_epoch():
    model, _, _ = setup(); runner = ShadowPilotRunner(AnchorShadow(model))
    for epoch in range(6):
        # Simulate epoch bookkeeping only, no optimizer creation/update.
        runner.next_epoch = epoch + 1; runner.pending_selection_epoch = epoch
        improved = runner.record_dev_selection(epoch, 44, 6, .7)
        assert improved == (epoch == 0)
        assert runner.best_epoch == 0 and runner.stale_epochs == epoch
        try: runner.record_dev_selection(epoch, 44, 6, .7)
        except ValueError: pass
        else: raise AssertionError("Duplicate selection accepted")
    assert runner.stale_epochs == PROTOCOL.patience
    runner.next_epoch = 7; runner.pending_selection_epoch = 6
    try: runner.record_dev_selection(6, 44, 6, float('nan'))
    except ValueError: pass
    else: raise AssertionError("Nonfinite dev metric accepted")
    assert runner.pending_selection_epoch == 6
    try: runner.start_optimizer({"status": "S1_SHADOW_PREFLIGHT_PASS", "technical_gates_pass": True})
    except ValueError: pass
    else: raise AssertionError("Unmatched preflight receipt accepted")
    assert runner.optimizer is None
