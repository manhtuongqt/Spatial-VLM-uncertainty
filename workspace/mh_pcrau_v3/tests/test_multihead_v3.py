import io
import json
from pathlib import Path
import unittest

import jsonschema
import torch

from workspace.mh_pcrau_v3.loss_v3 import (
    LossWeights, SupervisionBatch, compute_multitask_loss,
)
from workspace.mh_pcrau_v3.multihead_v3 import (
    MHMultiHeadConfig, build_seeded_model, to_development_records,
)


def targets(batch, *, all_masked=False):
    off = torch.zeros(batch, dtype=torch.bool)
    on = torch.ones(batch, dtype=torch.bool)
    answerability = torch.arange(batch, dtype=torch.long) % 4
    spatial = answerability == 0
    return SupervisionBatch(
        relation=torch.arange(batch, dtype=torch.long) % 4,
        relation_mask=off.clone() if all_masked else on.clone(),
        reasoning_depth=torch.arange(batch, dtype=torch.long) % 3,
        reasoning_mask=off.clone(),
        target_uv=torch.full((batch, 2), 0.25),
        spatial_mask=off.clone() if all_masked else spatial,
        uncertainty_source=torch.zeros(batch, dtype=torch.long),
        source_mask=off.clone(),
        answerability=answerability,
        answerability_mask=off.clone() if all_masked else on.clone(),
    )


class HeadTests(unittest.TestCase):
    def test_head_b1_shapes_ranges_and_finite(self):
        model = build_seeded_model(24092026).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1536))
        shapes = {
            "relation_logits": (1, 4), "reasoning_logits": (1, 3),
            "mu_uv": (1, 2), "log_variance_uv": (1, 2),
            "source_logits": (1, 5), "answerability_logits": (1, 4),
            "confidence_logit": (1,), "z_spatial": (1, 512),
        }
        for name, shape in shapes.items():
            value = getattr(out, name)
            self.assertEqual(tuple(value.shape), shape)
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(((out.mu_uv >= 0) & (out.mu_uv <= 1)).all())
        self.assertTrue(((out.log_variance_uv >= -8) & (out.log_variance_uv <= 2)).all())
        self.assertTrue(((out.raw_safe_score >= 0) & (out.raw_safe_score <= 1)).all())

    def test_head_mixed_batch_probabilities(self):
        model = build_seeded_model(24092026).eval()
        with torch.no_grad():
            out = model(torch.randn(7, 1536))
        for value in (out.relation_probabilities, out.reasoning_probabilities,
                      out.source_probabilities, out.answerability_probabilities):
            torch.testing.assert_close(value.sum(-1), torch.ones(7))

    def test_head_input_rejection(self):
        model = build_seeded_model(1)
        for bad in (torch.zeros(2, 2), torch.zeros(0, 1536),
                    torch.full((1, 1536), float("nan")), torch.ones(1, 1536, dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(bad)

    def test_head_architecture_drift_rejected(self):
        with self.assertRaises(ValueError):
            MHMultiHeadConfig(trunk_size=256).validate()

    def test_head_parameter_inventory_and_stage_freeze(self):
        model = build_seeded_model(1)
        total = sum(p.numel() for p in model.parameters())
        self.assertEqual(total, 1_062_421)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1_061_908)
        state = model.configure_trainable("s1b")
        self.assertEqual(state, {"total": 1_062_421, "trainable": 513})
        self.assertTrue(all(not p.requires_grad for p in model.shared_trunk.parameters()))

    def test_head_seed_does_not_mutate_caller_rng(self):
        torch.manual_seed(12)
        state = torch.random.get_rng_state().clone()
        first = build_seeded_model(99)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second = build_seeded_model(99)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_serialization_round_trip_reproduces_output(self):
        model = build_seeded_model(24092026).eval()
        h = torch.randn(4, 1536)
        expected = model(h).mu_uv.detach()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_seeded_model(0).eval()
        restored.load_state_dict(torch.load(stream, map_location="cpu"))
        torch.testing.assert_close(restored(h).mu_uv, expected, rtol=0, atol=0)

    def test_head_development_record_has_no_oracle_and_schema_valid(self):
        model = build_seeded_model(4).eval()
        record = to_development_records(
            model(torch.zeros(1, 1536)), ["sample-1"], model_id="smoke",
            source_hash="a" * 64,
        )[0]
        forbidden = {"target_uv", "answerability_state", "ground_truth", "family_id"}
        self.assertTrue(forbidden.isdisjoint(record))
        schema = json.loads(Path(
            "ketqua1/05_dau_ra_tong_hop/schema/inference_record.schema.json"
        ).read_text(encoding="utf-8"))
        jsonschema.validate(record, schema)


class LossTests(unittest.TestCase):
    def test_loss_mixed_batch_counts_finite_and_backward(self):
        model = build_seeded_model(2)
        out = model(torch.randn(8, 1536))
        result = compute_multitask_loss(out, targets(8))
        self.assertEqual(result.valid_counts, {
            "relation": 8, "reasoning": 0, "spatial": 2, "source": 0,
            "answerability": 8, "confidence": 0, "language_modeling": 0,
        })
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.confidence_head.parameters()))

    def test_loss_all_masked_is_connected_zero_not_nan(self):
        model = build_seeded_model(3)
        result = compute_multitask_loss(model(torch.randn(5, 1536)), targets(5, all_masked=True))
        self.assertEqual(float(result.total.detach()), 0.0)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()

    def test_loss_denominator_is_valid_count_not_batch(self):
        model = build_seeded_model(5).eval()
        h = torch.randn(2, 1536)
        out = model(h)
        t = targets(2)
        t.relation_mask[:] = torch.tensor([True, False])
        one = compute_multitask_loss(out, t).components["relation"]
        expected = torch.nn.functional.cross_entropy(out.relation_logits[:1], t.relation[:1])
        torch.testing.assert_close(one, expected)

    def test_loss_nonfound_has_zero_coordinate_variance_gradient(self):
        model = build_seeded_model(6)
        out = model(torch.randn(4, 1536))
        out.mu_uv.retain_grad()
        out.log_variance_uv.retain_grad()
        t = targets(4)
        compute_multitask_loss(out, t).total.backward()
        self.assertGreater(float(out.mu_uv.grad[0].abs().sum()), 0)
        self.assertGreater(float(out.log_variance_uv.grad[0].abs().sum()), 0)
        self.assertEqual(float(out.mu_uv.grad[1:].abs().sum()), 0.0)
        self.assertEqual(float(out.log_variance_uv.grad[1:].abs().sum()), 0.0)

    def test_loss_spatial_requires_found(self):
        model = build_seeded_model(7)
        t = targets(2)
        t.spatial_mask[:] = torch.tensor([False, True])
        with self.assertRaises(ValueError):
            compute_multitask_loss(model(torch.randn(2, 1536)), t)

    def test_loss_invalid_active_label_rejected_but_masked_sentinel_allowed(self):
        model = build_seeded_model(8)
        t = targets(3)
        t.reasoning_depth[:] = -1
        compute_multitask_loss(model(torch.randn(3, 1536)), t)
        t.reasoning_mask[0] = True
        with self.assertRaises(ValueError):
            compute_multitask_loss(model(torch.randn(3, 1536)), t)

    def test_loss_s1a_forbids_confidence_and_lm(self):
        model = build_seeded_model(9)
        out, t = model(torch.randn(2, 1536)), targets(2)
        with self.assertRaises(ValueError):
            compute_multitask_loss(out, t, LossWeights(confidence=1.0))
        with self.assertRaises(ValueError):
            compute_multitask_loss(out, t, LossWeights(language_modeling=1.0))

    def test_loss_exact_gaussian_nll_formula(self):
        model = build_seeded_model(10).eval()
        out = model(torch.randn(1, 1536))
        t = targets(1)
        observed = compute_multitask_loss(out, t).components["spatial"]
        delta2 = (t.target_uv - out.mu_uv).square()
        expected = 0.5 * (torch.exp(-out.log_variance_uv) * delta2 + out.log_variance_uv).sum()
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
