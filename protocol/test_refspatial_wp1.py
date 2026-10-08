import unittest
from refspatial_manual_review import summarize, wilson
from refspatial_point_evaluator import evaluate
from refspatial_wp1_rules import (QUESTION_TYPES, anchor_candidates, corrected_categories,
                                  fruit_evidence, point, relation_tags, semantic_review_requirements, target_kind)


class RulesTests(unittest.TestCase):
    def test_anchor_rank_is_not_direct_target_supervision(self):
        result = semantic_review_requirements('Closest cup to the leftmost cup', 'object', 'object_distance_from_anchor')
        self.assertIsNone(result['direct_relation'])
        self.assertIn('anchor_distance_metric_unverified', result['blockers'])
        self.assertFalse(result['training_eligible'])
        direct = semantic_review_requirements('Select leftmost cup', 'object', 'object_features_ranking')
        self.assertEqual(direct['direct_relation'], 'leftmost_ranking')

    def test_malformed_rank_and_proxy_depth_fail_closed(self):
        for query, reason in [('Third cup fromt to left', 'malformed_horizontal_direction'),
                              ('Which object ranks in proximity to the cup', 'missing_distance_rank'),
                              ('Select nearest cup', 'distance_reference_frame_unverified')]:
            result = semantic_review_requirements(query, 'object', 'object_features_ranking')
            self.assertIn(reason, result['blockers'])
            self.assertIsNone(result['direct_relation'])

    def test_ai_provenance_and_unsure_denominator(self):
        q = [{'sample_id': 'a', 'relation_candidates': [], 'anchor_candidates_offline': [], 'audit_strata': ['object_grounding_general']}]
        r = {'sample_id': 'a', 'decision': 'unsure', 'reviewer': 'AI', 'review_origin': 'ai_visual',
             'reviewed_at': '2026-09-09', 'target_correct': 'unsure', 'tabletop_appropriate': 'yes',
             'relation_correct': 'na', 'anchor_correct': 'na'}
        result = summarize(q, [r])
        self.assertEqual(result['precision']['target_correct']['n'], 1)
        self.assertEqual(result['precision']['target_correct']['precision'], 0)
        self.assertFalse(result['independent_human_review_complete'])
        self.assertEqual(result['review_origins'], {'ai_visual': 1})
        r.update(decision='accept', target_correct='yes', reference_frame='unknown', reasoning_depth='0')
        self.assertTrue(summarize(q, [r])['validation_errors'])

    def test_placement_is_structured(self):
        for raw, canonical in QUESTION_TYPES.items():
            expected = 'object' if canonical.startswith('object_') else 'free_space'
            self.assertEqual(target_kind(raw)[0], expected)
        self.assertEqual(target_kind('unrecognised')[0], 'unknown')
        self.assertEqual(relation_tags('Point between the apple and cup', 'free_space'), [])

    def test_point_rejects_malformed_and_boolean(self):
        for bad in ['[(True, 0)]', '[(1.01,0)]', '[(-.1,.2)]', '[0, 1]', '[(0,1),(1,0)]', '[(nan,0)]', 'bad']:
            self.assertIsNone(point(bad), bad)
        self.assertEqual(point('[(0.2, 1)]'), (0.2, 1.0))

    def test_orange_disambiguation(self):
        for label in ['the biggest orange Dixie cup', 'red-orange can', 'fruit juice', 'orange juice carton']:
            self.assertEqual(fruit_evidence(label)[0], 'nonfruit', label)
        for label in ['the orange', 'the biggest orange', 'the apple', 'orange slice']:
            self.assertEqual(fruit_evidence(label)[0], 'fruit', label)
        self.assertEqual(fruit_evidence('cup beside the orange')[0], 'unresolved')
        self.assertEqual(corrected_categories(['orange dixie cup', 'third object'], ['fruit', 'container'])[0], ['container'])
        self.assertIn('fruit', corrected_categories(['orange', 'orange cup'], ['fruit'])[0])

    def test_ranking_is_not_pairwise(self):
        self.assertEqual(relation_tags('Select leftmost cup. Your answer should be between 0 and 1.', 'object'), ['leftmost_ranking'])
        self.assertEqual(relation_tags('Third object from left to right', 'object'), ['horizontal_ordinal_ranking'])
        self.assertEqual(relation_tags('Cup left of bottle', 'object'), ['left_pairwise'])

    def test_anchor_is_offline_candidate_not_target(self):
        rationale = 'Step 1: [Position] [the cup]: [(0.2, 0.3)]'
        self.assertEqual(len(anchor_candidates('nearest to the cup', rationale, (.5, .5))), 1)
        self.assertEqual(anchor_candidates('nearest to the cupcake', rationale, (.5, .5)), [])
        self.assertEqual(anchor_candidates('nearest to the cup', rationale, (.2, .3)), [])

    def test_empty_reviews_never_pass(self):
        q = [{'sample_id': 'a', 'relation_candidates': [], 'anchor_candidates_offline': [], 'audit_strata': ['general']}]
        result = summarize(q, [{'sample_id': 'a', 'decision': ''}])
        self.assertEqual(result['completed_reviews'], 0)
        self.assertFalse(result['training_eligible'])
        self.assertIsNone(result['precision']['target_correct']['precision'])
        self.assertIsNone(wilson(0, 0))

    def test_missing_prediction_is_failure_and_unknown_id_rejected(self):
        targets = [{'sample_id': s, 'family_id': s, 'split': 'dev', 'target_xy': [.5, .5]} for s in ['a', 'b']]
        result = evaluate(targets, [{'sample_id': 'a', 'prediction_xy': [.5, .5]}], .01)
        self.assertEqual(result['point_hit_all_samples'], .5)
        self.assertEqual(result['missing_predictions'], 1)
        with self.assertRaises(ValueError):
            evaluate(targets, [{'sample_id': 'unknown', 'prediction_xy': [.5, .5]}], .01)
        with self.assertRaises(ValueError):
            evaluate([{**targets[0], 'split': 'test_iid'}], [], .01)


if __name__ == '__main__':
    unittest.main()
