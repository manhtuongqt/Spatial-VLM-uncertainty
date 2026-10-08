import unittest

from workspace.mh_pcrau_v3.g1.prepare_targeted_review import prepare


class ReviewPreparationTests(unittest.TestCase):
    def test_blind_worklist_excludes_candidates_and_preserves_partial_status(self):
        queue = [{"queue_id": "G1-001", "dataset": "D_tabletop_clean_v1",
                  "sample_id": "s", "family_id": "f", "split": "train",
                  "rgb_path": "rgb.jpg", "depth_view_path": "depth.png",
                  "instruction": "leftmost object", "candidate_relation": "leftmost",
                  "candidate_answerability": "", "candidate_target_uv": "[0.2,0.3]"}]
        saved = [{"queue_id": "G1-001", "dataset": "D_tabletop_clean_v1",
                  "sample_id": "s", "family_id": "f", "review_status": "IN_PROGRESS",
                  "review_answerability": "FOUND", "review_target_correct": "yes"}]
        rows, counts = prepare(queue, saved)
        self.assertEqual(counts["queue_rows"], 1)
        self.assertEqual(counts["prior_target_yes_unverifiable_autopreset_risk"], 1)
        self.assertNotIn("candidate_relation", rows[0])
        self.assertNotIn("candidate_target_uv", rows[0])
        self.assertIn("review_source", rows[0]["missing_review_fields"])


if __name__ == "__main__":
    unittest.main()
