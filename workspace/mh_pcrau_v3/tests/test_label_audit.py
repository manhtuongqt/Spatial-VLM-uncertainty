import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workspace.mh_pcrau_v3.g1.build_label_audit import select_queue, summary
from workspace.mh_pcrau_v3.g1.validate_human_review import validate_rows
from workspace.mh_pcrau_v3.g1.human_review_gui import current_issues, load_progress, write_progress_atomic


class LabelAuditTests(unittest.TestCase):
    def test_head_counts_do_not_certify_missing_labels(self):
        row = {"dataset": "Gazebo_train_uq_v2_full_r3", "split": "train_uq",
               "family_id": "f1", "relation_candidate": "leftmost",
               "answerability_candidate": "FOUND", "camera_info_valid": True,
               "tf_snapshot_valid": True, "legacy_b1_target_valid": False,
               "core_schema_candidate_valid": True}
        for head in ("relation", "answerability", "coordinate", "variance"):
            row[f"{head}_raw_valid"] = True
        for head in ("reasoning", "source", "confidence"):
            row[f"{head}_raw_valid"] = False
        counts = summary([row])["Gazebo_train_uq_v2_full_r3"]["train_uq"]["raw_label_candidates_not_v3_certified"]
        self.assertEqual(counts["coordinate"], 1)
        self.assertEqual(counts["source"], 0)
        self.assertEqual(counts["confidence"], 0)

    def test_review_join_and_completion(self):
        queue = [{"queue_id": "G1-001", "dataset": "A", "sample_id": "s", "family_id": "f",
                  "candidate_relation": "leftmost", "candidate_answerability": "FOUND",
                  "candidate_target_uv": "[0.2,0.3]"}]
        review = [{"queue_id": "G1-001", "dataset": "A", "sample_id": "s", "family_id": "f",
                   "reviewer_id": "r1", "review_relation": "leftmost",
                   "review_answerability": "FOUND", "review_target_correct": "yes",
                   "review_source": "NONE", "review_single_source": "no",
                   "review_status": "DONE"}]
        result = validate_rows(queue, review)
        self.assertFalse(result["issues"])
        self.assertEqual(result["done_count"], 1)
        self.assertEqual(result["relation_candidate_agreement"], {"agree": 1, "compared": 1})
        review[0]["review_source"] = "relation_tie_conflict"
        self.assertTrue(any("single_source=yes" in issue for issue in validate_rows(queue, review)["issues"]))

    def test_named_source_requires_evidence_and_metric_depth(self):
        queue = [{"queue_id": "G1-001", "dataset": "D_tabletop_clean_v1", "sample_id": "s",
                  "family_id": "f", "candidate_relation": "leftmost",
                  "candidate_answerability": "", "candidate_target_uv": "[0.2,0.3]"}]
        review = [{"queue_id": "G1-001", "dataset": "D_tabletop_clean_v1", "sample_id": "s",
                   "family_id": "f", "reviewer_id": "r", "review_relation": "leftmost",
                   "review_answerability": "FOUND", "review_target_correct": "yes",
                   "review_source": "depth_invalid_noisy", "review_single_source": "yes",
                   "review_status": "DONE", "review_notes": ""}]
        issues = validate_rows(queue, review)["issues"]
        self.assertTrue(any("evidence" in issue for issue in issues))
        self.assertTrue(any("relative depth" in issue for issue in issues))

    def test_selection_requires_expected_strata(self):
        with self.assertRaisesRegex(ValueError, "Unexpected audit strata"):
            select_queue([], "seed")

    def test_gui_progress_roundtrip_and_done_guard(self):
        queue = [{"queue_id": "G1-001", "dataset": "A", "sample_id": "s",
                  "family_id": "f", "candidate_target_uv": "[0.2,0.3]"}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            progress = load_progress(path, queue)
            row = progress["G1-001"]
            row["review_status"] = "DONE"
            self.assertIn("reviewer_id", current_issues(row, queue[0]))
            row.update({"reviewer_id": "r1", "review_relation": "leftmost",
                        "review_answerability": "FOUND", "review_target_correct": "yes",
                        "review_source": "NONE", "review_single_source": "no"})
            self.assertFalse(current_issues(row, queue[0]))
            write_progress_atomic(path, queue, progress)
            loaded = load_progress(path, queue)
            self.assertEqual(loaded["G1-001"]["reviewer_id"], "r1")


if __name__ == "__main__":
    unittest.main()
