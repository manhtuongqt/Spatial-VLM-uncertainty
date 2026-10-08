import unittest

from workspace.mh_pcrau_v3.g1.extract_tabletop_qa import extract_rows, parse_single_point


def fixture():
    question = "What is the 2D location of the leftmost object?"
    answer = "[(0.197, 0.820)]"
    source = {
        "sample_id": "s1", "family_id": "f1", "split": "train",
        "image": "data/image/a.jpg", "depth": "data/depth/a.png",
        "instruction": question, "answer_original": answer,
        "target_xy": [0.197, 0.82], "relation": "leftmost_ranking",
    }
    qa = {
        "source_sample_id": "s1", "image": "a.jpg", "depth": "a.png",
        "conversations": [{"from": "human", "value": question},
                          {"from": "gpt", "value": answer}],
    }
    return source, qa


class TabletopQaExtractTests(unittest.TestCase):
    def test_exact_qa_is_b1_only(self):
        source, qa = fixture()
        rows, counts = extract_rows([source], [qa])
        self.assertEqual(counts["qa_rows_verified"], 1)
        self.assertEqual(rows[0]["target_uv_b1"], [0.197, 0.82])
        self.assertEqual(rows[0]["relation_v3_candidate"], "leftmost")
        self.assertIsNone(rows[0]["answerability_v3"])
        self.assertIsNone(rows[0]["reasoning_depth_v3"])
        self.assertIsNone(rows[0]["primary_uncertainty_source_v3"])
        self.assertFalse(rows[0]["v3_training_eligible"])

    def test_answer_mismatch_fails_closed(self):
        source, qa = fixture()
        qa["conversations"][1]["value"] = "[(0.198, 0.820)]"
        with self.assertRaisesRegex(ValueError, "QA differs"):
            extract_rows([source], [qa])

    def test_train_id_mismatch_fails_closed(self):
        source, qa = fixture()
        qa["source_sample_id"] = "other"
        with self.assertRaisesRegex(ValueError, "sample IDs differ"):
            extract_rows([source], [qa])

    def test_relation_question_mismatch_fails_closed(self):
        source, qa = fixture()
        source["relation"] = "rightmost_ranking"
        with self.assertRaisesRegex(ValueError, "relation disagrees"):
            extract_rows([source], [qa])

    def test_non_point_or_unsafe_expression_fails(self):
        for answer in ("[]", "[(2, 0.5)]", "[(float('nan'), 0.5)]", "__import__('os').system('true')"):
            with self.subTest(answer=answer):
                with self.assertRaises(ValueError):
                    parse_single_point(answer)


if __name__ == "__main__":
    unittest.main()
