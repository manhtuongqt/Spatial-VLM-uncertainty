import unittest

from workspace.mh_pcrau_v3.g1.audit_development import map_relation, split_overlap, valid_point


class G1AuditTests(unittest.TestCase):
    def test_relation_mapping_is_prompt_based(self):
        self.assertEqual(map_relation("leftmost_ranking", "leftmost cup"), "leftmost")
        self.assertEqual(
            map_relation("horizontal_ordinal_ranking", "Point to the second cup from right to left."),
            "second_from_right",
        )
        self.assertIsNone(map_relation("horizontal_ordinal_ranking", "Point to the third cup from right to left."))

    def test_family_overlap_and_target_range(self):
        rows = [
            {"split": "train", "family_id": "a"},
            {"split": "val", "family_id": "b"},
            {"split": "val", "family_id": "a"},
        ]
        self.assertEqual(split_overlap(rows), {"train|val": 1})
        self.assertTrue(valid_point([0.1, 1.0]))
        self.assertFalse(valid_point([1.1, 0.2]))
        self.assertFalse(valid_point(None))


if __name__ == "__main__":
    unittest.main()
