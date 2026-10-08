import unittest

from refspatial_wp1_clean_release import select_disjoint, training_record


class CleanReleaseTests(unittest.TestCase):
    def test_selection_is_deterministic_and_family_disjoint(self):
        pools = {
            "left": [
                {"sample_id": "b", "family_id": "f2"},
                {"sample_id": "a", "family_id": "f1"},
            ],
            "right": [
                {"sample_id": "c", "family_id": "f1"},
                {"sample_id": "d", "family_id": "f3"},
            ],
        }
        selected = select_disjoint(pools, 1, ("left", "right"))
        self.assertEqual([row["sample_id"] for row in selected], ["a", "d"])
        self.assertEqual(len({row["family_id"] for row in selected}), 2)

    def test_selection_fails_closed_on_shortfall(self):
        pools = {"left": [{"sample_id": "a", "family_id": "same"}],
                 "right": [{"sample_id": "b", "family_id": "same"}]}
        with self.assertRaises(ValueError):
            select_disjoint(pools, 1, ("left", "right"))

    def test_training_schema_excludes_privileged_fields(self):
        row = {
            "sample_id": "sample", "image": "path/image.jpg", "depth": "path/depth.png",
            "instruction": "Point to the object.", "answer_original": "[(0.2, 0.3)]",
            "target_xy": [0.2, 0.3], "relation": "leftmost_ranking", "thinking": "oracle",
        }
        output = training_record(row)
        self.assertEqual(set(output), {"image", "depth", "conversations", "source_sample_id"})
        self.assertFalse({"target_xy", "relation", "thinking"} & set(output))
        self.assertEqual(output["image"], "image.jpg")
        self.assertEqual(output["depth"], "depth.png")


if __name__ == "__main__":
    unittest.main()
