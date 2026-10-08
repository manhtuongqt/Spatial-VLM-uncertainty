import json
import unittest

from workspace.mh_pcrau_v3.g1.validate_schema_v3 import (
    INPUT_SCHEMA_PATH, SCHEMA_PATH, candidate_record, validate_model_input,
    validate_record,
)


class SchemaV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.input_schema = json.loads(INPUT_SCHEMA_PATH.read_text(encoding="utf-8"))

    def fixture(self):
        return {
            "dataset": "D_tabletop_clean_v1", "sample_id": "s", "family_id": "f", "split": "train",
            "rgb_path": "a.jpg", "depth_view_path": "b.png", "metric_depth_path": "",
            "instruction": "leftmost object", "relation_candidate": "leftmost",
            "answerability_candidate": "", "target_uv_candidate": "[0.2, 0.3]",
            "legacy_b1_target_valid": True, "coordinate_raw_valid": False,
            "camera_info_path": "", "tf_snapshot_path": "",
        }

    def test_tabletop_b1_not_promoted(self):
        row = candidate_record(self.fixture())
        self.assertFalse(validate_record(row, self.schema))
        self.assertEqual(row["target_scope"], "B1_LEGACY_ONLY")
        self.assertIsNone(row["target_uv"])
        self.assertFalse(row["training_eligible"])

    def test_gazebo_found_coordinate_candidate_only(self):
        item = self.fixture()
        item.update({"dataset": "Gazebo_train_uq_v2_full_r3", "answerability_candidate": "FOUND",
                     "coordinate_raw_valid": True, "metric_depth_path": "depth.npy"})
        row = candidate_record(item)
        self.assertEqual(row["target_scope"], "V3_2D_CANDIDATE")
        self.assertFalse(validate_record(row, self.schema))
        self.assertFalse(row["training_eligible"])

    def test_invalid_point_and_uncertified_training_rejected(self):
        row = candidate_record(self.fixture())
        row["target_uv"] = [float("nan"), 0.3]
        self.assertTrue(validate_record(row, self.schema))
        row = candidate_record(self.fixture())
        row["training_eligible"] = True
        self.assertTrue(validate_record(row, self.schema))

    def test_model_input_rejects_oracle_fields(self):
        good = {"rgb_path": "x", "depth_path": "d", "instruction": "q"}
        self.assertEqual(validate_model_input(good, self.input_schema), [])
        bad = {**good, "target_uv": [0.2, 0.3]}
        self.assertTrue(validate_model_input(bad, self.input_schema))


if __name__ == "__main__":
    unittest.main()
