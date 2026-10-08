#!/usr/bin/env python3
"""Stdlib unittest for the atomic checkpoint manager (run in RoboRefer env)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training_checkpoint_manager import (
    CheckpointError,
    audit_checkpoint_root,
    resume_training,
    save_checkpoint,
    verify_checkpoint,
)


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        self.temporary = tempfile.TemporaryDirectory(prefix="pcra-checkpoint-test-")
        self.root = Path(self.temporary.name) / "checkpoints"
        self.identity = {
            "config_sha256": "1" * 64,
            "dataset_sha256": "2" * 64,
            "split_manifest_sha256": "3" * 64,
            "code_sha256": "4" * 64,
            "feature_contract_sha256": "5" * 64,
        }
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.GELU(), torch.nn.Linear(7, 2))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        loss = self.model(torch.ones(3, 4)).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def save(self, step: int, value: float):
        return save_checkpoint(
            self.root,
            run_id="unit_test_run",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=1,
            global_step=step,
            selection_split="dev",
            selection_metric="dev_loss",
            selection_mode="min",
            selection_value=value,
            metrics={"dev_loss": value, "train_loss": value / 2},
            identity=self.identity,
            sampler_state={"epoch": 1, "offset": step},
            extra_state={"gradient_accumulation_offset": 0},
            runtime={"torch": torch.__version__, "device": "cpu"},
        )

    def test_atomic_save_best_last_and_exact_resume(self) -> None:
        first = self.save(10, 0.7)
        self.assertTrue(first["verification_passed"])
        self.assertTrue(first["promoted_best"])
        expected = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        expected_scheduler_epoch = self.scheduler.last_epoch
        expected_python = random.random()
        expected_numpy = float(np.random.rand())
        expected_torch = float(torch.rand(()))
        random.random()
        np.random.rand()
        torch.rand(())
        for parameter in self.model.parameters():
            parameter.data.add_(100)
        resumed = resume_training(
            self.root,
            pointer="last",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_identity=self.identity,
        )
        self.assertEqual(resumed["metadata"]["global_step"], 10)
        self.assertEqual(resumed["sampler_state"], {"epoch": 1, "offset": 10})
        self.assertEqual(self.scheduler.last_epoch, expected_scheduler_epoch)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.rand()), expected_numpy)
        self.assertEqual(float(torch.rand(())), expected_torch)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)

        second = self.save(20, 0.9)
        self.assertFalse(second["promoted_best"])
        self.assertEqual(json.loads((self.root / "last.json").read_text())["checkpoint_id"], "step_000000020")
        self.assertEqual(json.loads((self.root / "best_dev_loss.json").read_text())["checkpoint_id"], "step_000000010")
        third = self.save(30, 0.5)
        self.assertTrue(third["promoted_best"])
        self.assertEqual(json.loads((self.root / "best_dev_loss.json").read_text())["checkpoint_id"], "step_000000030")
        self.assertTrue(audit_checkpoint_root(self.root)["passed"])
        self.assertEqual(list(self.root.glob(".tmp-*")), [])

    def test_forbidden_split_identity_and_nonfinite_guard(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "forbidden"):
            save_checkpoint(
                self.root, run_id="unit_test_run", model=self.model, optimizer=self.optimizer,
                epoch=0, global_step=1, selection_split="calibration", selection_metric="nll",
                selection_mode="min", selection_value=1.0, metrics={"nll": 1.0}, identity=self.identity,
            )
        self.save(10, 0.7)
        wrong_identity = dict(self.identity)
        wrong_identity["dataset_sha256"] = "9" * 64
        with self.assertRaisesRegex(CheckpointError, "identity mismatch"):
            resume_training(self.root, pointer="last", model=self.model, optimizer=self.optimizer,
                            scheduler=self.scheduler, expected_identity=wrong_identity)
        next(self.model.parameters()).data[0, 0] = float("nan")
        with self.assertRaisesRegex(CheckpointError, "non-finite"):
            self.save(20, 0.6)

    def test_manifest_tamper_is_rejected(self) -> None:
        result = self.save(10, 0.7)
        checkpoint = Path(result["checkpoint_dir"])
        with (checkpoint / "metadata.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(CheckpointError, "verification failed"):
            verify_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
