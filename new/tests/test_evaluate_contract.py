from __future__ import annotations

from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.losses import class_weights
from pcrau.utils import load_config


def test_evaluation_loss_weights_are_train_locked() -> None:
    config = load_config()
    train = ArchivedPCRAUDataset(config, "train")
    dev = ArchivedPCRAUDataset(config, "dev")
    train_answer, train_source = class_weights(train, device="cpu")
    dev_answer, dev_source = class_weights(dev, device="cpu")
    assert not train_answer.equal(dev_answer) or not train_source.equal(dev_source)
