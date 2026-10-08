import unittest
import torch

from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.language_augmentation import TrainLanguageAugmentation, validate_prefix
from pcrau.utils import load_config


def test_prefix_keeps_relations_and_cannot_truncate():
    cfg=load_config()['model']
    text='Find the apple nearer than both the box and the cup.'
    assert validate_prefix(text,'Please ',cfg).endswith(text)
    with unittest.TestCase().assertRaises(ValueError):
        validate_prefix(text,'Please ',{**cfg,'max_tokens':3})
    with unittest.TestCase().assertRaises(ValueError):
        validate_prefix('Find the apple.','Find an object left of ',cfg)


def test_train_only_augmentation_preserves_features_and_supervision():
    cfg=load_config();base=ArchivedPCRAUDataset(cfg,'train')
    view=TrainLanguageAugmentation(base)
    assert len(view)==4*len(base)
    a=base[0];b=view[1]
    for name,value in a.items():
        if isinstance(value,torch.Tensor) and name not in {'token_ids','token_mask'}:
            assert torch.equal(value,b[name]),name
    assert a['sample_id']!=b['sample_id'] and a['family_id']==b['family_id']
    with unittest.TestCase().assertRaises(ValueError):
        TrainLanguageAugmentation(ArchivedPCRAUDataset(cfg,'dev'))
