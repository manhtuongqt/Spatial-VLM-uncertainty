"""Train-only prompt prefixes preserving the original referent and relations."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .text import words, tokenize, relation_ids, prompt_anchor_mask

PREFIXES = ('', 'Please ', 'In this scene, ', 'Can you ')


def augmented_id(sample_id, variant):
    return sample_id if variant == 0 else f'{sample_id}__language_prefix_{variant}'


def validate_prefix(prompt, prefix, config):
    augmented = prefix+prompt
    if len(words(augmented)) > int(config['max_tokens']):
        raise ValueError('Prefix would truncate original command')
    original_words = words(prompt)
    if words(augmented)[-len(original_words):] != original_words:
        raise ValueError('Original command changed')
    if relation_ids(augmented,int(config['max_relations'])) != relation_ids(prompt,int(config['max_relations'])):
        raise ValueError('Relation syntax changed')
    if prompt_anchor_mask(augmented,int(config['max_anchors'])) != prompt_anchor_mask(prompt,int(config['max_anchors'])):
        raise ValueError('Anchor syntax changed')
    return augmented


class TrainLanguageAugmentation(Dataset):
    def __init__(self, base):
        if base.split != 'train' or base.profile != 'development':
            raise ValueError('Language augmentation is restricted to train')
        self.base, self.config, self.model_cfg = base, base.config, base.model_cfg
        self.entries=[]
        for entry in base.entries:
            prompt=entry.get('feature_input',entry).get('prompt',entry.get('prompt',''))
            for variant,prefix in enumerate(PREFIXES):
                validate_prefix(prompt,prefix,self.model_cfg)
                self.entries.append({**entry,'sample_id':augmented_id(entry['sample_id'],variant)})

    def __len__(self):
        return len(self.entries)

    def __getitem__(self,index):
        original,variant=divmod(index,len(PREFIXES))
        sample=dict(self.base[original])
        prompt=validate_prefix(sample['prompt'],PREFIXES[variant],self.model_cfg)
        ids,mask=tokenize(prompt,int(self.model_cfg['max_tokens']),int(self.model_cfg['vocab_size']))
        sample['sample_id']=augmented_id(sample['sample_id'],variant)
        sample['prompt']=prompt
        sample['token_ids']=torch.tensor(ids,dtype=torch.long)
        sample['token_mask']=torch.tensor(mask,dtype=torch.bool)
        return sample
