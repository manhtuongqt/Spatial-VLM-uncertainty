import inspect
import unittest
from types import SimpleNamespace
import torch
from workspace.mh_pcrau_v3.hidden_state_adapter import (
    last_valid_indices, gather_last_hidden, extract_h_spatial, collate_prompts, PromptInput,
)


class FakeModel:
    training = False
    tokenizer = SimpleNamespace(padding_side="right", media_token_ids={"image": 99})

    def _embed(self, ids, media, config, labels, mask):
        assert labels is None
        assert mask.all(), "upstream must receive unpadded tokens"
        values = []
        for token in ids[0]:
            values.extend([50., 51., 52.] if token == 99 else [float(token)])
        emb = torch.tensor(values).view(1, -1, 1).expand(-1, -1, 1536)
        return emb, None, torch.ones(emb.shape[:2], dtype=torch.bool)

    def llm(self, **kwargs):
        assert kwargs['labels'] is None and not kwargs['use_cache']
        # Cumulative context detects row/token alignment, unlike an identity mock.
        emb = kwargs['inputs_embeds'] * kwargs['attention_mask'].unsqueeze(-1)
        return SimpleNamespace(hidden_states=(emb.cumsum(1),))


class AdapterTests(unittest.TestCase):
    def test_left_right_indices(self):
        self.assertEqual(last_valid_indices(torch.tensor([[0, 1, 1], [1, 1, 0]])).tolist(), [2, 1])

    def test_invalid_masks(self):
        for mask in [torch.zeros(1, 3), torch.ones(0, 3), torch.ones(3), torch.tensor([[2]])]:
            with self.assertRaises(ValueError): last_valid_indices(mask)

    def test_shape_and_nonfinite(self):
        with self.assertRaises(ValueError): gather_last_hidden(torch.zeros(1, 3, 2), torch.ones(1, 3))
        with self.assertRaises(ValueError): gather_last_hidden(torch.full((1, 3, 1536), float('nan')), torch.ones(1, 3))

    def test_media_expansion_and_padding(self):
        prompts = [PromptInput(torch.tensor([99, 2, 3]), {"image": [torch.zeros(1)]}, {}, ""),
                   PromptInput(torch.tensor([99, 4]), {"image": [torch.zeros(1)]}, {}, "")]
        model = FakeModel()
        for side in ("left", "right"):
            model.tokenizer.padding_side = side
            ids, media, mask = collate_prompts(prompts, 0, side, 3)
            result = extract_h_spatial(model, ids, media, mask)
            self.assertEqual(result.h_spatial[:, 0].tolist(), [158., 157.])
            self.assertEqual(result.post_insertion_sequence_lengths.tolist(), [5, 4])

    def test_media_missing_or_extra(self):
        model = FakeModel()
        with self.assertRaises(ValueError): extract_h_spatial(model, torch.tensor([[99]]), {}, torch.ones(1, 1))
        with self.assertRaises(ValueError): extract_h_spatial(model, torch.tensor([[2]]), {"image": [0]}, torch.ones(1, 1))

    def test_training_rejected(self):
        model = FakeModel(); model.training = True
        with self.assertRaises(ValueError): extract_h_spatial(model, torch.tensor([[2]]), {}, torch.ones(1, 1))

    def test_api_has_no_oracle_parameters(self):
        self.assertEqual(list(inspect.signature(extract_h_spatial).parameters),
                         ['model', 'input_ids', 'media', 'attention_mask', 'media_config'])


if __name__ == '__main__': unittest.main()
