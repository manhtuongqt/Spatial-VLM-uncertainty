"""Prompt-only, parameter-free MH-PCRA-U-v3 spatial feature adapter.

The public tensor API assumes tokens came from the prompt-only builder below.
It cannot determine whether arbitrary caller-supplied text contains an answer.
"""
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
import warnings

import torch


@dataclass
class PromptInput:
    input_ids: torch.Tensor
    media: dict
    media_config: dict
    prompt_text: str


@dataclass
class HSpatialExtraction:
    h_spatial: torch.Tensor
    last_prompt_indices: torch.Tensor
    post_insertion_sequence_lengths: torch.Tensor
    attention_mask: torch.Tensor
    hidden_layer: int = -1


def last_valid_indices(attention_mask):
    if attention_mask.ndim != 2 or not all(attention_mask.shape):
        raise ValueError("Expected a nonempty [batch, sequence] mask")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Mask must be binary")
    mask = attention_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("Every sample needs a valid prompt token")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~mask, -1).max(dim=1).values


def gather_last_hidden(hidden, mask):
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or hidden.shape[-1] != 1536:
        raise ValueError("Expected hidden [B,T,1536] aligned with post-insertion mask")
    idx = last_valid_indices(mask)
    feature = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
    if not torch.isfinite(feature).all():
        raise ValueError("Non-finite spatial feature")
    return feature.detach(), idx


def prepare_rgbd_prompt(model, *, rgb_path, depth_path, instruction):
    """Build the same RGB-D prompt as canonical generate_content, without generation."""
    from llava.media import Image, Depth
    from llava.utils.media import extract_media
    from llava.utils.tokenizer import tokenize_conversation
    from llava.mm_utils import (
        process_rgbd_inference_conversation, dynamic_process_images_and_prompt,
        dynamic_process_depths_and_prompt, process_image, process_depth,
    )
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    messages = [{"from": "human", "value": [Image(str(rgb_path)), Depth(str(depth_path)), instruction]}]
    media = extract_media(messages, model.config)
    config = copy(model.config)
    if config.image_aspect_ratio == "dynamic":
        messages[0]["value"] = process_rgbd_inference_conversation(messages[0]["value"])
    elif config.image_aspect_ratio == "dynamic_s2":
        raise ValueError("dynamic_s2 requires a separately validated preprocessing contract")
    for name, processor in (("image", model.get_vision_tower().image_processor),
                            ("depth", model.get_depth_tower().image_processor)):
        config.image_processor = processor
        if config.image_aspect_ratio == "dynamic":
            fn = dynamic_process_images_and_prompt if name == "image" else dynamic_process_depths_and_prompt
            tensors, messages[0]["value"] = fn(media[name], messages[0]["value"], config)
        else:
            fn = process_image if name == "image" else process_depth
            tensors = torch.stack([fn(x, config, None) for x in media[name]])
        tower = model.get_vision_tower() if name == "image" else model.get_depth_tower()
        parameter = next(tower.parameters())
        media[name] = list(tensors.to(device=parameter.device, dtype=parameter.dtype))
    ids = tokenize_conversation(deepcopy(messages), model.tokenizer, add_generation_prompt=True)
    ids = ids.to(model.device)
    rendered = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": messages[0]["value"].strip()}],
        add_generation_prompt=True, tokenize=False,
    )
    return PromptInput(ids, dict(media), defaultdict(dict), rendered)


def collate_prompts(prompts, pad_token_id, padding_side="right", extra_padding=0):
    if not prompts or padding_side not in ("left", "right") or extra_padding < 0:
        raise ValueError("Invalid prompt batch or padding setting")
    length = max(p.input_ids.numel() for p in prompts) + extra_padding
    ids = prompts[0].input_ids.new_full((len(prompts), length), pad_token_id)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    media = defaultdict(list)
    for row, prompt in enumerate(prompts):
        n = prompt.input_ids.numel()
        start = length - n if padding_side == "left" else 0
        ids[row, start:start+n] = prompt.input_ids
        mask[row, start:start+n] = True
        for name, values in prompt.media.items():
            media[name].extend(values)
    return ids, dict(media), mask


@torch.inference_mode()
def extract_h_spatial(model, input_ids, media, attention_mask, media_config=None):
    """Extract final prompt states; keep left padding aligned before media insertion.

    Upstream _embed removes text padding but scans the original input_ids. Embed
    each unpadded row first, then batch the resulting embeddings. LLM processing
    is a real mixed-length batch with logical position IDs independent of padding.
    """
    if model.training:
        raise ValueError("Frozen feature extraction requires model.eval()")
    last_valid_indices(attention_mask)
    if input_ids.shape != attention_mask.shape:
        raise ValueError("Token/mask shapes differ")
    if model.tokenizer.padding_side not in ("left", "right"):
        raise ValueError("Unknown tokenizer padding side")
    media_config = media_config or defaultdict(dict)
    offsets = {name: 0 for name in media}
    embeddings = []
    for row in range(input_ids.shape[0]):
        tokens = input_ids[row][attention_mask[row].bool()].unsqueeze(0)
        selected = {}
        for name, token_id in model.tokenizer.media_token_ids.items():
            count = int((tokens == token_id).sum())
            if count:
                start = offsets.get(name, 0)
                values = media.get(name, [])[start:start+count]
                if len(values) != count:
                    raise ValueError("Media count does not match prompt tokens")
                selected[name] = values
                offsets[name] = start + count
        with warnings.catch_warnings(record=True) as caught:
            emb, _, mask = model._embed(tokens, selected, deepcopy(media_config), None,
                                        torch.ones_like(tokens, dtype=torch.bool))
        if any("truncat" in str(w.message).lower() for w in caught):
            raise ValueError("Prompt truncation would invalidate the extraction index")
        for warning in caught:
            warnings.warn(str(warning.message), warning.category)
        embeddings.append(emb[0][mask[0].bool()])
    if any(offsets.get(name, 0) != len(values) for name, values in media.items()):
        raise ValueError("Unused media supplied")
    lengths = torch.tensor([e.shape[0] for e in embeddings], device=input_ids.device)
    max_len = int(lengths.max())
    batch = embeddings[0].new_zeros((len(embeddings), max_len, embeddings[0].shape[-1]))
    mask = torch.zeros(batch.shape[:2], dtype=torch.bool, device=batch.device)
    for row, emb in enumerate(embeddings):
        start = max_len - len(emb) if model.tokenizer.padding_side == "left" else 0
        batch[row, start:start+len(emb)] = emb
        mask[row, start:start+len(emb)] = True
    position_ids = (mask.long().cumsum(-1) - 1).clamp_min(0)
    outputs = model.llm(inputs_embeds=batch, attention_mask=mask,
                        position_ids=position_ids, labels=None, use_cache=False,
                        output_hidden_states=True, return_dict=True)
    feature, indices = gather_last_hidden(outputs.hidden_states[-1], mask)
    return HSpatialExtraction(feature, indices, lengths, mask)
