"""Opt-in frozen Sidecar → noun-conditioned anchor → geometry → calibrated risk.

Runtime boundary is features + prompts only. Labels belong to external audits.
Sequential calls only (AnchorShadow uses temporary hooks).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from .anchor_shadow import AnchorShadow
from .answerability_evidence import EVIDENCE_NAMES
from .checkpoint import load_model_checkpoint
from .dataset import ANSWER_CLASSES, SOURCE_CLASSES
from .engine import autocast_context
from .horizontal_verifier import verify
from .model import PCRAUTargetV2
from .postprocess import summarize_heatmap
from .selective_experiment import experimental_action, predicted_answer
from .text import prompt_anchor_mask, relation_ids, tokenize
from .utils import read_json, sha256_file
from .verifier_risk import apply_verifier_risk, feature_names, risk_vector

VERSION = 'pcrau_unified_live_evidence_v1'
FEATURE_SHAPES = {'r0': (24, 32, 1152), 'd0': (24, 32, 1152),
                  'r_thumb': (1152,), 'd_thumb': (1152,)}
CACHE_KEYS = dict(zip(('R0_GRID', 'D0_GRID', 'R0_THUMB', 'D0_THUMB'), FEATURE_SHAPES))


def load_features(path: str | Path) -> dict[str, torch.Tensor]:
    tensors = load_file(str(path), device='cpu')
    if set(tensors) != set(CACHE_KEYS):
        raise ValueError('Feature file must contain only the four frozen visual tensors')
    return {CACHE_KEYS[k]: v for k, v in tensors.items()}


def make_batch(features: Mapping[str, torch.Tensor], prompts: Sequence[str], cfg, device):
    if set(features) != set(FEATURE_SHAPES):
        raise ValueError('Runtime accepts only four visual tensors; no oracle/metadata')
    if isinstance(prompts, str) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError('Expected nonempty sequence of nonempty prompts')
    n = len(prompts)
    batch = {}
    for key, shape in FEATURE_SHAPES.items():
        value = features[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (n, *shape):
            raise ValueError(f'Invalid batched feature shape: {key}')
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid floating visual evidence: {key}')
        # Identical storage quantization for extracted and cached visual features.
        batch[key] = value.to(device=device, dtype=torch.float16).float()
    values = {k: [] for k in ('token_ids', 'token_mask', 'relation_ids', 'relation_mask', 'anchor_mask')}
    for prompt in prompts:
        ids, mask = tokenize(prompt, int(cfg['max_tokens']), int(cfg['vocab_size']))
        rid, rm = relation_ids(prompt, int(cfg['max_relations']))
        am = prompt_anchor_mask(prompt, int(cfg['max_anchors']))
        for key, value in zip(values, (ids, mask, rid, rm, am)):
            values[key].append(value)
    for key, value in values.items():
        batch[key] = torch.tensor(value, dtype=torch.long if key.endswith('ids') else torch.bool, device=device)
    if any(not bool(torch.isfinite(v).all()) for v in batch.values()):
        raise ValueError('Feature quantization overflow')
    return batch


def export_observable(output, batch, prompts, anchor_logits, bundle_sha256):
    """No labels or saved prediction rows accepted. Pre-Adapter evidence preserved."""
    if set(batch) != PCRAUTargetV2.MODEL_INPUT_KEYS:
        raise ValueError('Forbidden evidence input fields')
    if any(not bool(torch.isfinite(v).all()) for v in output.values()):
        raise ValueError('Nonfinite model output')
    target = output['target_logits'].detach().float().cpu()
    answer = output['answerability_logits'].detach().float().softmax(-1).cpu()
    source = output['source_logits'].detach().float().sigmoid().cpu()
    edge = output['relation_edge_logits'].detach().float().sigmoid().cpu()
    observed = output['observable_answerability_evidence'].detach().float().cpu()
    if observed.shape != (len(prompts), len(EVIDENCE_NAMES)):
        raise ValueError('Pre-Adapter evidence schema drift')
    old = output['anchor_logits'].detach().float().cpu().numpy()
    new = anchor_logits.detach().float().cpu().numpy()
    edge_mask = (batch['relation_mask'] & batch['anchor_mask'][:, :edge.shape[1]]).cpu()
    rows = []
    for i, prompt in enumerate(prompts):
        spatial = summarize_heatmap(target[i])
        probability = target[i].flatten().softmax(0).reshape(24, 32).numpy()
        spatial['probability_grid'] = probability.tolist()
        rgb, depth = batch['r_thumb'][i].detach().float().cpu(), batch['d_thumb'][i].detach().float().cpu()
        trace = verify(prompt, probability, new[i, 0], old[i, 0])
        row = {'schema_version': 1, 'evidence_producer': VERSION, 'prompt': prompt,
               'spatial': spatial,
               'answerability_probabilities': {name: float(answer[i, j]) for j, name in enumerate(ANSWER_CLASSES)},
               'source_probabilities': {name: float(source[i, j]) for j, name in enumerate(SOURCE_CLASSES)},
               'relation_edge_probabilities': edge[i].tolist(),
               'relation_consistency': float(edge[i][edge_mask[i]].mean()) if bool(edge_mask[i].any()) else 1.,
               'fusion_gate_mean': float(output['fusion_gate_mean'][i].detach().float().cpu()),
               'rgb_depth_cosine': float(torch.nn.functional.cosine_similarity(rgb[None], depth[None]).item()),
               'rgb_depth_mae': float((rgb-depth).abs().mean().item()),
               'roborefer_point_xy': None, 'roborefer_disagreement_normalized': None,
               'observable_answerability_evidence': dict(zip(EVIDENCE_NAMES, observed[i].tolist())),
               'verifiers': {'P1': trace}, 'verifier_bundle_sha256': bundle_sha256}
        x = risk_vector(row, 'P1_G44')
        if x.shape != (44,) or not np.isfinite(x).all():
            raise ValueError('Invalid live evidence vector')
        rows.append(row)
    return rows


def attach_decisions(rows, calibrator, profile):
    if calibrator.get('evidence_producer') != VERSION or calibrator['model'] != 'P1_G44':
        raise ValueError('Calibrator was not fit for this live evidence producer')
    if profile['policy'] != 'hard_found' or profile['bundle_sha256'] != calibrator['bundle_sha256']:
        raise ValueError('Policy/bundle mismatch')
    risk = apply_verifier_risk(rows, calibrator)
    names = feature_names('P1_G44')
    mean, scale, weights = [np.asarray(calibrator[k]) for k in ('mean', 'scale', 'coefficients')]
    results = []
    for row, value in zip(rows, risk):
        x = risk_vector(row, 'P1_G44')
        contributions = (x-mean)/scale*weights
        logit = float(contributions.sum()+calibrator['intercept'])
        # Neutralize raw geometry features only; this is an inference diagnostic,
        # not a retrained ablation or uncertainty decomposition.
        neutral = x.copy(); neutral[-3:] = 0
        neutral_logit = float(((neutral-mean)/scale*weights).sum()+calibrator['intercept'])
        action = experimental_action(row, float(value), 'hard_found', profile['threshold'])
        decision = {'predicted_answerability': predicted_answer(row), 'risk': float(value),
                    'risk_event': 'truth_non_FOUND_OR_target_MAP_outside_target',
                    'threshold': profile['threshold'], 'policy': 'hard_found', 'action': action,
                    'decision_level': 'perception_only', 'robot_motion_commanded': False}
        trace = {'feature_names': names, 'feature_values': x.tolist(),
                 'standardized_logit_contributions': contributions.tolist(),
                 'intercept': calibrator['intercept'], 'risk_logit': logit,
                 'geometry_raw_logit_effect_vs_zero': logit-neutral_logit,
                 'geometry_zero_diagnostic_is_retrained_ablation': False,
                 'target_map_before_xy': row['spatial']['map_pixel_xy'],
                 'target_map_after_xy': row['spatial']['map_pixel_xy'],
                 'target_map_changed': False,
                 'evidence_origin': 'current_model_forward_only',
                 'aleatoric_epistemic_decomposition': False}
        results.append({**row, 'decision': decision, 'risk_trace': trace})
    return results


class UnifiedInference:
    def __init__(self, config, baseline, residual_path, bundle_sha256, calibrator=None, profile=None):
        self.config = config
        self.device = next(baseline.parameters()).device
        self.shadow = AnchorShadow(baseline, 'phrase', 24082026).eval()
        self.shadow.residual.load_state_dict(load_file(str(residual_path), device='cpu'), strict=True)
        self.shadow.requires_grad_(False)
        self.bundle_sha256 = bundle_sha256
        self.calibrator, self.profile = calibrator, profile

    @classmethod
    def from_bundle(cls, path: str | Path, device='cuda'):
        path = Path(path).resolve()
        manifest = read_json(path)
        if manifest['version'] != VERSION:
            raise ValueError('Unsupported inference bundle version')
        def checked(desc):
            p = Path(desc['path'])
            p = p if p.is_absolute() else path.parent/p
            if sha256_file(p) != desc['sha256']:
                raise ValueError(f'Bundle hash mismatch: {p}')
            return p
        lock_path = checked(manifest['freeze_lock'])
        lock = read_json(lock_path)
        for desc in lock['source_files'].values():
            checked(desc)
        # Calibration operational contract fixes CUDA BF16, not CPU FP32.
        if torch.device(device).type != lock['device_type']:
            raise ValueError('Device/precision differs from the live calibration contract')
        if torch.__version__ != lock['runtime']['torch']:
            raise ValueError('Sidecar torch runtime differs from calibrated bundle; backbone uses a separate worker')
        cfg_path = checked(lock['config'])
        checkpoint = checked(lock['baseline_checkpoint'])
        checked(lock['baseline_metadata'])
        residual = checked(lock['anchor_residual'])
        config = read_json(cfg_path)
        baseline = PCRAUTargetV2(config).to(device).eval()
        load_model_checkpoint(baseline, checkpoint, lock['config']['sha256'])
        cal = read_json(checked(manifest['calibrator']))
        profile = read_json(checked(manifest['profile']))
        if cal['bundle_sha256'] != sha256_file(lock_path) or profile['bundle_sha256'] != sha256_file(lock_path):
            raise ValueError('Calibration/freeze lock mismatch')
        return cls(config, baseline, residual, sha256_file(lock_path), cal, profile)

    @torch.no_grad()
    def observe(self, features, prompts):
        batch = make_batch(features, prompts, self.config['model'], self.device)
        with autocast_context(self.device, self.config['optimization']):
            cap = self.shadow.capture(batch, prompts)
            anchor = self.shadow.predict(cap)
        inactive = ~cap.active_slots
        if not torch.equal(anchor[inactive], cap.baseline['anchor_logits'][inactive]):
            raise RuntimeError('Bypass baseline anchor was changed')
        return export_observable(cap.baseline, batch, prompts, anchor, self.bundle_sha256)

    def predict(self, features, prompts):
        if self.calibrator is None or self.profile is None:
            raise RuntimeError('Prediction decisions require a frozen calibrated bundle')
        return attach_decisions(self.observe(features, prompts), self.calibrator, self.profile)
