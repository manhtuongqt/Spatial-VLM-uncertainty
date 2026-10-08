#!/usr/bin/env python3
"""Replay the fixed RefCOCO pilot through the new primary-point runtime adapter.

No generation, tuning or calibration fit; original predictions remain intact.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'new/outputs/refcoco_val_pilot300_20261007'
OUT = ROOT/'new/outputs/pcrau_roborefer_primary_pilot300_20261007'
sys.path[:0] = [str(ROOT/'new/src'), str(ROOT/'new/scripts'), str(ROOT/'RoboRefer/Evaluation')]
from pcrau.roborefer_primary import compose_prediction, generation_prompt, VERSION
from evaluate_refspatial_location import sha, write, rows, protected


def main():
    import numpy as np
    from summarize_acc import text2pts
    from evaluate_refcoco_pilot import point_inside
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import textwrap
    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads((SOURCE/'protocol.json').read_text())
    protected_files = protected()
    assert sha(SOURCE/'runtime_manifest.json') == original['runtime_manifest_sha256']
    runtime = json.loads((SOURCE/'runtime_manifest.json').read_text())
    backbone, auxiliary = rows(SOURCE/'backbone_predictions.jsonl'), rows(SOURCE/'predictions.jsonl')
    assert len(runtime) == len(backbone) == len(auxiliary) == 300
    OUT.mkdir()
    source_files = {str(SOURCE/name): sha(SOURCE/name) for name in
                    ('runtime_manifest.json', 'evaluator_manifest.json', 'backbone_predictions.jsonl',
                     'predictions.jsonl', 'protocol.json')}
    protocol = {'version': VERSION, 'mode': 'REPLAY_EXISTING_FIXED_GENERATION_NOT_NEW_TEST',
                'primary_source': 'matched640_RGBD_RoboRefer_answer',
                'target_selection': 'always_RoboRefer; no_sidecar_or_GT_fallback',
                'samples': 300, 'source_files': source_files,
                'source_code_sha256': sha(ROOT/'new/src/pcrau/roborefer_primary.py'),
                'runner_sha256': sha(__file__), 'protected_files': protected_files,
                'calibration_status': 'PENDING_NEW_PRIMARY_EVENT',
                'train_steps': 0, 'fit_calls': 0, 'threshold_tuned': False,
                'selection_chosen_after_observing_original_pilot': True}
    write(OUT/'protocol.json', protocol)
    predictions = []
    # Only oracle-free runtime data and generated outputs enter the composer.
    with (OUT/'predictions.jsonl').open('w') as f:
        for e, b, s in zip(runtime, backbone, auxiliary):
            assert e['id'] == b['id'] == s['id'] and e['image_id'] == b['image_id'] == s['image_id']
            assert e['backbone_prompt'] == generation_prompt(e['prompt'])
            assert sha(e['feature_path']) == b['feature_sha256']
            r = compose_prediction(e['prompt'], b['answer'], image_size=e['original_size'], sidecar_observation=s)
            r.update(id=e['id'], image_id=e['image_id'])
            predictions.append(r)
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    # Ground truth is read only after final target predictions have been saved.
    assert sha(SOURCE/'evaluator_manifest.json') == original['evaluator_manifest_sha256']
    evaluator = json.loads((SOURCE/'evaluator_manifest.json').read_text())
    paired = []
    for e, r, s, b in zip(evaluator, predictions, auxiliary, backbone):
        assert e['id'] == r['id']
        w, h = e['image_size']
        p = r['target']['pixel_xy']
        official = text2pts(b['answer'], w, h, False).tolist()
        assert official == [p], 'Coordinate mismatch with official point parser'
        sx, sy = s['spatial']['map_pixel_xy']
        sidepoint = [int(sx/640*w), int(sy/480*h)]
        hit = p is not None and point_inside(p, e['bbox_xywh'], e['image_size'])
        paired.append({**e, 'primary_point': p, 'primary_hit': hit, 'sidecar_point': sidepoint,
                       'sidecar_hit': point_inside(sidepoint, e['bbox_xywh'], e['image_size']),
                       'new_action': r['decision']['action'], 'new_risk': r['decision']['risk'],
                       'old_sidecar_action_diagnostic': s['decision']['action'],
                       'old_sidecar_risk_diagnostic': s['decision']['risk']})
    hit = np.array([r['primary_hit'] for r in paired], dtype=float)
    old = np.array([r['sidecar_hit'] for r in paired], dtype=float)
    ids = sorted({r['image_id'] for r in paired})
    groups = np.array([[i for i, r in enumerate(paired) if r['image_id'] == iid] for iid in ids])
    assert groups.shape == (100, 3)
    rng = np.random.default_rng(24082026)
    indices = groups[rng.integers(0, 100, (10000, 100))].reshape(10000, 300)
    fixed = [r['id'] for r in paired if r['primary_hit'] and not r['sidecar_hit']]
    broken = [r['id'] for r in paired if not r['primary_hit'] and r['sidecar_hit']]
    summary = {'version': VERSION, 'samples': 300, 'images': 100,
               'grounding_hits': int(hit.sum()), 'point_in_box': float(hit.mean()),
               'sidecar_grounding_hits': int(old.sum()), 'sidecar_point_in_box': float(old.mean()),
               'point_in_box95_percent': (100*np.quantile(hit[indices].mean(1), [.025, .975])).tolist(),
               'delta_pp': float(100*(hit-old).mean()),
               'delta95_pp': (100*np.quantile((hit-old)[indices].mean(1), [.025, .975])).tolist(),
               'fixed_ids': fixed, 'broken_ids': broken,
               'official_point_decoder_crosscheck': True,
               'same_as_backbone_control_by_construction': True,
               'calibrated_primary_risk': None, 'primary_selective_coverage': None,
               'new_independent_test': False, 'GT_input': False,
               'train_steps': 0, 'fit_calls': 0, 'threshold_tuned': False}
    write(OUT/'paired_evaluator.json', paired)
    write(OUT/'summary.json', summary)
    # Show recovered cases and counterexamples, including the same banana errors.
    chosen = list(dict.fromkeys([141430, 141432]+fixed[:2]+broken[:2]))
    idx = {r['id']: i for i, r in enumerate(paired)}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(hspace=.35)
    for ax, sid in zip(axes.flat, chosen):
        i = idx[sid];r = paired[i]
        ax.imshow(Image.open(runtime[i]['original_rgb']).convert('RGB'))
        bx, by, bw, bh = r['bbox_xywh']
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor='lime', lw=2))
        ax.scatter(*r['sidecar_point'], marker='x', c='red', s=100, label='Sidecar target')
        ax.scatter(*r['primary_point'], marker='+', c='cyan', s=110, label='RoboRefer primary')
        ax.set_title(textwrap.fill(f"{sid}: {r['phrase']}", 48)+
                     f"\nSidecar={int(r['sidecar_hit'])}; primary={int(r['primary_hit'])}; risk=pending", fontsize=9)
        ax.legend(fontsize=7);ax.axis('off')
    fig.savefig(OUT/'cases.png', dpi=160, bbox_inches='tight');plt.close(fig)
    for p, value in {**protected_files, **source_files}.items():
        assert sha(p) == value, p
    write(OUT/'FINAL_STATUS.json', {'status': 'COMPLETE_PRIMARY_GROUNDING_REPLAY_CALIBRATION_PENDING',
                                   'protected_and_source_artifacts_unchanged': True,
                                   'primary_predictions_sha256': sha(OUT/'predictions.jsonl'),
                                   'robot_motion_commanded': False})
    report = f'''# RoboRefer làm grounding chính — pilot RefCOCO

Ngày 07/10/2026. Biến thể opt-in, chưa thay active profile.

## Thay đổi đã triển khai

RGB/depth → RoboRefer đầy đủ → một normalized point → đầu ra target cuối.
Song song, pre-projector features → frozen Tasks60 Sidecar → diagnostics/anchor.
Parser + điểm target RoboRefer + predicted anchor peak → kiểm tra trái/phải dạng point evidence.
Direct/unsupported không bịa kết quả kiểm chứng. Không tạo heatmap giả từ point.

Entry point live: `new/scripts/infer_roborefer_primary.py`; adapter: `new/src/pcrau/roborefer_primary.py`.
Sidecar target, answerability và policy cũ không thay/veto đầu ra grounding mới.

## Kết quả replay pilot đã quan sát

| Đầu ra target | Point-in-box |
|---|---:|
| Sidecar trước thay đổi | {int(old.sum())}/300 ({100*old.mean():.2f}%) |
| RoboRefer primary | {int(hit.sum())}/300 ({100*hit.mean():.2f}%) |

Delta +{summary['delta_pp']:.2f} pp; 95% CI theo 100 image groups {summary['delta95_pp']} pp.
Sửa {len(fixed)} câu, làm sai {len(broken)} câu so với Sidecar. Không chọn nhánh theo nhãn thật.
Điểm primary khớp 300/300 với parser điểm official và đúng bằng đối chứng RoboRefer RGB-D đã chạy.
**Đây là lấy lại năng lực backbone, không là cải thiện RoboRefer hoặc bằng chứng uncertainty cải thiện grounding.**
Thiết kế được chọn sau khi thấy pilot cũ; replay dùng nguyên predictions đã lưu, không test độc lập mới.

![Case thật](../new/outputs/pcrau_roborefer_primary_pilot300_20261007/cases.png)

Xanh lá: GT bbox hậu kiểm; đỏ: Sidecar; cyan: primary. Có cả case sửa và case bị làm sai.

## Calibration và MC: phần chưa được chuyển sang target mới

Primary risk/threshold là null; output valid báo `REVIEW_UNCALIBRATED`, không tự EXECUTE.
Invalid/multiple/out-of-range point không fallback Sidecar mà báo `REOBSERVE_INVALID_POINT`.
Các risk và MC distributions cũ chỉ nằm trong `sidecar_diagnostic`, gắn rõ event/nhánh chúng đo.
Không gọi entropy map Sidecar là uncertainty của điểm RoboRefer. Điểm deterministic không tự cung cấp phân bố.
Geometry dùng point target thật, không đẩy Dirac/gaussian giả vào calibrator44/60.

Để hoàn thiện quyết định: freeze nhánh primary; tạo evidence train/dev/calibration từ đúng primary point;
fit calibrator mới trên calibration với event non-FOUND OR điểm RoboRefer ngoài target mask, chọn threshold ở calibration;
sau đó reevaluate IID cũ. Không fit/chọn ngưỡng trên pilot RefCOCO.
Các semantic/depth/completion quantities phụ thuộc vị trí cần được lấy lại tại vị trí primary trước khi gọi là uncertainty cho target mới.
Spatial MC hiện là bất định của Sidecar; muốn bất định grounding backbone phải bổ sung cơ chế phân bố có căn cứ.

## Kiểm chứng và bảo toàn

Adapter kiểm tra prompt alignment, coordinate boundary, không oracle và không đổi outputs Sidecar.
Live worker dùng cùng RGB-D640×480, frozen model, greedy max_new_tokens128, features FP16, inventory hash được kiểm tra.
Không train/fit ở bước này; original pilot, bundle, calibrator và active profile được kiểm tra hash không đổi.
Chưa đưa robot chuyển động. Kết quả này là chất lượng candidate grounding; coverage/risk của pipeline mới còn chưa đánh giá.
'''
    (ROOT/'plan/ROBOREFER_PRIMARY_PILOT_20261007.md').write_text(report)
    print(json.dumps({k: summary[k] for k in ('grounding_hits', 'point_in_box', 'delta_pp', 'delta95_pp')}, indent=2))


if __name__ == '__main__':
    main()
