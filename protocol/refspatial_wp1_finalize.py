#!/usr/bin/env python3
"""Verify generated artifacts and write Vietnamese WP0-WP1 reports."""
import collections
import csv
import hashlib
import json
from pathlib import Path

from refspatial_wp1_audit import ROOT, NAMES, sha, write_json, jsonl
from refspatial_build_supplement import json_array

BASE = ROOT/'results/spatial_vlm_refspatial_v1'
OUT = BASE/'wp1_source_audit'


def main():
    summary = json.loads((OUT/'audit_summary.json').read_text())
    media = json.loads((OUT/'media_qc_report.json').read_text())
    qa = json.loads((OUT/'qa_type_report.json').read_text())
    queue = list(jsonl(OUT/'manual_audit_queue.jsonl'))
    checks = {}
    checks['source_files_unchanged'] = all(sha(ROOT/r['path']) == r['sha256'] for r in json.loads((OUT/'source_inventory.json').read_text()))
    shared_content = collections.defaultdict(set)
    for r in jsonl(OUT/'media_manifest.jsonl'):
        for sid in r['referenced_by_scenes']:
            shared_content[(sid, r['kind'])].add(r['sha256'])
    inconsistent_shared = [{'scene_id': sid, 'kind': kind, 'sha256_values': sorted(values)}
                           for (sid, kind), values in shared_content.items() if len(values) > 1]
    write_json(OUT/'shared_scene_content_consistency.json', {'inconsistent_scene_media': inconsistent_shared})
    checks['shared_scene_media_identical_across_folders'] = not inconsistent_shared
    equivalence = []
    for name in NAMES:
        folder = ROOT/'datasets'/name
        if not (folder/'metadata.json').exists():
            equivalence.append({'dataset': name, 'status': 'JSON_ARRAY_NOT_PRESENT_JSONL_CANONICAL'})
            continue
        digests = []
        for rows in [jsonl(folder/'metadata.jsonl'), json_array(folder/'metadata.json')]:
            h = hashlib.sha256()
            count = 0
            for row in rows:
                h.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
                h.update(b'\n')
                count += 1
            digests.append((count, h.hexdigest()))
        equivalence.append({'dataset': name, 'jsonl_count_sha256': digests[0], 'json_array_count_sha256': digests[1],
                            'equivalent': digests[0] == digests[1]})
    write_json(OUT/'metadata_representation_consistency.json', equivalence)
    checks['json_jsonl_equivalent_where_available'] = all(r.get('equivalent', True) for r in equivalence)
    ids, family_splits, n = set(), collections.defaultdict(set), 0
    for r in jsonl(OUT/'object_candidates.jsonl'):
        assert r['sample_id'] not in ids
        ids.add(r['sample_id'])
        assert r['target_kind'] == 'object' and not r['training_eligible']
        assert 'think' not in r and 'thinking' not in r
        family_splits[r['family_id']].add(r['split'])
        n += 1
    checks['unique_candidate_ids'] = len(ids) == n == summary['candidate_qa_after_structural_gate']
    checks['candidate_family_disjoint'] = all(len(s)==1 for s in family_splits.values())
    checks['queue_ids_exist_and_unique'] = len({r['sample_id'] for r in queue}) == len(queue) and all(r['sample_id'] in ids for r in queue)
    checks['gallery_media_exists'] = all((ROOT/r[k]).is_file() for r in queue for k in ['image','depth'])
    checks['minimum_300_scene_queries'] = len(queue) >= 300
    checks['available_relation_quotas'] = all(v['queued'] >= 50 for v in json.loads((OUT/'manual_audit_status.json').read_text())['relation_quotas'].values() if v['queued'])
    write_json(OUT/'artifact_validation.json', {'checks': checks, 'passed': all(checks.values()), 'scope': 'AUTOMATED_ARTIFACT_INTEGRITY_NOT_LABEL_QC'})
    taxonomy = list(csv.DictReader((OUT/'taxonomy_audit.csv').open()))
    changed = collections.Counter(r['dataset'] for r in taxonomy if r['changed']=='True')
    transitions = collections.Counter((r['original_categories'],r['derived_categories']) for r in taxonomy if r['dataset']==NAMES[0] and r['changed']=='True')
    relations = list(csv.DictReader((OUT/'relation_feasibility.csv').open()))
    lines = ['# WP1 — Báo cáo audit RefSpatial cục bộ', '', 'Ngày: 08/09/2026.', '',
        '**Trạng thái: audit tự động hoàn tất; manual QC chưa hoàn tất; chưa được training.**', '',
        '## Media và split', '',
        f'- Đã hash/decode toàn bộ **{media["counts"]["files"]:,} file**: 13.204 RGB và 13.206 depth, đều 960×540 RGB; depth có ba kênh bằng nhau.',
        '- Không thiếu file theo scene manifest, không lỗi pairing/shape, không phát hiện duplicate SHA-256 hoặc pixel-content hash qua split.',
        '- Hợp năm folder có 11.048 scene ID. Những folder chồng lấn chỉ cung cấp tag; không nhân bản candidate từ Large.',
        '- Hai depth `(1).png` trùng byte với file được manifest tham chiếu. Đã loại khỏi candidate selection; không xóa file nguồn.',
        '- Kiểm tra hash không loại trừ việc reuse asset/layout hoặc scene từng xuất hiện trong pretraining.', '',
        '## QA và candidate pool', '',
        '| Folder | Records | Object QA | Placement QA |', '|---|---:|---:|---:|']
    for name in NAMES:
        c = qa['datasets'][name]['counts']
        lines.append(f'| {name} | {c["records"]:,} | {c["object_qa"]:,} | {c["free_space_qa"]:,} |')
    lines += ['', '- Large: 201.442 object QA − 7 normalized duplicates − 2 target cần review = **201.433 candidate** thuộc 10.323 scene.',
        '- Hai target cách ly thuộc `95c794c73e8535aa` và `9344468c70fdbbe0`: point in-bounds nhưng không khớp recovered scene point. Chưa kết luận annotation gốc sai.',
        '- 256.623 placement QA của Large không vào object pool. Có 196 occurrence cùng scene/query nhưng khác answer trong placement; có thể có nhiều đáp án free-space hợp lệ.',
        '- `object_candidates.jsonl` là manifest annotation offline, có target/anchor candidates và luôn `training_eligible=false`. Inference chỉ được dùng image/depth/instruction, không đọc các trường offline.',
        '- `qualitative_only_scenes.jsonl` giữ stress scenes ngoài Large riêng biệt.', '', '## Taxonomy', '',
        f'- Derived table sửa {changed[NAMES[0]]:,} point trong Large: 360 bỏ fruit khỏi container màu orange và 2.215 bỏ fruit khỏi nhãn đồ uống/container chỉ mang category fruit cũ.',
        f'- Tabletop có {changed[NAMES[1]]:,} point thay đổi, là subset chồng lấn; không cộng thành số point độc lập.',
        '- Đây là sửa bằng ngữ cảnh noun trong label, chưa phải visual annotation. Nhãn nhiều đối tượng/quan hệ được để unresolved; không sửa metadata/scenes nguồn và không rebuild selection bias của pool.', '',
        '## Relation feasibility', '', '| Nhóm lexical candidate | QA | Train family | Train family có một anchor candidate offline |', '|---|---:|---:|---:|']
    for r in relations:
        lines.append(f'| {r["relation_candidate"]} | {r["qa_candidates"]} | {r["train_scene_families"]} | {r["train_families_with_single_offline_anchor"]} |')
    lines += ['', '- Năm nhóm ranking vượt ngưỡng **candidate** 500 train family. Số audited-clean family hiện bằng 0 cho mọi nhóm.',
        '- Nearest/farthest ở đây bao gồm diễn đạt xếp hạng thứ N; không mặc định tất cả là argmin/argmax 1 bước. Frame, target và reasoning depth vẫn phải review.',
        '- Không tìm thấy lexical candidate pairwise left/right, front/behind hoặc object-between. Đây không phải chứng minh không có mọi diễn đạt tương đương.',
        '- Anchor trích từ rationale là ứng viên offline, không được đưa vào model input hoặc coi là GT đã xác minh.', '',
        '## Manual audit và bước tiếp theo', '',
        '- Đã tạo **300 scene-query** (299 scene duy nhất), gồm 50 mẫu cho từng nhóm ranking khả dụng; mỗi nhóm lấy 30 train / 10 dev / 10 diagnostic.',
        '- Phiếu người review hiện để trống; chưa có precision hoặc Wilson CI để kết luận nhãn đạt 95%.',
        '- Mở [gallery](MANUAL_AUDIT.html), đọc [hướng dẫn](MANUAL_AUDIT_GUIDE.md), điền [phiếu CSV](manual_audit_decisions.csv), rồi chạy validator.',
        '- Sau review: sửa nhãn hệ thống, audit lại khi cần, chốt ontology/frame/anchor extraction và tạo clean release. Không tự suy 500 family sạch từ 50 ảnh đã review nếu chưa có protocol label extraction được kiểm chứng.',
        '- Raw source revision/filter/license chưa phục hồi; chỉ các output cục bộ đã khóa SHA-256. RefSpatial vẫn potentially seen khi RoboRefer pretrain.', '',
        '## Tái lập', '', '```bash', 'python3 -m unittest discover -s protocol -p test_refspatial_wp1.py -v',
        'python3 protocol/refspatial_wp1_audit.py', 'python3 protocol/refspatial_manual_review.py',
        'python3 protocol/refspatial_wp1_finalize.py', '```', '',
        'Chạy từ root workspace. Không sửa source; rerun giữ phiếu review hiện có và yêu cầu sample IDs khớp queue.', '',
        'Nguồn số liệu: `audit_summary.json`, `media_qc_report.json`, `qa_type_report.json`, `taxonomy_audit.csv`, `relation_feasibility.csv`, `artifact_validation.json`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    wp0 = BASE/'wp0_inventory'
    env = json.loads((wp0/'environment.json').read_text())
    inv = json.loads((wp0/'workspace_inventory.json').read_text())
    reuse = json.loads((wp0/'artifact_reuse_matrix.json').read_text())
    weights = json.loads((wp0/'model_weight_inventory.json').read_text())
    (wp0/'REPORT.md').write_text(f'''# WP0 — Môi trường, tài nguyên và khả năng tái sử dụng

Ngày: 08/09/2026. Đã kiểm tra trực tiếp trên workspace; chưa chạy inference RoboRefer end-to-end hoặc training.

- Python **3.10.14** trong `.conda-roborefer`; PyTorch **{env['torch_runtime']['version']}**, CUDA build **{env['torch_runtime']['cuda_build']}**. Khác phiên bản ghi trong DEPENDENCIES.md cũ.
- GPU NVIDIA RTX 2000 Ada, **16.380 MiB** VRAM; CUDA forward/backward smoke trên tensor nhỏ PASS. Đây không phải benchmark khả năng train RoboRefer.
- RAM khoảng 15,3 GiB; disk free đầu inventory **{env['disk']['free']/1024**3:.2f} GiB**. Không nên sao chép toàn bộ dataset/cache hoặc bắt đầu download lớn trước dự toán.
- ROS Humble import `rclpy` PASS; có cả Humble và Rolling, phải chọn Humble rõ ràng. Gazebo in version 11.10.2 nhưng command trả code 255; chưa coi simulator launch là PASS.
- Có đủ **{len(weights['required_files'])} / {len(weights['required_files'])}** file weights bắt buộc theo launcher; safetensors header đọc được, Depth-Anything weights-only load PASS. Local SHA-256 đã lưu, chưa đối chiếu checksum nguồn phát hành và chưa chạy full RoboRefer inference.
- **{inv['checkpoint_pass']}/{inv['checkpoint_manifests']}** checkpoint manifests khớp file size/SHA-256; V1 step 2.000 còn tồn tại và khớp lock cũ. Training-state được hash, chưa thử resume optimizer.
- Development cache **1.242/1.242** và Calibration cache **618/618** feature files khớp SHA-256, size, tensor shape. Chỉ dùng được khi model/preprocessing/input lineage tương thích; không tự tái sử dụng làm thí nghiệm adaptation mới.
- **{len(reuse['missing_tracked_dataset_artifacts'])}** tracked dataset artifacts đang thiếu; raw Gazebo development/calibration cũ không có trong `datasets/` hiện tại. Cần restore xác minh hoặc capture mới trước đánh giá với mask/metric GT.
- Không xóa dữ liệu, không thay đổi môi trường cài đặt, không mở final-test predictions.

## Việc còn lại để đóng G0 đầy đủ

1. Chạy một inference RoboRefer development để xác nhận toàn bộ loader/preprocessing/API, đo VRAM/latency.
2. Lập ngân sách dung lượng và thời gian B1/B2 từ resource smoke; chưa thể kết luận training feasible chỉ từ CUDA tensor smoke.
3. Xác minh simulator launch và camera/TF trong Humble; quyết định restore/capture Gazebo.

## Tái lập

```bash
.conda-roborefer/bin/python -s protocol/refspatial_wp0_inventory.py
```

Chi tiết: `environment.json`, `model_weight_inventory.json`, `checkpoint_integrity.json`, `feature_cache_integrity.json`, `artifact_reuse_matrix.json`, `storage_inventory.json`, `git_baseline.json`.
''')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
