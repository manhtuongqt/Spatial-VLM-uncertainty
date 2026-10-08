"""Read G0 evidence, verify provenance and publish a terminal engineering decision."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DAY = ROOT/'ketqua1/03_backbone_h_spatial/ngay_02'
RUN = DAY/'attempt_03_fp32_llm'
LOCK = ROOT/'protocol/MH_PCRAU_V3_G0_RUN_LOCK_R3.json'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def ref(p): return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def now(): return datetime.now(timezone.utc).isoformat()
def matches(r): return (ROOT/r['path']).is_file() and sha(ROOT/r['path'])==r['sha256']
def write(name,data,status='PASS'):
    obj={'schema_version':'1.0','method_id':'MH-PCRA-U-v3','namespace':'mh_pcrau_v3',
         'created_at_utc':now(),'status':status,'run_lock':ref(LOCK),**data}
    with (DAY/name).open('x') as f: json.dump(obj,f,indent=2,ensure_ascii=False); f.write('\n')
    return obj


def main():
    lock=read(LOCK)
    names=['G0_PREFLIGHT.json','G0_SOURCE_MAP.json','G0_TEST_REPORT.json','H_SPATIAL_TOKEN_AUDIT.json',
      'H_SPATIAL_PADDING_INVARIANCE.json','H_SPATIAL_DETERMINISM.json','H_SPATIAL_RELOAD.json',
      'H_SPATIAL_LEAKAGE_AUDIT.json','H_SPATIAL_RESOURCE_REPORT.json','H_SPATIAL_FORWARD_BACKWARD_SMOKE.json',
      'CACHE_SCHEMA.json','CACHE_ROUNDTRIP.json','G0_SOURCE_IMMUTABILITY.json']
    evidence={n:read(RUN/n) for n in names}
    checks={}
    checks['required_evidence_present']=len(evidence)==len(names)
    checks['locked_prerequisites_match']=all(matches(r) for r in lock['prerequisites'])
    checks['locked_code_matches']=all(matches(r) for r in lock['code'])
    checks['checkpoint_inventory_matches']=all(matches(r) for r in lock['model_inventory'])
    checks['planning_documents_unchanged']=all(matches(r) for r in lock['plan_sources'])
    checks['manifest_hashes_match']=matches(lock['dataset_manifest']) and matches(lock['inference_manifest'])
    checks['sample_file_hashes_match']=all(matches(s[k]) for s in lock['samples'] for k in ['rgb','depth','metric_depth'])
    checks['result_lock_references_match']=all(matches(e['run_lock']) for e in evidence.values() if 'run_lock' in e)
    checks['lock_precedes_forward']=lock['created_at_utc']<read(RUN/'G0_RUN_START.json')['created_at_utc']
    checks['reload_is_fresh_process']=read(RUN/'G0_RUN_START.json')['pid']!=read(RUN/'G0_RELOAD_START.json')['pid']
    checks['tolerances_unchanged']=lock['tolerances']==read(ROOT/'protocol/MH_PCRAU_V3_G0_RUN_LOCK.json')['tolerances']
    checks['budget_unchanged']=lock['resource']==read(ROOT/'protocol/MH_PCRAU_V3_G0_RUN_LOCK.json')['resource']
    checks['samples_unchanged']=lock['samples']==read(ROOT/'protocol/MH_PCRAU_V3_G0_RUN_LOCK.json')['samples']
    checks['earlier_runner_archives_match']=all(
      sha(p)==next(r['sha256'] for r in read(l)['code'] if r['path'].endswith('/run_g0_feasibility.py'))
      for p,l in [(DAY/'attempt_01_source/run_g0_feasibility.py',ROOT/'protocol/MH_PCRAU_V3_G0_RUN_LOCK.json'),
                   (DAY/'attempt_02_bf16/run_g0_feasibility.py',ROOT/'protocol/MH_PCRAU_V3_G0_RUN_LOCK_R2.json')])
    smoke=evidence['H_SPATIAL_FORWARD_BACKWARD_SMOKE.json']
    checks['zero_optimizer_steps']=smoke['optimizer_steps']==0
    checks['backbone_unchanged']=smoke['backbone_parameter_versions_unchanged']
    cache=evidence['CACHE_ROUNDTRIP.json']; schema=evidence['CACHE_SCHEMA.json']
    checks['cache_schema_exact']=set(cache['metadata'])==set(schema['required_fields'])
    checks['cache_file_hash_matches']=matches(cache['feature_file'])
    checks['all_evidence_schemas_match']=all(e['schema_version']=='1.0' and e['method_id']=='MH-PCRA-U-v3' for e in evidence.values())
    audit=write('DAY_02_CONSISTENCY_AUDIT.json',{'checks':checks,'checks_pass':sum(checks.values()),
      'checks_total':len(checks),'checks_fail':sum(not v for v in checks.values()),
      'evidence':[ref(RUN/n) for n in names],'finalizer':ref(Path(__file__))},'PASS' if all(checks.values()) else 'SMOKE_FAIL')
    failed=[n for n,e in evidence.items() if e['status']!='PASS']
    if not all(checks.values()): failed.append('DAY_02_CONSISTENCY_AUDIT.json')
    outcome='G0_PASS' if not failed else 'G0_STOP'
    ledger=write('GATE_LEDGER_DAY_02.json',{'previous_ledger':ref(ROOT/'ketqua1/00_quan_tri_khoa/ngay_01/GATE_LEDGER.json'),
      'work_packages':{'WP-V3.0':'FROZEN','WP-V3.1':'COMPLETE' if not failed else 'STOPPED'},
      'gates':{'G0':'PASS' if not failed else 'STOP','G1':'READY_NOT_RUN' if not failed else 'NOT_OPENED',
               'G2':'NOT_OPENED','G3':'NOT_OPENED','G4':'SEALED','G5':'SEALED','G6':'SEALED'},
      'hypotheses':{'H1':'NOT_TESTED','H2':'NOT_TESTED','H3':'NOT_TESTED'},
      'activity':{'engineering_attempts':3,'completed_main_forward_suites':2,'fresh_reload_processes':2,
                  'training_runs':0,'calibrator_fits':0,'test_runs':0,'robot_runs':0},
      'scope':'Development engineering evidence only; no trained prediction head or scientific accuracy result.'})
    decision=write('G0_DECISION.json',{'outcome':outcome,'failed_evidence':failed,
      'audit':ref(DAY/'DAY_02_CONSISTENCY_AUDIT.json'),'ledger':ref(DAY/'GATE_LEDGER_DAY_02.json'),
      'validated_precision':'BF16 RGB/depth towers and projectors, FP32 frozen LLM; eager attention; TF32 disabled',
      'scope_limitations':['Only two selected development RGB-D samples; not a scientific or full-dataset evaluation.',
        'Backward/save-load uses a temporary linear probe, not trained multi-head predictions.',
        'FP16 and all-BF16 attempts failed; passing precision policy must accompany future feature caching.'],
      'next_authorized_action':{'count':1,'action':'Day 3 G1 power/precision, schema, labels and split audit' if not failed
        else 'Repair failing development-only G0 checks under an append-only run lock'},
      'hypotheses':{'H1':'NOT_TESTED','H2':'NOT_TESTED','H3':'NOT_TESTED'},
      'sealed':['Calibration-v3','Test-IID','Test-OOD','robot-policy']},outcome)
    resource=evidence['H_SPATIAL_RESOURCE_REPORT.json']
    comparisons=evidence['H_SPATIAL_PADDING_INVARIANCE.json']['comparisons']
    rows=[f'| {key} | {c["max_abs"]:.9g} | {c["cosine"]:.12g} | {"PASS" if c["pass"] else "FAIL"} |' for key,c in comparisons.items()]
    report=f'''# Kết quả Ngày 02 — WP-V3.1/G0

Kết luận: **{outcome}**. Thời điểm: {decision['created_at_utc']}.

## Kết quả thực tế

- Adapter parameter-free đã triển khai, lấy hidden state lớp cuối từ prompt-only RGB-D.
- Shape mỗi sample `(1,1536)`; batch ghép `(2,1536)`.
- Bộ test synthetic: {evidence['G0_TEST_REPORT.json']['tests_run']} test.
- Audit nhất quán: {sum(checks.values())}/{len(checks)}.
- Forward RGB-D median: {resource['median_forward_seconds']:.6f} giây.
- Peak reserved lớn nhất của ba lần profile: {max(x['peak_reserved_mib'] for x in resource['measurements']):.3f} MiB / ngân sách 14.742 MiB.
- Tải model: {resource['load_seconds']:.6f} giây, báo cáo riêng với forward.
- Reload trong process mới: {evidence['H_SPATIAL_RELOAD.json']['status']}.
- Leakage static + paired synthetic metadata: {evidence['H_SPATIAL_LEAKAGE_AUDIT.json']['status']}.
- Probe backward/save-load: {smoke['status']}; optimizer steps = {smoke['optimizer_steps']}.
- Cache một vector: {cache['status']}; không bulk-cache dataset.

## Các lần chạy và thay đổi có bằng chứng

1. R1 dùng loader FP16 hiện có: dừng vì vector chứa NaN/Inf. Evidence giữ ở `G0_RUN_ERROR.json`.
2. R2 khôi phục BF16 theo checkpoint: finite, token, repeat, leakage, resource, reload PASS; mixed-batch FAIL, max abs tới 0,15625.
3. R3 giữ towers/projectors BF16 và nâng phép tính LLM lên FP32; giữ nguyên checkpoint, mẫu, seed, tolerance và budget. Run lock bổ sung được tạo trước forward R3.

Đây là điều chỉnh độ chính xác tính toán đã ghi trong chuỗi run lock. Không cập nhật weights bằng optimizer. R1/R2 và source runner tương ứng được lưu nguyên để truy vết.

## Token và dữ liệu

Hai sample train-UQ development là `gazebo_uq_v2_full_r3_train_uq_000` và `gazebo_uq_v2_full_r3_train_uq_002`, được chọn theo ID tăng dần trước forward.
Model nhận RGB, depth-view đã chuẩn bị theo pipeline và instruction. Metric-depth chỉ được hash làm provenance.
Đầu vào có assistant generation marker và không có nội dung assistant answer.
Adapter bỏ padding mỗi mẫu trước `_embed`, ghép lại embedding rồi dùng attention mask sau chèn RGB/depth và logical position IDs.

## Padding và batch invariance

Ngưỡng không thay đổi: max abs ≤ 0,002; cosine ≥ 0,99999; assert_close rtol=0,001, atol=0,002.

| So sánh | Max abs | Cosine | Kết quả |
|---|---:|---:|---|
{chr(10).join(rows)}

## Artifact bàn giao

Code:

- `RoboRefer/llava/model/spatial/hidden_state_adapter.py`
- `RoboRefer/tests/mh_pcrau_v3/test_h_spatial_adapter.py`
- `RoboRefer/scripts/mh_pcrau_v3/run_g0_feasibility.py`
- `RoboRefer/scripts/mh_pcrau_v3/finalize_g0.py`

Evidence cuối nằm trong `attempt_03_fp32_llm/`; `DAY_02_CONSISTENCY_AUDIT.json` liệt kê path và SHA-256 của từng file.
Các tài liệu kế hoạch Markdown gốc giữ nguyên vì đã được hash trong run lock. Trạng thái cuối xem `G0_DECISION.json` và báo cáo này.

| Artifact | SHA-256 |
|---|---|
| Run lock R3 | `{sha(LOCK)}` |
| Consistency audit | `{sha(DAY/'DAY_02_CONSISTENCY_AUDIT.json')}` |
| Gate ledger ngày 2 | `{sha(DAY/'GATE_LEDGER_DAY_02.json')}` |
| G0 decision | `{sha(DAY/'G0_DECISION.json')}` |

## Tái lập

Environment: `.conda-roborefer/bin/python`, `PYTHONNOUSERSITE=1`, offline model, seed 20260924.
Commands đã chạy từ `RoboRefer/`: `python scripts/mh_pcrau_v3/run_g0_feasibility.py prepare`, sau đó `run`, rồi `reload`.
Runner ghi file ở chế độ exclusive, nên chạy lại tại cùng thư mục bị chặn để tránh ghi đè evidence. Một lần tái lập mới cần output directory và append-only run lock mới.

## Phạm vi và bước tiếp theo

{decision['next_authorized_action']['action']}.
H1/H2/H3 vẫn NOT_TESTED. G0 chứng minh khả thi kỹ thuật trên hai mẫu; chưa đánh giá accuracy, uncertainty calibration hoặc an toàn robot. Backward probe không chứng minh bảy head đã triển khai/train.
Calibration-v3, Test-IID/OOD và robot-policy vẫn SEALED.
'''
    with (DAY/'KET_QUA_NGAY_02.md').open('x') as f: f.write(report)
    print(json.dumps({'outcome':outcome,'checks_pass':sum(checks.values()),'checks_total':len(checks),'failed':failed},indent=2))


if __name__=='__main__': main()
