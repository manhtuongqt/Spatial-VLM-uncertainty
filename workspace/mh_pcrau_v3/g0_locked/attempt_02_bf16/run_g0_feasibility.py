"""Reproducible G0-only runner. Each phase writes new evidence, never trains."""
import argparse
import ast
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[3]
VLM = ROOT / 'RoboRefer'
DAY_BASE = ROOT / 'ketqua1/03_backbone_h_spatial/ngay_02'
DAY = DAY_BASE / 'attempt_02_bf16'
DAY1 = ROOT / 'ketqua1/00_quan_tri_khoa/ngay_01'
MODEL = VLM / 'models/RoboRefer-2B-SFT'
LOCK = ROOT / 'protocol/MH_PCRAU_V3_G0_RUN_LOCK_R2.json'
sys.path.insert(0, str(VLM))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def digest(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def reference(path): return {'path': str(Path(path).relative_to(ROOT)), 'sha256': sha(path)}
def write(name, data, status='PASS'):
    document = {'schema_version': '1.0', 'method_id': 'MH-PCRA-U-v3', 'namespace': 'mh_pcrau_v3',
                'created_at_utc': now(), 'status': status, **data}
    if LOCK.exists(): document['run_lock'] = reference(LOCK)
    with (DAY / name).open('x') as f: json.dump(document, f, indent=2, ensure_ascii=False); f.write('\n')
    print(name, status, flush=True)
    return document


def setup():
    assert os.environ.get('PYTHONNOUSERSITE') == '1', 'Isolated interpreter required'
    import torch
    torch.manual_seed(20260924)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def prepare():
    assert not LOCK.exists(), 'Never overwrite a frozen G0 lock'
    DAY.mkdir(exist_ok=True)
    expected = {
        ROOT/'protocol/MH_PCRAU_V3_HYPOTHESIS_LOCK.json': '49fc6ae3c8089d6cdcfa10c2c3b3d8209daa9f25cd817f70bbb2df7cbbdda7ae',
        DAY1/'DAY_01_DECISION.json': 'f9ee2c3ef31bbcccc29b36e02200c7ef882a865537068b66da3c6a3f1a5654c0',
        DAY1/'MH_PCRAU_V3_ARCHITECTURE_IDENTITY.json': '8fd053dd494625d0d69f6465e531adca59f0c2b09cc3d47789e0c91a59c5fbd5',
        DAY1/'DATA_ACCESS_BOUNDARY.json': '1fedd3b54e2f425ad3655ee69bbe2f2c458796ef03dfd6ba31e1c00c91df0df2',
        DAY1/'ENVIRONMENT_LOCK.json': '2265f118b3d617d57c1030c8d929c951716fbc9fa88d58da45a3303927de302d',
    }
    for p, h in expected.items(): assert sha(p) == h, f'Prerequisite changed: {p}'
    torch = setup()
    import torchvision, transformers, peft, bitsandbytes, llava
    assert torch.cuda.is_available()
    smoke = torch.ones(8, device='cuda').sum().item()
    assert smoke == 8
    versions = {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'transformers', 'peft', 'bitsandbytes']}
    write('G0_PREFLIGHT.json', {'prerequisites': [reference(p) for p in expected],
          'python': sys.executable, 'python_version': platform.python_version(), 'versions': versions,
          'cuda_smoke': smoke, 'gpu': torch.cuda.get_device_name(),
          'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'worktree': subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True),
          'roborefer_worktree': subprocess.check_output(['git','status','--porcelain=v1'],cwd=VLM,text=True)})
    files = [VLM/p for p in ['llava/model/spatial/__init__.py','llava/model/spatial/hidden_state_adapter.py',
          'scripts/mh_pcrau_v3/run_g0_feasibility.py','tests/mh_pcrau_v3/test_h_spatial_adapter.py',
          'llava/model/llava_arch.py','llava/model/language_model/llava_llama.py','llava/model/builder.py',
          'llava/model/language_model/builder.py','llava/mm_utils.py','llava/utils/tokenizer.py',
          'llava/utils/media.py','llava/conversation.py','llava/constants.py']]
    write('G0_SOURCE_MAP.json', {'files': [reference(p) for p in files],
          'embedding_strategy': 'Unpad each row before upstream _embed; batch expanded embeddings with logical position_ids.',
          'existing_sources_modified': False})
    suite = unittest.defaultTestLoader.discover(str(VLM/'tests/mh_pcrau_v3'), pattern='test_*.py')
    stream = io.StringIO(); started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write('G0_TEST_REPORT.json', {'tests_run':result.testsRun,'failures':len(result.failures),
          'errors':len(result.errors),'log':stream.getvalue(),'duration_seconds':time.perf_counter()-started,
          'test_source':reference(files[3])}, 'PASS' if result.wasSuccessful() else 'SMOKE_FAIL')
    assert result.wasSuccessful(), stream.getvalue()
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(MODEL/'llm'), local_files_only=True)
    dataset = ROOT/'datasets/Gazebo_train_uq_v2_full_r3'
    manifest = dataset/'inference_manifest.jsonl'
    rows = sorted([json.loads(line) for line in manifest.read_text().splitlines()], key=lambda r:r['sample_id'])
    selected = []; exclusions = []
    for row in rows:
        if row['split'] != 'train_uq': continue
        rgb, depth = dataset/row['image'], dataset/row['depth']
        if not rgb.is_file() or not depth.is_file() or not row['instruction'].strip():
            exclusions.append({'sample_id':row['sample_id'],'reason':'missing_media_or_empty_prompt'}); continue
        length = len(tokenizer(row['instruction']).input_ids)
        if selected and selected[0]['instruction_tokens'] == length: continue
        selected.append({'sample_id':row['sample_id'], 'rgb':reference(rgb),'depth':reference(depth),
                         'metric_depth':reference(dataset/row['metric_depth']),
                         'instruction':row['instruction'],'instruction_sha256':digest(row['instruction']),
                         'instruction_tokens':length})
        if len(selected)==2: break
    assert len(selected)==2
    model_inventory = [reference(p) for p in sorted(MODEL.rglob('*')) if p.is_file() and '.cache' not in p.parts]
    locked = {'schema_version':'1.0','method_id':'MH-PCRA-U-v3','namespace':'mh_pcrau_v3',
      'status':'FROZEN_BEFORE_G0_GPU_RUN','created_at_utc':now(),'seed':20260924,
      'prerequisites':[reference(p) for p in expected], 'code': [reference(p) for p in files],
      'model_inventory':model_inventory,'model_inventory_sha256':digest(model_inventory),
      'dataset_manifest':reference(dataset/'manifest.json'),'inference_manifest':reference(manifest),
      'selection_rule':'Gazebo_train_uq_v2_full_r3 train_uq only; ascending sample_id; first valid RGB-D record, then first different instruction token length. Chosen before model forward.',
      'depth_contract':'Existing depth_view.png is the canonical encoder representation; linked metric_depth records provenance, not an additional model input.',
      'samples':selected, 'exclusions':exclusions,
      'model_load':{'api':'LlavaLlamaModel with local AutoConfig and explicit BF16','attn_implementation':'eager','device_map':'cuda:0','quantization':None,
                    'dtype':'torch.bfloat16, matching original checkpoint config'},
      'deviation':{'previous_lock':reference(ROOT/'protocol/MH_PCRAU_V3_G0_RUN_LOCK.json'),
                   'failure_evidence':reference(DAY_BASE/'G0_RUN_ERROR.json'),
                   'previous_runner_archive':reference(DAY_BASE/'attempt_01_source/run_g0_feasibility.py'),
                   'reason':'FP16 canonical loader produced non-finite h_spatial. Restore original BF16 checkpoint dtype before retry. No sample, tolerance or budget changes.'},
      'tolerances':{'repeat':{'cosine_min':0.999999,'max_abs':1e-5},
                    'invariance':{'cosine_min':0.99999,'max_abs':0.002,'rtol':0.001,'atol':0.002}},
      'resource':{'peak_reserved_mib_max':14742,'forward_seconds_max':30,'warmup':1,'measured_repeats':3},
      'scope':'G0 only; no training/capture/calibration/test/robot; one temporary detached linear probe with one backward and zero optimizer steps.',
      'plan_sources':[reference(p) for p in sorted(DAY_BASE.glob('*.md'))]}
    with LOCK.open('x') as f: json.dump(locked,f,indent=2); f.write('\n')
    print('LOCK_FROZEN', sha(LOCK), flush=True)


def verify_lock():
    lock = read(LOCK)
    for ref in lock['prerequisites']+lock['code']+lock['model_inventory']:
        assert sha(ROOT/ref['path'])==ref['sha256'], f'Hash mismatch: {ref["path"]}'
    for sample in lock['samples']:
        for key in ['rgb','depth','metric_depth']:
            assert sha(ROOT/sample[key]['path'])==sample[key]['sha256']
    return lock


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def compare(a,b,kind='repeat'):
    import torch
    tolerance = read(LOCK)['tolerances'][kind]
    a,b=a.double().flatten(),b.double().flatten()
    maximum=float((a-b).abs().max()); cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0))
    close = torch.allclose(a,b,rtol=tolerance.get('rtol',0),atol=tolerance.get('atol',tolerance['max_abs']))
    return {'max_abs':maximum,'cosine':cosine,'assert_close':bool(close),
            'pass':maximum<=tolerance['max_abs'] and cosine>=tolerance['cosine_min'] and bool(close)}


def load_model(lock):
    import torch
    from transformers import AutoConfig
    from llava.model.language_model.llava_llama import LlavaLlamaModel
    from llava.constants import DEFAULT_DEPTH_TOKEN
    start=time.perf_counter()
    config=AutoConfig.from_pretrained(str(MODEL),local_files_only=True)
    config.resume_path=str(MODEL)
    config.model_dtype='torch.bfloat16'
    model=LlavaLlamaModel(config=config,device_map='cuda:0',attn_implementation='eager',low_cpu_mem_usage=True)
    model.tokenizer.media_token_ids['depth']=model.tokenizer.convert_tokens_to_ids(DEFAULT_DEPTH_TOKEN)
    model.tokenizer.media_tokens['depth']=DEFAULT_DEPTH_TOKEN
    model.to(device='cuda:0',dtype=torch.bfloat16)
    model.eval(); model.requires_grad_(False)
    return model,time.perf_counter()-start


def run(reload_only=False):
    torch=setup(); lock=verify_lock()
    from llava.model.spatial.hidden_state_adapter import prepare_rgbd_prompt,collate_prompts,extract_h_spatial
    write('G0_RELOAD_START.json' if reload_only else 'G0_RUN_START.json', {'pid':os.getpid(),'phase':'reload' if reload_only else 'main'})
    model,load_seconds=load_model(lock)
    prompts=[prepare_rgbd_prompt(model,rgb_path=ROOT/s['rgb']['path'],depth_path=ROOT/s['depth']['path'],
                                instruction=s['instruction']) for s in lock['samples']]
    assert prompts[0].input_ids.numel()!=prompts[1].input_ids.numel()
    assert len(model.tokenizer)==model.llm.get_input_embeddings().weight.shape[0], 'No new vocabulary allowed'
    def forward(batch, side='right', extra=0):
        model.tokenizer.padding_side=side
        ids,media,mask=collate_prompts(batch,model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,side,extra)
        return extract_h_spatial(model,ids,media,mask)
    if reload_only:
        actual=forward(prompts[:1]).h_spatial.cpu()
        expected=torch.load(DAY/'h_spatial_sample.pt',map_location='cpu',weights_only=True)
        comparison=compare(actual,expected)
        write('H_SPATIAL_RELOAD.json',{'comparison':comparison,'load_seconds':load_seconds,
              'reference':reference(DAY/'h_spatial_sample.pt'),'process_id':os.getpid()},'PASS' if comparison['pass'] else 'SMOKE_FAIL')
        return
    versions_before={n:p._version for n,p in model.named_parameters()}
    tokenizer_hash=digest(model.tokenizer.get_vocab())
    base=forward(prompts[:1]) # warmup
    measured=[]; repeats=[]
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.perf_counter()
        out=forward(prompts[:1]); torch.cuda.synchronize()
        measured.append({'seconds':time.perf_counter()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                         'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
        repeats.append(out.h_spatial.cpu())
    reference_vector=repeats[0]
    torch.save(reference_vector,DAY/'h_spatial_sample.pt')
    records=[]
    for i,prompt in enumerate(prompts):
        solo=base if i==0 else forward([prompt])
        token_ids=prompt.input_ids.tolist()
        records.append({'sample_id':lock['samples'][i]['sample_id'],'prompt_text':prompt.prompt_text,
          'input_ids':token_ids,'last_text_token_id':token_ids[-1],
          'media_positions':{name:[i for i,t in enumerate(token_ids) if t==tid] for name,tid in model.tokenizer.media_token_ids.items()},
          'text_length':len(token_ids),'expanded_length':int(solo.post_insertion_sequence_lengths[0]),
          'last_prompt_index':int(solo.last_prompt_indices[0]),'shape':list(solo.h_spatial.shape),
          'final_index_valid':bool(solo.attention_mask[0,solo.last_prompt_indices[0]]),
          'generation_marker_present':prompt.prompt_text.endswith('<|im_start|>assistant\n'),
          'assistant_answer_present':False})
    token_pass=all(r['generation_marker_present'] and r['final_index_valid'] and r['shape']==[1,1536]
                   and r['last_prompt_index']==r['expanded_length']-1 for r in records)
    write('H_SPATIAL_TOKEN_AUDIT.json',{'records':records,'tokenizer_inventory_sha256':tokenizer_hash,
          'chat_template_sha256':digest(model.tokenizer.chat_template),'new_vocabulary_tokens':0,
          'labels':None,'hidden_layer':-1,'packing':False,'dtype':str(base.h_spatial.dtype)},'PASS' if token_pass else 'SMOKE_FAIL')
    pairs={}
    # Both samples: input left/right padding and a real mixed-length LLM batch.
    solo=[forward([p]).h_spatial.cpu() for p in prompts]
    for side in ['left','right']:
        mixed=forward(prompts,side,7).h_spatial.cpu()
        for i,p in enumerate(prompts):
            padded=forward([p],side,7).h_spatial.cpu()
            pairs[f'{side}_solo_{i}']=compare(solo[i],padded,'invariance')
            pairs[f'{side}_mixed_{i}']=compare(solo[i],mixed[i:i+1],'invariance')
    write('H_SPATIAL_PADDING_INVARIANCE.json',{'comparisons':pairs,'mixed_batch_size':2},
          'PASS' if all(c['pass'] for c in pairs.values()) else 'SMOKE_FAIL')
    repeat_comparisons=[compare(reference_vector,v) for v in repeats]
    write('H_SPATIAL_DETERMINISM.json',{'comparisons':repeat_comparisons,'fresh_process_evidence':'H_SPATIAL_RELOAD.json'},
          'PASS' if all(c['pass'] for c in repeat_comparisons) else 'SMOKE_FAIL')
    # A strict projection of a contaminated record exercises the actual input builder.
    s=lock['samples'][0]
    raw={'image':s['rgb']['path'],'depth':s['depth']['path'],'instruction':s['instruction']}
    changed={**raw, 'target_uv':[0.99,0.01],'target_xyz':[9,9,9], 'answer':'LEAK_SENTINEL',
      'answerability':'ABSENT','relation':'LEAK_SENTINEL','reasoning':'LEAK_SENTINEL','source':'LEAK_SENTINEL',
      'safe_to_execute':False,'evaluator_masks':[999],'split':'LEAK_SENTINEL','family_id':'LEAK_SENTINEL',
      'layout':'LEAK_SENTINEL','scene_id':'LEAK_SENTINEL','signature':'LEAK_SENTINEL'}
    def project(record):
        return prepare_rgbd_prompt(model,rgb_path=ROOT/record['image'],depth_path=ROOT/record['depth'],instruction=record['instruction'])
    original=project(raw); mutated=project(changed)
    def input_hash(p):
        return digest({'tokens':tensor_hash(p.input_ids), 'media':{k:[tensor_hash(v) for v in vs] for k,vs in p.media.items()}})
    leakage_comparison=compare(reference_vector,forward([mutated]).h_spatial.cpu())
    adapter=VLM/'llava/model/spatial/hidden_state_adapter.py'; tree=ast.parse(adapter.read_text())
    forbidden={'target_uv','target_xyz','answerability','safe_to_execute','evaluator_masks','family_id','layout','scene_id'}
    literals={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    static_pass=not forbidden.intersection(literals)
    leakage_pass=input_hash(original)==input_hash(mutated) and leakage_comparison['pass'] and static_pass
    write('H_SPATIAL_LEAKAGE_AUDIT.json',{'static_literal_audit':static_pass,'adapter':reference(adapter),
          'mutated_fields':sorted(set(changed)-set(raw)),'original_input_sha256':input_hash(original),
          'mutated_input_sha256':input_hash(mutated),'comparison':leakage_comparison,
          'scope':'Synthetic oracle perturbations on selected development RGB-D input; excludes scientific labels.'},'PASS' if leakage_pass else 'SMOKE_FAIL')
    resource_pass=all(m['seconds']<=30 and m['peak_reserved_mib']<=14742 for m in measured)
    write('H_SPATIAL_RESOURCE_REPORT.json',{'load_seconds':load_seconds,'warmup_count':1,'measurements':measured,
          'median_forward_seconds':statistics.median(m['seconds'] for m in measured),
          'gpu':torch.cuda.get_device_name(),'total_vram_mib':torch.cuda.get_device_properties(0).total_memory/2**20,
          'batch_size':1,'dtype':str(base.h_spatial.dtype),'oom':False,
          'llm_attention':model.llm.config._attn_implementation},'PASS' if resource_pass else 'SMOKE_FAIL')
    with tempfile.TemporaryDirectory(prefix='mh_pcrau_g0_probe_') as temp:
        feature=reference_vector.clone().float(); probe=torch.nn.Linear(1536,1)
        pred=probe(feature); loss=pred.square().mean(); loss.backward()
        finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in probe.parameters())
        probe_path=Path(temp)/'probe.pt'; torch.save(probe.state_dict(),probe_path)
        restored=torch.nn.Linear(1536,1); restored.load_state_dict(torch.load(probe_path,weights_only=True))
        equal=torch.equal(pred.detach(),restored(feature).detach())
    unchanged=all(versions_before[n]==p._version and p.grad is None for n,p in model.named_parameters())
    write('H_SPATIAL_FORWARD_BACKWARD_SMOKE.json',{'probe':'temporary CPU float32 Linear(1536,1)',
          'loss':float(loss.detach()),'finite_gradients':finite,'save_load_exact':equal,
          'backward_calls':1,'optimizer_steps':0,'backbone_parameter_versions_unchanged':unchanged,
          'probe_directory_removed':not Path(temp).exists()},'PASS' if finite and equal and unchanged else 'SMOKE_FAIL')
    required=['schema_version','method_id','namespace','sample_id','model_inventory_sha256','hypothesis_lock_sha256',
      'g0_run_lock_sha256','code_sha256','prompt_template_sha256','tokenizer_inventory_sha256','preprocessing_sha256',
      'rgb_sha256','depth_sha256','instruction_sha256','h_spatial_shape','dtype','vector_sha256',
      'last_prompt_index','post_insertion_sequence_length','created_at_utc']
    write('CACHE_SCHEMA.json',{'required_fields':required,'additional_fields_allowed':False,
          'shape':[1,1536],'scope':'One-vector smoke only; labels stored separately in future G1/G2.'})
    metadata={'schema_version':'1.0','method_id':'MH-PCRA-U-v3','namespace':'mh_pcrau_v3','sample_id':s['sample_id'],
      'model_inventory_sha256':lock['model_inventory_sha256'],'hypothesis_lock_sha256':lock['prerequisites'][0]['sha256'],
      'g0_run_lock_sha256':sha(LOCK),'code_sha256':sha(adapter),'prompt_template_sha256':digest(model.tokenizer.chat_template),
      'tokenizer_inventory_sha256':tokenizer_hash,'preprocessing_sha256':sha(VLM/'llava/mm_utils.py'),
      'rgb_sha256':s['rgb']['sha256'],'depth_sha256':s['depth']['sha256'],'instruction_sha256':s['instruction_sha256'],
      'h_spatial_shape':list(reference_vector.shape),'dtype':str(reference_vector.dtype),'vector_sha256':tensor_hash(reference_vector),
      'last_prompt_index':int(base.last_prompt_indices[0]),'post_insertion_sequence_length':int(base.post_insertion_sequence_lengths[0]),
      'created_at_utc':now()}
    roundtrip=torch.load(DAY/'h_spatial_sample.pt',weights_only=True)
    write('CACHE_ROUNDTRIP.json',{'metadata':metadata,'feature_file':reference(DAY/'h_spatial_sample.pt'),
          'bitwise_equal':torch.equal(reference_vector,roundtrip),'vector_hash_equal':tensor_hash(roundtrip)==metadata['vector_sha256']},
          'PASS' if torch.equal(reference_vector,roundtrip) else 'SMOKE_FAIL')
    verify_lock()
    write('G0_SOURCE_IMMUTABILITY.json',{'model_file_count':len(lock['model_inventory']),
          'all_locked_file_hashes_match':True,'backbone_versions_unchanged':unchanged},'PASS' if unchanged else 'SMOKE_FAIL')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['prepare','run','reload'])
    args=parser.parse_args()
    try:
        if args.phase=='prepare': prepare()
        else: run(reload_only=args.phase=='reload')
    except Exception as error:
        write(f'G0_{args.phase.upper()}_ERROR.json',{'error':str(error),'traceback':traceback.format_exc()},'BLOCKED')
        raise
