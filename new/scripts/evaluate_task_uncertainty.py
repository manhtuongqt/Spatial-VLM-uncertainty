#!/usr/bin/env python3
"""Freeze tasks, fit calibration60, then reevaluate the previously observed IID."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,time,hashlib
from pathlib import Path
from functools import lru_cache
import cv2,numpy as np,torch
from scipy.special import logsumexp
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcrau.task_uncertainty import VERSION,TaskHeads
from pcrau.task_inference import TaskInference,task_vector,RISK_FEATURES,apply_task_calibrator
from pcrau.unified_inference import UnifiedInference,load_features
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.anchor_shadow_experiment import state_digest
from pcrau.calibration import fit_logistic,predict_risk
from pcrau.selective_experiment import choose_policy_threshold,predicted_answer,probability_metrics
from pcrau.live_iid_analysis import model_metrics,paired_changes,paired_family_bootstrap
from pcrau.utils import atomic_json,sha256_file,seed_everything

TRAIN_ROOT=Path('new/outputs/pcrau_task_uncertainty_20261007')
OUT=TRAIN_ROOT/'evaluation_r2'

def write_rows(path,rows):
    with path.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def fit(rows):
    x=np.stack([task_vector(r) for r in rows]);y=np.asarray([r['evaluation']['error_event'] for r in rows],float)
    families=[r['family_id'] for r in rows];unique=sorted(set(families),key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    assignment={f:i%5 for i,f in enumerate(unique)};oof=np.zeros(len(rows))
    for fold in range(5):
        held=np.asarray([assignment[f]==fold for f in families]);train=~held
        mean,scale=x[train].mean(0),x[train].std(0);scale[scale<1e-8]=1
        co,b=fit_logistic((x[train]-mean)/scale,y[train],.01);oof[held]=predict_risk((x[held]-mean)/scale,co,b)
    mean,scale=x.mean(0),x.std(0);scale[scale<1e-8]=1
    co,b=fit_logistic((x-mean)/scale,y,.01)
    cal={'version':VERSION,'method':'logistic60_family_crossfit_fit_all','feature_names':RISK_FEATURES,
         'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':co.tolist(),'intercept':b,
         'fit_samples':len(rows),'fit_families':len(unique),'family_folds':assignment,'l2':.01,
         'crossfit_metrics':probability_metrics(oof,y)}
    profile=choose_policy_threshold(oof,y,families,[predicted_answer(r)=='FOUND' for r in rows],.075,60)
    profile.update(policy='hard_found',threshold_selection='calibration_family_OOF_only',version=VERSION)
    return cal,profile,oof

def join_evaluator(ds,rows,raw,classes,supplement=None):
    xs=np.arange(32)*20+10;ys=np.arange(24)*20+10
    catalog=json.loads(Path('old/protocol/dataset_v2_1_shutdown_gate_capture_plan.json').read_text())['object_registry']
    idlabel={v['id']:v['label'] for v in catalog.values()}
    labelclass={int(v['label']):classes.index(v['semantic_class']) for v in catalog.values()}
    pairs={p['sample_id']:p for p in supplement['completion_pairs']} if supplement else {}
    evaluated=[];task_eval=[]
    for i,(e,row) in enumerate(zip(ds.entries,rows)):
        sup=e['supervision'];mask=cv2.imread(str(ds.layout.dataset_path(sup['target_mask_path'])),0)>0
        x,y=row['spatial']['map_pixel_xy'];inside=bool(mask[y,x]);truth=sup['answerability_state']
        evaluation={'answerability_state':truth,'target_exists':bool(mask.any()),'map_inside_target':inside,
                    'error_event':truth!='FOUND' or not inside}
        evaluated.append({**row,'variant':e['variant'],'evaluation':evaluation})
        tasks=row['task_uncertainty'];record=json.loads(ds.layout.dataset_path(e['record_path']).read_text())
        inst=cv2.imread(str(ds.layout.dataset_root/record['evaluator_only']['semantic_instance_labels']['path']),cv2.IMREAD_UNCHANGED)
        depth=np.load(ds.layout.dataset_root/record['sensor_evidence']['depth_metric']['path'])
        dep=float(depth[y,x]);valid=bool(np.isfinite(dep) and .1<dep<5)
        tc=labelclass.get(int(inst[y,x]),0)
        sp=np.asarray(tasks['semantic']['target']['mean_distribution']);pred=int(sp.argmax())
        wanted=tasks['semantic']['target']['requested_phrase'];wantedidx=classes.index(wanted) if wanted in classes else None
        entry={'sample_id':e['sample_id'],'family_id':e['family_id'],'variant':e['variant'],
            'target_hit':inside,'target_exists':bool(mask.any()),'semantic_identity_truth':classes[tc],
            'semantic_identity_correct':pred==tc,'semantic_identity_nll':float(-np.log(max(sp[tc],1e-12))),
            'semantic_matching_truth':bool(tc==wantedidx) if wantedidx is not None else None,
            'semantic_requested_probability':tasks['semantic']['target']['requested_class_probability'],
            'depth_reference_m':dep if valid else None,'depth_valid':valid,
            'depth_error_m':tasks['depth']['mean_m']-dep if valid else None,
            'depth_interval_hit':bool(tasks['depth']['lower_95_m']<=dep<=tasks['depth']['upper_95_m']) if valid else None,
            'spatial_target_MI':tasks['spatial']['target']['MI'],
            'semantic_target_MI':tasks['semantic']['target']['MI'],'completion_MI':tasks['occlusion_completion']['MI']}
        if valid:
            mu=raw['depth_mean'][:,i].astype(float);var=raw['depth_variance'][:,i].astype(float)
            entry['depth_mixture_nll']=float(-logsumexp(-.5*((dep-mu)**2/var+np.log(2*np.pi*var)))+np.log(20))
        anchor=row['verifiers']['P1']['anchor_peak_xy']
        if anchor is not None and len(sup['anchor_masks'])==1:
            am=cv2.imread(str(ds.layout.dataset_path(sup['anchor_masks'][0]['path'])),0)>0;ax,ay=anchor
            entry.update(anchor_visible=bool(am.any()),anchor_hit=bool(am[ay,ax]),
                semantic_anchor_truth=classes[labelclass.get(int(inst[ay,ax]),0)],
                semantic_anchor_correct=tasks['semantic']['anchor']['predicted_class']==classes[labelclass.get(int(inst[ay,ax]),0)])
        # Geometric reference from actual requested object masks, not FOUND proxy.
        sl=record['evaluator_only']['spatial_label'];cids=sl['candidate_target_ids'];aids=sl['anchor_ids']
        if tasks['relation']['supported'] and len(cids)==len(aids)==1 and all(k in idlabel for k in cids+aids):
            tm=inst==idlabel[cids[0]];am=inst==idlabel[aids[0]]
            if tm.any() and am.any():
                sign=1 if row['verifiers']['P1']['query']['predicate']=='right_of' else -1
                satisfied=bool(sign*(np.nonzero(tm)[1].mean()-np.nonzero(am)[1].mean())>12)
                pr=tasks['relation']['compatibility_mean']
                entry.update(relation_centroid_reference=satisfied,relation_brier=float((pr-satisfied)**2),
                    relation_prediction_correct=bool((pr>=.5)==satisfied),relation_MI=tasks['relation']['MI'])
        maskgrid=cv2.resize(mask.astype('float32'),(32,24),interpolation=cv2.INTER_AREA).flatten()
        completion=np.asarray(tasks['occlusion_completion']['mean_distribution']).flatten()
        if mask.any():entry['completion_observed_target_mass']=float(completion[maskgrid>0].sum())
        if e['sample_id'] in pairs:
            pair=pairs[e['sample_id']];lab=pair['reference_label_id_supervision_only']
            A=cv2.imread(pair['reference_semantic_labels_supervision_only']['path'],cv2.IMREAD_UNCHANGED)==lab
            B=inst==lab;hidden=(cv2.erode(A.astype('uint8'),np.ones((5,5),'uint8'))>0)&~B
            ref=cv2.resize(A.astype('float32'),(32,24),interpolation=cv2.INTER_AREA).flatten()
            hid=cv2.resize(hidden.astype('float32'),(32,24),interpolation=cv2.INTER_AREA).flatten()
            cx,cy=tasks['occlusion_completion']['map_xy'];base=np.asarray(row['spatial']['probability_grid']).flatten()
            entry.update(completion_reference_pair=True,meaningful_hidden=pair['hidden_core_fraction']>=.02,
                completion_reference_mass=float(completion[ref>0].sum()),baseline_reference_mass=float(base[ref>0].sum()),
                completion_hidden_core_mass=float(completion[hid>0].sum()),baseline_hidden_core_mass=float(base[hid>0].sum()),
                completion_reference_PIT=bool(A[cy,cx]),baseline_reference_PIT=bool(A[y,x]),
                hidden_core_pixels=pair['hidden_core_pixels'],reference_pixels=pair['reference_pixels'])
        task_eval.append(entry)
    return evaluated,task_eval

def summarize_task_evaluation(rows):
    depth=[r for r in rows if r['depth_valid']];relation=[r for r in rows if 'relation_brier' in r]
    anchors=[r for r in rows if r.get('anchor_visible')];pairs=[r for r in rows if r.get('completion_reference_pair')]
    hidden=[r for r in pairs if r['meaningful_hidden']]
    matching=[r for r in rows if r['semantic_matching_truth'] is not None]
    foreground=[r for r in rows if r['semantic_identity_truth']!='background']
    def avg(rs,k):return float(np.mean([r[k] for r in rs])) if rs else None
    return {'samples':len(rows),'semantic':{'node_identity_accuracy':avg(rows,'semantic_identity_correct'),
        'foreground_nodes':len(foreground),'foreground_identity_accuracy':avg(foreground,'semantic_identity_correct'),
        'node_identity_NLL':avg(rows,'semantic_identity_nll'),'noun_matching_samples':len(matching),
        'noun_matching_brier':float(np.mean([(r['semantic_requested_probability']-r['semantic_matching_truth'])**2 for r in matching])) if matching else None},
        'spatial':{'target_present':sum(r['target_exists'] for r in rows),'target_hits':sum(r['target_hit'] for r in rows),
            'visible_anchors':len(anchors),'anchor_hits':sum(r['anchor_hit'] for r in anchors)},
        'relation':{'centroid_reference_samples':len(relation),'accuracy':avg(relation,'relation_prediction_correct'),
            'brier':avg(relation,'relation_brier'),'reference_definition':'single_visible_requested_target_and_anchor_centroids_12px'},
        'depth':{'valid_predicted_node_samples':len(depth),'MAE_m':float(np.mean([abs(r['depth_error_m']) for r in depth])),
            'RMSE_m':float(np.sqrt(np.mean([r['depth_error_m']**2 for r in depth]))),'mixture_NLL':avg(depth,'depth_mixture_nll'),
            'interval_95_coverage':avg(depth,'depth_interval_hit'),'reference_is_sensor_measurement':True},
        'completion':{'eligible_pairs':len(pairs),'meaningful_hidden_pairs':len(hidden),
            'reference_PIT':avg(pairs,'completion_reference_PIT'),'baseline_reference_PIT':avg(pairs,'baseline_reference_PIT'),
            'reference_mass':avg(pairs,'completion_reference_mass'),'baseline_reference_mass':avg(pairs,'baseline_reference_mass'),
            'hidden_core_mass_mean_on_meaningful_pairs':avg(hidden,'completion_hidden_core_mass'),
            'baseline_hidden_core_mass_on_meaningful_pairs':avg(hidden,'baseline_hidden_core_mass'),
            'full_amodal_ground_truth':False,'IID_completion_reference_evaluation_performed':False}}

def main():
    assert (TRAIN_ROOT/'training_result.json').exists() and not OUT.exists()
    OUT.mkdir()
    torch.set_num_threads(4);seed_everything(24082026)
    baseline_path=Path('new/outputs/pcrau_unified_spatial_20261005_r2/bundle.json').resolve()
    pipeline=UnifiedInference.from_bundle(baseline_path)
    supplement=json.loads((TRAIN_ROOT/'data/supplement.json').read_text());classes=supplement['classes']
    heads=TaskHeads(len(classes)).cuda();heads.load_state_dict(load_file(str(TRAIN_ROOT/'best_task_heads.safetensors')))
    model=TaskInference(pipeline,heads,classes);initial=state_digest(pipeline.shadow.state_dict())
    source_files={p:{'path':str(Path(p).resolve()),'sha256':sha256_file(Path(p))} for p in [
        'new/src/pcrau/task_uncertainty.py','new/src/pcrau/task_inference.py','new/scripts/evaluate_task_uncertainty.py']}
    freeze={'version':VERSION,'baseline_bundle':{'path':str(baseline_path),'sha256':sha256_file(baseline_path)},
        'task_checkpoint':{'path':str((TRAIN_ROOT/'best_task_heads.safetensors').resolve()),'sha256':sha256_file(TRAIN_ROOT/'best_task_heads.safetensors')},
        'source_files':source_files,'classes':classes,'passes':20,'batch_size':20,'seed':24082026,
        'risk_feature_names':RISK_FEATURES,'protocol_sha256':sha256_file(Path('plan/S6_TASK_UNCERTAINTY_PROTOCOL_20261007.md')),
        'baseline_state':initial,'calibration_not_yet_fit':True}
    atomic_json(OUT/'neural_freeze.json',freeze)
    all_summaries={};batch_index=0
    for split,profile in [('train','development'),('dev','development'),('calibration','calibration'),('test_iid','test_iid')]:
        if split=='test_iid':
            assert (OUT/'bundle.json').exists();model=TaskInference.from_bundle(OUT/'bundle.json')
            assert sha256_file(OUT/'calibrator.json')==json.loads((OUT/'bundle.json').read_text())['calibrator']['sha256']
            atomic_json(OUT/'pre_IID_lock.json',{'bundle_sha256':sha256_file(OUT/'bundle.json'),
                'threshold':model.profile['threshold'],'IID_history':'previously_observed_reevaluation','no_fit_on_IID':True})
        ds=ArchivedPCRAUDataset(pipeline.config,split,profile=profile);directory=OUT/split;directory.mkdir(exist_ok=False)
        @lru_cache(maxsize=32)
        def feature(key):
            d=ds.features[key];p=ds.layout.feature_root/d['path'];assert sha256_file(p)==d['sha256'];return load_features(p)
        rows=[];arrays=[];times=[]
        for start in range(0,len(ds),20):
            es=ds.entries[start:start+20];fs=[feature(ds.sample_to_feature[e['sample_id']]) for e in es]
            f={k:torch.stack([v[k] for v in fs]) for k in fs[0]};prompts=[e['feature_input']['prompt'] for e in es]
            seed=24082026+batch_index;batch_index+=1;torch.cuda.synchronize();tick=time.perf_counter()
            observed,raw=model.predict(f,prompts,seed) if model.calibrator is not None else model.observe(f,prompts,seed)
            torch.cuda.synchronize();times.append((time.perf_counter()-tick)*1000)
            rows.extend({'sample_id':e['sample_id'],'family_id':e['family_id'],**r} for e,r in zip(es,observed));arrays.append(raw)
            if start%200==0:print(f'TASK_MC {split} {start+len(es)}/{len(ds)}',flush=True)
        raw={k:np.concatenate([a[k] for a in arrays],axis=1) for k in arrays[0]}
        write_rows(directory/'observable_predictions.jsonl',rows)
        np.savez_compressed(directory/'raw_MC.npz',sample_ids=np.asarray([r['sample_id'] for r in rows]),**raw)
        # Evaluator-only records/masks are opened after the split's observable export.
        evaluated,task_eval=join_evaluator(ds,rows,raw,classes,supplement if split in ('train','dev') else None)
        write_rows(directory/'task_evaluation.jsonl',task_eval)
        if split=='calibration':
            cal,policy,oof=fit(evaluated);atomic_json(OUT/'calibrator.json',cal);atomic_json(OUT/'profile.json',policy)
            bundle={**freeze,'calibrator':{'path':str((OUT/'calibrator.json').resolve()),'sha256':sha256_file(OUT/'calibrator.json')},
                'profile':{'path':str((OUT/'profile.json').resolve()),'sha256':sha256_file(OUT/'profile.json')},
                'neural_freeze_sha256':sha256_file(OUT/'neural_freeze.json')}
            atomic_json(OUT/'bundle.json',bundle)
            model.calibrator=cal;model.profile=policy
            evaluated=apply_task_calibrator(evaluated,cal,policy)
            np.save(directory/'oof_risk.npy',oof)
        write_rows(directory/'evaluator_predictions.jsonl',evaluated)
        task_summary=summarize_task_evaluation(task_eval)
        summary={'tasks':task_summary,'batch20_feature_cache_pipeline_MC_latency_median_ms':float(np.median(times))}
        if model.calibrator is not None:
            primary_risk=np.asarray([r['decision']['risk'] for r in evaluated])
            reference=[{**r,'decision':r.get('reference_live44_decision',r['decision'])} for r in evaluated]
            reference_risk=np.asarray([r['decision']['risk'] for r in reference])
            summary['primary']=model_metrics(evaluated,primary_risk,model.profile['threshold'] if model.profile['threshold'] is not None else -1)
            summary['reference_live44']=model_metrics(reference,reference_risk,pipeline.profile['threshold'])
            summary['paired'],changes=paired_changes(evaluated,reference);write_rows(directory/'changed_decisions.jsonl',changes)
            summary['family_bootstrap']=paired_family_bootstrap(evaluated,reference)
        atomic_json(directory/'summary.json',summary);all_summaries[split]=summary
        del arrays,raw,rows,evaluated,task_eval
    assert state_digest(pipeline.shadow.state_dict())==initial;pipeline.shadow._check_frozen()
    protected=json.loads((TRAIN_ROOT/'training_lock.json').read_text())['protected_files']
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    atomic_json(OUT/'summary.json',{'status':'COMPLETE','version':VERSION,'splits':all_summaries,
        'baseline_unchanged':True,'original_protected_files_unchanged':len(protected),
        'active_profile_promoted':False,'full_amodal_or_five_causal_decomposition_claimed':False})
    atomic_json(OUT/'RUN_STATUS.json',{'status':'COMPLETE_REAL_TASK_MC_CALIBRATION_OLD_IID_REEVALUATION'})
    print(json.dumps({'status':'COMPLETE','IID':all_summaries['test_iid']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
