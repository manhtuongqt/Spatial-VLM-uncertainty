#!/usr/bin/env python3
"""Three-stage development experiment: train mining, freeze, split calibration."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from pcrau.answerability_evidence import AnswerabilityEvidenceAdapter
from pcrau.checkpoint import load_model_checkpoint, save_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset, ANSWER_CLASSES
from pcrau.engine import autocast_context, evaluate
from pcrau.losses import class_weights
from pcrau.model import PCRAUTargetV2
from pcrau.selective_experiment import (
    apply_risk, choose_policy_threshold, fit_risk, mine_hard_examples, policy_result,
    predicted_answer, probability_metrics, weighted_answer_loss,
)
from pcrau.utils import atomic_json, load_config, runtime_info, seed_everything, sha256_file, workspace_path
from experiment_answerability_head import answer_metrics, make_loader, selection_key
from experiment_answerability_evidence import evidence_eligible, predict


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def capture(base, loader, dataset, device, config, weights, source_weights, output_dir):
    inputs = []
    hook = base.answer_head.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().cpu()))
    try:
        metrics, rows = evaluate(base, loader, device, config, weights, source_weights)
    finally:
        hook.remove()
    output_dir.mkdir()
    atomic_json(output_dir/"metrics.json", metrics)
    write_rows(output_dir/"predictions.jsonl", rows)
    observed = torch.tensor([list(row["observable_answerability_evidence"].values()) for row in rows])
    feature = torch.cat(inputs).float()
    answer = torch.tensor([[row["answerability_probabilities"][name] for name in ANSWER_CLASSES] for row in rows])
    # Preserve actual V2 logits rather than log(probability), for cached residual parity.
    logits = []
    base.answer_head.eval()
    start = 0
    ranges = []
    with torch.no_grad():
        for inp in inputs:
            with autocast_context(device, config["optimization"]):
                logits.append(base.answer_head(inp.to(device)).float().cpu())
            ranges.append((start, start+len(inp)))
            start += len(inp)
    logits = torch.cat(logits)
    torch.testing.assert_close(logits.softmax(-1), answer, atol=1e-5, rtol=1e-5)
    cache = {"features": feature, "evidence": torch.cat([observed, feature], -1), "baseline_logits": logits,
             "truth": torch.tensor([row["evaluation"]["answerability_index"] for row in rows]),
             "sample_ids": [row["sample_id"] for row in rows], "families": [row["family_id"] for row in rows],
             "variants": [row["variant"] for row in rows], "batch_ranges": ranges,
             "lookup": {row["sample_id"]:i for i,row in enumerate(rows)}}
    assert set(cache["sample_ids"]) == {entry["sample_id"] for entry in dataset.entries}
    return metrics, rows, cache


def update(adapter, optimizer, indices, cache, device, config, class_weight, example_weight):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, config["optimization"]):
        logits = cache["baseline_logits"][indices].to(device) + adapter(cache["evidence"][indices].to(device)).float()
        loss = weighted_answer_loss(logits, cache["truth"][indices].to(device), class_weight,
                                    example_weight[indices].to(device))
    if not torch.isfinite(loss):
        raise ValueError("Non-finite hard example loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
    if not torch.isfinite(norm):
        raise ValueError("Non-finite hard example gradient")
    optimizer.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    base_root = workspace_path("new/outputs/pcrau_target_v2_full_seed_24082026")
    original_config = base_root/"config.json"
    original_checkpoint = base_root/"checkpoints/best/model.safetensors"
    root = workspace_path("new/outputs")/args.run_name
    if root.parent != workspace_path("new/outputs"):
        raise ValueError("Run name must be one directory name")
    root.mkdir(exist_ok=False)
    config = load_config(original_config)
    config["model"]["export_answerability_evidence"] = True
    base_config = deepcopy(config)
    config["model"]["answerability_evidence_adapter"] = {"hidden_dim":64,"dropout":0.3,"include_context":True}
    config["experiment_id"] = "hard_cases_selective_development_only"
    config["experiment"] = {"epochs":25,"learning_rate":3e-4,"weight_decay":1e-3,
                            "hard_positive_weight":2.0,"hard_negative_weight":3.0,
                            "risk_targets":[0.05,0.075,0.1],"calibration_family_partitions":[100,60,40],
                            "split_seed":20261003,"default_profile":"risk_only_0.05"}
    atomic_json(root/"config.json",config)
    config_hash = sha256_file(root/"config.json")
    protected=[original_checkpoint,original_config,base_root/"evaluation/calibration/calibrator.json",
               workspace_path("new/hinhanh/19_answerability_confusion/fig19_answerability_confusion_test_iid.png")]
    hashes={str(path):sha256_file(path) for path in protected}
    atomic_json(root/"run.json",{"status":"DEVELOPMENT_ONLY","runtime":runtime_info(),"source_hashes":hashes,
                "protocol_sha256":sha256_file(workspace_path("new/docs/EXPERIMENT_HARD_CASES_SELECTIVE.md")),
                "script_sha256":sha256_file(Path(__file__)),"test_splits_accessed":False,
                "prior_test_inspection_informed_research_question":True})
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=ArchivedPCRAUDataset(base_config,"train")
    dev=ArchivedPCRAUDataset(base_config,"dev")
    assert (len(train),len(dev))==(1600,400)
    assert {e["family_id"] for e in train.entries}.isdisjoint(e["family_id"] for e in dev.entries)
    train_loader,_=make_loader(train,base_config,False)
    dev_loader,_=make_loader(dev,base_config,False)
    _,sampler=make_loader(train,base_config,True)
    base=PCRAUTargetV2(base_config).to(device).eval().requires_grad_(False)
    load_model_checkpoint(base,original_checkpoint,sha256_file(original_config))
    weights,source_weights=class_weights(train,device)
    print('STAGE 1: train/dev extraction and train-only mining',flush=True)
    train_metrics,train_rows,train_cache=capture(base,train_loader,train,device,base_config,weights,source_weights,root/"baseline_train")
    dev_metrics,dev_rows,dev_cache=capture(base,dev_loader,dev,device,base_config,weights,source_weights,root/"baseline_dev")
    baseline=answer_metrics(dev_cache["truth"].tolist(),dev_cache["baseline_logits"].argmax(-1).tolist())
    assert baseline["confusion_matrix"]==[[140,0,3,25],[0,42,0,0],[4,0,34,15],[14,0,13,110]]
    proxy,proxy_crossfit=fit_risk(train_rows)
    proxy_error=np.asarray([r["evaluation"]["error_event"] for r in train_rows])
    proxy_threshold=choose_policy_threshold(proxy_crossfit,proxy_error,train_cache["families"],
                                            np.ones(len(train_rows),bool),0.05,60)
    positive,negative,localization=mine_hard_examples(train_rows,proxy_crossfit,proxy_threshold["threshold"])
    mining_rows=[{"sample_id":r["sample_id"],"family_id":r["family_id"],"truth":r["evaluation"]["answerability_state"],
                  "predicted":predicted_answer(r),"map_inside_target":r["evaluation"]["map_inside_target"],
                  "train_crossfit_proxy_risk":float(proxy_crossfit[i]),"hard_positive":bool(positive[i]),
                  "hard_negative":bool(negative[i]),"localization_failure_found":bool(localization[i])}
                 for i,r in enumerate(train_rows)]
    atomic_json(root/"mining.json",{"proxy_calibrator":proxy,"proxy_threshold":proxy_threshold,
                "hard_positive":int(positive.sum()),"hard_negative":int(negative.sum()),
                "localization_failure_found":int(localization.sum()),"calibration_accessed":False})
    write_rows(root/"mined_samples.jsonl",mining_rows)
    print(json.dumps({"hard_positive":int(positive.sum()),"hard_negative":int(negative.sum()),
                      "location_failures_not_relabelled":int(localization.sum())}),flush=True)
    model=PCRAUTargetV2(config).to(device).eval().requires_grad_(False)
    missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
    assert not unexpected and all(name.startswith("answerability_adapter.") for name in missing)
    initial=AnswerabilityEvidenceAdapter(64,0.3,768).to(device)
    initial.fit_standardization(train_cache["evidence"])
    model.answerability_adapter.load_state_dict(initial.state_dict())
    global_best,selected_arm,selected_epoch=baseline,"v2_fallback",-1
    candidate_states={}
    training_results={}
    for arm in ("control","hard_mining"):
        arm_root=root/arm;arm_root.mkdir()
        seed_everything(int(config["seed"]))
        adapter=deepcopy(initial).to(device).train().requires_grad_(True)
        example_weight=torch.ones(1600)
        if arm=="hard_mining":
            example_weight[torch.from_numpy(positive)]=2.0
            example_weight[torch.from_numpy(negative)]=3.0
        trial=deepcopy(adapter)
        trial_optimizer=torch.optim.AdamW(trial.parameters(),lr=3e-4,weight_decay=1e-3)
        smoke_loss=update(trial,trial_optimizer,list(range(20)),train_cache,device,config,weights,example_weight)
        assert any(not torch.equal(initial.state_dict()[name],value) for name,value in trial.state_dict().items())
        atomic_json(arm_root/"smoke.json",{"status":"PASS","loss":smoke_loss,"discarded_update":True})
        seed_everything(int(config["seed"]))
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=3e-4,weight_decay=1e-3)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0)
        best,best_epoch,diagnostic,step=baseline,-1,None,0
        for epoch in range(25):
            adapter.train();sampler.set_epoch(epoch);total=0.0
            for indices in sampler:
                indices=[train_cache["lookup"][train.entries[i]["sample_id"]] for i in indices]
                total+=update(adapter,optimizer,indices,train_cache,device,config,weights,example_weight)*len(indices)
                step+=1;scheduler.step()
            metric,_=predict(adapter,dev_cache,device,config)
            passed=evidence_eligible(metric,baseline) and metric["confusion_matrix"][2][0]<=baseline["confusion_matrix"][2][0]
            record={"epoch":epoch,"global_step":step,"train_ce":total/1600,"dev":metric,"eligible":passed}
            with (arm_root/"history.jsonl").open('a',encoding='utf-8') as handle:handle.write(json.dumps(record)+'\n')
            model.answerability_adapter.load_state_dict(adapter.state_dict())
            if diagnostic is None or selection_key(metric)>selection_key(diagnostic["dev"]):
                diagnostic=record
                save_checkpoint(arm_root/"checkpoints/diagnostic",model,optimizer,scheduler,epoch,step,
                                metric["macro_f1"],config_hash,record,replace=True)
            if passed and selection_key(metric)>selection_key(best):
                best,best_epoch=metric,epoch
                candidate_states[arm]=deepcopy(adapter.state_dict())
                save_checkpoint(arm_root/"checkpoints/best",model,optimizer,scheduler,epoch,step,
                                metric["macro_f1"],config_hash,record,replace=True)
            if epoch%5==0 or epoch==24:
                print(json.dumps({"arm":arm,"epoch":epoch,"macro_f1":metric["macro_f1"],
                                  "false_found":metric["false_found"],"absent_to_found":metric["confusion_matrix"][2][0],
                                  "eligible":passed}),flush=True)
        training_results[arm]={"best_epoch":best_epoch,"best_eligible_dev":best,"highest_macro_f1":diagnostic}
        if best_epoch>=0 and selection_key(best)>selection_key(global_best):
            global_best,selected_arm,selected_epoch=best,arm,best_epoch
    model.answerability_adapter.load_state_dict(candidate_states.get(selected_arm,initial.state_dict()))
    assert all(torch.equal(value.cpu(),model.state_dict()[name].cpu()) for name,value in base.state_dict().items())
    checked,chosen_dev_rows=evaluate(model,dev_loader,device,config,weights,source_weights)
    assert checked["answerability"]["confusion_matrix"]==global_best["confusion_matrix"]
    assert checked["grounding_accuracy"]==326/335
    frozen=root/"frozen";frozen.mkdir()
    freeze_optimizer=torch.optim.AdamW(model.answerability_adapter.parameters(),lr=3e-4)
    freeze_scheduler=torch.optim.lr_scheduler.LambdaLR(freeze_optimizer,lambda step:1.0)
    saved=save_checkpoint(frozen/"checkpoint",model,freeze_optimizer,freeze_scheduler,max(0,selected_epoch),
                          80*(selected_epoch+1) if selected_epoch>=0 else 0,global_best["macro_f1"],config_hash,
                          {"source_arm":selected_arm,"source_epoch":selected_epoch,"dev":checked})
    frozen_checkpoint=frozen/"checkpoint/model.safetensors"
    lock={"checkpoint":str(frozen_checkpoint),"checkpoint_sha256":saved["model_sha256"],
          "config_sha256":config_hash,"selection_arm":selected_arm,"selection_epoch":selected_epoch,
          "calibration_opened_before_freeze":False,"test_splits_accessed":False,"dev":global_best}
    atomic_json(frozen/"freeze_lock.json",lock)
    atomic_json(root/"training_summary.json",{"baseline_dev":baseline,"arms":training_results,"selected":lock})
    write_rows(frozen/"dev_predictions.jsonl",chosen_dev_rows)
    print('STAGE 2: checkpoint frozen; calibration extraction',flush=True)
    calibration=ArchivedPCRAUDataset(config,"calibration",profile="calibration")
    assert len(calibration)==1000
    families=sorted({e["family_id"] for e in calibration.entries});assert len(families)==200
    assert set(families).isdisjoint(set(train_cache["families"])|set(dev_cache["families"]))
    permutation=np.random.default_rng(20261003).permutation(families).tolist()
    partitions={"fit":permutation[:100],"threshold":permutation[100:160],"audit":permutation[160:]}
    atomic_json(root/"calibration_partitions.json",partitions)
    calibration_loader,_=make_loader(calibration,config,False)
    _,reference_rows=evaluate(base,calibration_loader,device,base_config,weights,source_weights)
    _,selected_rows=evaluate(model,calibration_loader,device,config,weights,source_weights)
    write_rows(frozen/"calibration_reference_predictions.jsonl",reference_rows)
    write_rows(frozen/"calibration_selected_predictions.jsonl",selected_rows)
    def subset(rows,partition):return [r for r in rows if r["family_id"] in set(partitions[partition])]
    fit_reference,fit_selected=subset(reference_rows,"fit"),subset(selected_rows,"fit")
    reference_calibrator,_=fit_risk(fit_reference,False,0.001)
    candidates={}
    for name,extended,l2 in (("logistic18",False,0.001),("logistic33",True,0.01)):
        result,_=fit_risk(fit_selected,extended,l2)
        candidates[name]=result
        atomic_json(frozen/f"{name}.json",result)
    selected_calibrator_name=min(candidates,key=lambda name:(candidates[name]["crossfit_metrics"]["brier"],len(candidates[name]["feature_names"])))
    selected_calibrator=candidates[selected_calibrator_name]
    atomic_json(frozen/"reference_logistic18.json",reference_calibrator)
    atomic_json(frozen/"calibrator_selection.json",{"selected":selected_calibrator_name,"criterion":"fit_family_crossfit_brier",
                "fit_only_metrics":{name:c["crossfit_metrics"] for name,c in candidates.items()},"audit_used":False})
    comparison_models={"v2_reference":(reference_rows,reference_calibrator),"selected_model":(selected_rows,selected_calibrator)}
    profiles=[]
    for name,(rows,calibrator) in comparison_models.items():
        threshold_rows=subset(rows,"threshold");risk=apply_risk(threshold_rows,calibrator)
        error=np.asarray([r["evaluation"]["error_event"] for r in threshold_rows])
        for policy in ("hard_found","risk_only"):
            allowed=np.ones(len(risk),bool) if policy=="risk_only" else np.asarray([predicted_answer(r)=="FOUND" for r in threshold_rows])
            for target in (0.05,0.075,0.1):
                selected_threshold=choose_policy_threshold(risk,error,[r["family_id"] for r in threshold_rows],allowed,target,20)
                profiles.append({"model":name,"policy":policy,"risk_target":target,"selection":selected_threshold})
    atomic_json(frozen/"policy_lock.json",{"profiles":profiles,"default_profile":{"model":"selected_model","policy":"risk_only","risk_target":0.05},
                "selection_partition":"threshold","audit_used_for_selection":False})
    print('STAGE 3: profiles locked; held calibration audit',flush=True)
    audit_dir=root/"calibration_audit";audit_dir.mkdir()
    audit_results=[]
    for profile in profiles:
        rows,calibrator=comparison_models[profile["model"]]
        audit_rows=subset(rows,"audit");risk=apply_risk(audit_rows,calibrator)
        metric,decisions=policy_result(audit_rows,risk,profile["policy"],profile["selection"]["threshold"])
        metric["probability_quality"]=probability_metrics(risk,[r["evaluation"]["error_event"] for r in audit_rows])
        name=f"{profile['model']}_{profile['policy']}_{profile['risk_target']:.3f}"
        write_rows(audit_dir/f"{name}.jsonl",decisions)
        audit_results.append({**profile,"audit":metric})
    assert sha256_file(frozen_checkpoint)==lock["checkpoint_sha256"]
    assert hashes=={str(path):sha256_file(path) for path in protected}
    summary={"status":"DEVELOPMENT_CALIBRATION_AUDIT_ONLY","training_selection":lock,"mining_counts":{
        "hard_positive":int(positive.sum()),"hard_negative":int(negative.sum()),"localization_failure_found":int(localization.sum())},
        "selected_calibrator":selected_calibrator_name,"calibrator_candidates_fit_metrics":{n:c["crossfit_metrics"] for n,c in candidates.items()},
        "profiles":audit_results,"source_hashes_unchanged":True,"checkpoint_unchanged_after_freeze":True,
        "audit_used_for_selection":False,"test_splits_accessed":False,"robot_motion_commanded":False}
    atomic_json(root/"summary.json",summary)
    lines=["# Học ca khó và selective prediction: báo cáo phát triển","",
           f"Model đã freeze: {selected_arm}, epoch {selected_epoch}; macro-F1 dev {global_best['macro_f1']:.4f}.",
           f"Mining train-only: {int(positive.sum())} dương khó, {int(negative.sum())} âm khó; {int(localization.sum())} FOUND có MAP sai không bị sửa nhãn.",
           "","Calibration chia family 100 fit / 60 threshold / 40 audit; audit 200 sample.",
           f"Calibrator chọn bằng fit-only family crossfit Brier: {selected_calibrator_name}.","",
           "| Model | Policy | Budget | Ngưỡng khóa | Audit EXECUTE | Coverage | Lỗi | Risk thực nghiệm | Wilson upper | Đúng FOUND bị từ chối | Bypass đúng/sai |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in audit_results:
        m=item["audit"];threshold=item["selection"]["threshold"]
        rate='N/A' if m['empirical_risk'] is None else f"{m['empirical_risk']:.2%}"
        upper='N/A' if m['wilson_upper_95'] is None else f"{m['wilson_upper_95']:.2%}"
        tau='None' if threshold is None else f"{threshold:.6f}"
        lines.append(f"| {item['model']} | {item['policy']} | {item['risk_target']:.1%} | {tau} | {m['accepted']}/200 | {m['coverage']:.2%} | {m['errors']} | {rate} | {upper} | {m['correct_found_rejected']} | {m['bypass_correct']}/{m['bypass_errors']} |")
    lines += ["","Đây là held calibration audit; không thay Test-IID, không dùng audit để chọn lại model/calibrator/threshold.",
              "Budget là mục tiêu chọn ngưỡng, không đồng nhất threshold risk; cận Wilson chưa là bảo đảm theo family hoặc robot.",
              "Không dùng calibrator V2 lịch sử trong mining hoặc đối chứng audit; V2 reference fit lại trên đúng fit partition.",
              "Nhiều vòng phát triển đã biết Test-IID cũ; mọi kết luận khái quát cần test độc lập mới.",
              "Checkpoint/config/calibrator V2 và hình 19 giữ nguyên. Không chạy robot hoặc thay số liệu LaTeX."]
    (root/"SUMMARY.md").write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({"output":str(root),"selected_model":selected_arm,"calibrator":selected_calibrator_name,
                      "audit_profiles":len(audit_results)}),flush=True)


if __name__=='__main__':main()
