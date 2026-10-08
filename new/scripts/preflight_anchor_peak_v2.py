#!/usr/bin/env python3
"""V2 peak objective preflight only: frozen inference and backward, no optimizer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import statistics
import sys
import time

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrau.anchor_peak_experiment import PEAK_VERSION, anchor_peak_loss, prepare_peak_arms, supervision_from_full_masks, validate_locked_peak_config
from pcrau.anchor_shadow_experiment import PROTOCOL, anchor_shadow_loss, family_batches, observable_batch, pilot_presentations, state_digest
from pcrau.anchor_shadow_pilot import spatial_summary
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.utils import atomic_json, read_json, runtime_info, seed_everything, sha256_file, workspace_path
from run_anchor_shadow_pilot import FeatureCache, jsonl


def serialize_losses(losses):
    return {k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in losses.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="new/outputs/pcrau_s1_anchor_peak_v2_preflight_20261005")
    args=parser.parse_args()
    output=workspace_path(args.output)
    if workspace_path("new/outputs") not in output.parents: raise ValueError("Output must stay under new/outputs")
    output.mkdir(exist_ok=False)
    protected={}
    def protect(path,expected=None):
        path=Path(path).resolve()
        if str(path) not in protected: protected[str(path)]=sha256_file(path)
        if expected is not None and protected[str(path)]!=expected: raise ValueError(f"Hash mismatch: {path}")
        return protected[str(path)]
    config_path=workspace_path("new/configs/anchor_peak_pilot_v2_20261005.json")
    candidate=read_json(config_path); validate_locked_peak_config(candidate); config_hash=protect(config_path)
    design_path=workspace_path("plan/S1_ANCHOR_PEAK_PROTOCOL_V2_LOCK_20261005.md"); design_hash=protect(design_path)
    proposal_path=workspace_path("plan/S1_ANCHOR_PEAK_OBJECTIVE_V2_PROPOSAL_20261005.md"); protect(proposal_path)
    protect(workspace_path("plan/S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md"))
    active_path=workspace_path("new/outputs/active_experimental_profile.json"); active=read_json(active_path); protect(active_path)
    for key in ("config","checkpoint","calibrator","profiles"): protect(active[key],active[key+"_sha256"])
    config=read_json(Path(active["config"]))
    source_paths=list(workspace_path("new/src/pcrau").glob("*.py"))+[
        Path(__file__).resolve(),workspace_path("new/scripts/run_anchor_shadow_pilot.py"),
        workspace_path("new/tests/test_anchor_peak_experiment.py"),workspace_path("new/tests/test_anchor_shadow.py")]
    for path in source_paths: protect(path)
    # Synthetic tests run before real checkpoint load; they create no optimizer.
    torch.set_num_threads(4); synthetic=[]
    for name,fn in runpy.run_path(str(workspace_path("new/tests/test_anchor_peak_experiment.py"))).items():
        if name.startswith("test_") and callable(fn): fn(); synthetic.append({"test":name,"pass":True})
    atomic_json(output/"synthetic_checks.json",{"checks":synthetic,"optimizer_created":False})
    seed_everything(PROTOCOL.seed)
    device=torch.device("cuda")
    model=PCRAUTargetV2(config).to(device).eval(); load_model_checkpoint(model,active["checkpoint"],active["config_sha256"])
    arms=prepare_peak_arms(model)
    initial={k:state_digest(a.branch.residual.state_dict()) for k,a in arms.items()}
    assert len(set(initial.values()))==1
    count={k:sum(p.numel() for p in a.branch.parameters() if p.requires_grad) for k,a in arms.items()}
    assert all(v==33024 for v in count.values())
    frozen=state_digest(model.state_dict()); hook_counts={k:len(m._forward_hooks) for k,m in model.named_modules()}
    datasets={s:ArchivedPCRAUDataset(config,s,verify_feature_hash=True) for s in ("train","dev")}
    caches={s:FeatureCache(d,protect) for s,d in datasets.items()}
    layout=datasets["train"].layout; protect(layout.manifest); protect(layout.feature_index)
    entries=read_json(layout.manifest)["entries"]
    train=pilot_presentations(entries,config["model"],"train"); dev=pilot_presentations(entries,config["model"],"dev")
    all_dev=[{"entry":e,"prompt":e["feature_input"]["prompt"],"sample_id":e["sample_id"],"family_id":e["family_id"],"prefix_index":0} for e in datasets["dev"].entries]
    assert len(train)==1024 and len(dev)==64 and len(all_dev)==400
    assert len({p["family_id"] for p in train})==64 and len({p["family_id"] for p in dev})==16
    assert not {p["family_id"] for p in train}&{p["family_id"] for p in dev}
    plans={str(e):family_batches(train,e) for e in range(PROTOCOL.max_epochs)}
    pilot=workspace_path("new/outputs/pcrau_s1_anchor_shadow_pilot_20261005")
    protect(pilot/"family_batch_plans.json"); protect(pilot/"split_manifest.json")
    assert plans==read_json(pilot/"family_batch_plans.json")
    split_manifest={s:[{k:p[k] for k in ("sample_id","family_id","prefix_index","prompt")} for p in values] for s,values in (("train",train),("dev",dev))}
    assert split_manifest==read_json(pilot/"split_manifest.json")
    atomic_json(output/"family_batch_plans.json",plans); atomic_json(output/"split_manifest.json",split_manifest)
    atomic_json(output/"locked_config.json",candidate)
    source_hashes={str(p.resolve()):protect(p) for p in source_paths}
    lock={"version":PEAK_VERSION,"config_sha256":config_hash,"protocol_sha256":design_hash,
        "config":candidate,"baseline_bundle":active,"initialization_sha256":initial,"frozen_model_state_sha256":frozen,
        "source_sha256":source_hashes,"trainable_count_per_arm":count,
        "trainable_parameter_names":{k:[n for n,p in a.branch.named_parameters() if p.requires_grad] for k,a in arms.items()},
        "preflight_only":True,"optimizer_created":False,"optimizer_steps":0,"training_executed":False}
    atomic_json(output/"locked_protocol.json",lock)
    save_file({k:v.detach().cpu().contiguous() for k,v in arms["C1"].branch.residual.state_dict().items()},str(output/"initial_mlp.safetensors"))

    def batch(items):
        s=items[0]["entry"]["split"]
        assert all(p["entry"]["split"]==s for p in items)
        return observable_batch(caches[s],[p["entry"] for p in items],[p["prompt"] for p in items],device)

    modes={}
    with (output/"inference_predictions.jsonl").open("w") as predictions,(output/"identity_checks.jsonl").open("w") as checks:
        for mode in ("fp32","amp_bfloat16"):
            maximum={k:0. for k in arms}; checked=Counter(); scopes=Counter()
            for split,values in (("train",train),("dev",all_dev)):
                for start in range(0,len(values),8):
                    items=values[start:start+8]; inputs=batch(items); prompts=[p["prompt"] for p in items]
                    ctx=autocast_context(device,config["optimization"]) if mode!="fp32" else torch.autocast(device.type,enabled=False)
                    with torch.no_grad(),ctx:
                        plain=model(inputs); cap=arms["C1"].branch.capture(inputs,prompts)
                        logits={k:a.branch.predict(cap) for k,a in arms.items()}
                        exact={k:torch.equal(v,cap.baseline[k]) for k,v in plain.items()}
                        assert all(exact.values())
                        # Also check each wrapper's capture; shared predict could
                        # otherwise conceal a branch-specific capture regression.
                        for k in ("P1","P2"):
                            other=arms[k].branch.capture(inputs,prompts)
                            assert all(torch.equal(v,other.baseline[n]) for n,v in plain.items())
                        inactive=~cap.active_slots
                        errors={k:float((v.float()-plain["anchor_logits"].float()).abs().max()) for k,v in logits.items()}
                        assert all(v<=1e-6 for v in errors.values()),(mode,start,errors)
                        assert all(torch.equal(v[inactive],plain["anchor_logits"][inactive]) for v in logits.values())
                    assert hook_counts=={k:len(m._forward_hooks) for k,m in model.named_modules()}
                    for k,v in errors.items(): maximum[k]=max(maximum[k],v)
                    checked[split]+=len(items)
                    checks.write(json.dumps({"mode":mode,"split":split,"start":start,"baseline_exact":exact,"anchor_max_abs_error":errors,
                        "baseline_state_digest":state_digest(plain),"capture_state_digest":state_digest(cap.baseline)})+"\n")
                    for j,item in enumerate(items):
                        scopes[(split,cap.scope_status[j])]+=1
                        predictions.write(json.dumps({"mode":mode,"split":split,"sample_id":item["sample_id"],"family_id":item["family_id"],
                            "prefix_index":item["prefix_index"],"prompt":item["prompt"],"scope_status":cap.scope_status[j],
                            "parsed":cap.queries[j].to_dict(),"M0":spatial_summary(plain["anchor_logits"][j,0].float().cpu().numpy()),
                            **{k:spatial_summary(v[j,0].float().cpu().numpy()) for k,v in logits.items()}})+"\n")
                    if start%160==0 or start+8>=len(values): print(f"{mode} {split}: identity {min(start+8,len(values))}/{len(values)}",flush=True)
            modes[mode]={"presentations":dict(checked),"max_abs_error":maximum,"baseline_all_tensor_outputs_exact":True,
                "inactive_maps_exact":True,"scope_counts":[{"split":s,"status":k,"count":v} for (s,k),v in sorted(scopes.items())]}
    prediction_hash=sha256_file(output/"inference_predictions.jsonl")
    # Train/dev supervision is first accessed after all identity traces persist.
    by_sample={p["sample_id"]:p for p in train+dev}; full={}; support_rows=[]
    for sample,p in sorted(by_sample.items()):
        e=p["entry"]; descriptor=e["supervision"]["anchor_masks"][0]
        path=layout.dataset_path(descriptor["path"]); protect(path,descriptor["sha256"])
        mask=cv2.imread(str(path),0); assert mask is not None and mask.shape==(480,640)
        masks=np.zeros((1,3,480,640),dtype=bool); masks[0,0]=mask>0
        sup=supervision_from_full_masks(masks); full[sample]=sup
        support_rows.append({"sample_id":sample,"split":e["split"],"family_id":e["family_id"],"full_pixels":int(sup.full_pixel_counts[0,0]),
            "full_visible":bool(sup.full_visible[0,0]),"area_occupancy_sum":float(sup.area_masks[0,0].sum()),
            "center_count":int(sup.center_support[0,0].sum()),"auxiliary_skipped_only":bool(sup.full_visible[0,0] and not sup.center_support[0,0].any())})
    support={}
    for split in ("train","dev"):
        rs=[r for r in support_rows if r["split"]==split]
        support[split]={"samples":len(rs),"families":len({r["family_id"] for r in rs}),
            "visible":sum(r["full_visible"] for r in rs),"empty":sum(not r["full_visible"] for r in rs),
            "representable_visible":sum(r["full_visible"] and r["center_count"]>0 for r in rs),
            "skipped_auxiliary_only":[r["sample_id"] for r in rs if r["auxiliary_skipped_only"]]}
    tiny="v211dev_family_000077__relation_counterfactual"
    assert (support["train"]["visible"],support["train"]["empty"],support["train"]["representable_visible"])==(251,5,250)
    assert support["train"]["skipped_auxiliary_only"]==[tiny]
    assert (support["dev"]["visible"],support["dev"]["empty"],support["dev"]["representable_visible"])==(61,3,61)
    tiny_row=next(r for r in support_rows if r["sample_id"]==tiny); assert tiny_row["full_pixels"]==8
    jsonl(output/"supervision_support.jsonl",support_rows); atomic_json(output/"supervision_summary.json",support)
    positive=next(r["sample_id"] for r in support_rows if r["split"]=="train" and r["full_visible"] and r["center_count"]>0)
    negative=next(r["sample_id"] for r in support_rows if r["split"]=="train" and not r["full_visible"])
    probe_items=[p for p in train if p["sample_id"] in {positive,negative,tiny}]
    assert len(probe_items)==12
    from pcrau.anchor_peak_experiment import PeakSupervision
    probe_sup=PeakSupervision(*(torch.cat([getattr(full[p["sample_id"]],k) for p in probe_items]).to(device)
        for k in ("area_masks","center_support","full_visible","full_pixel_counts")))
    inputs=batch(probe_items); prompts=[p["prompt"] for p in probe_items]; gradients={}
    for mode in ("fp32","amp_bfloat16"):
        ctx=autocast_context(device,config["optimization"]) if mode!="fp32" else torch.autocast(device.type,enabled=False)
        with ctx: cap=arms["C1"].branch.capture(inputs,prompts)
        mode_rows={}
        for key,arm in arms.items():
            arm.branch.train(); arm.branch.zero_grad(set_to_none=True)
            losses=arm.loss(arm.branch.predict(cap),probe_sup,cap.active_slots)
            assert losses["total"].dtype==torch.float32
            if key=="C1":
                old=anchor_shadow_loss(arm.branch.predict(cap),probe_sup.area_masks,cap.active_slots)
                assert torch.equal(old["total"],losses["total"])
            else:
                assert losses["peak_skipped_unrepresentable_count"]==4 and losses["peak_empty_count"]==4 and losses["peak_visible_count"]==4
                assert losses["visible_count"]==8 and losses["empty_count"]==4
            losses["total"].backward()
            norms={n:float(p.grad.norm()) for n,p in arm.branch.residual.named_parameters()}
            assert all(torch.isfinite(p.grad).all() for p in arm.branch.residual.parameters())
            assert norms["2.weight"]>0 and norms["2.bias"]>0 and norms["0.weight"]==norms["0.bias"]==0
            assert all(p.grad is None for p in model.parameters())
            assert state_digest(arm.branch.residual.state_dict())==initial[key]
            mode_rows[key]={"losses":serialize_losses(losses),"total_gradient_norms":norms,"baseline_gradients":0}
            arm.branch.zero_grad(set_to_none=True)
            if key!="C1":
                aux=arm.loss(arm.branch.predict(cap),probe_sup,cap.active_slots)["peak"]
                aux.backward(); aux_norms={n:float(p.grad.norm()) for n,p in arm.branch.residual.named_parameters()}
                assert aux_norms["2.weight"]>0 and aux_norms["2.bias"]>0
                assert all(torch.isfinite(p.grad).all() for p in arm.branch.residual.parameters()) and all(p.grad is None for p in model.parameters())
                mode_rows[key]["aux_only_gradient_norms"]=aux_norms; arm.branch.zero_grad(set_to_none=True)
            arm.branch.eval()
        gradients[mode]=mode_rows
    atomic_json(output/"gradient_probe.json",{"samples":[positive,negative,tiny],"presentations":12,"supervision_access":"train only, after identity traces saved",
        "modes":gradients,"optimizer_created":False,"optimizer_steps":0,"zero_W1_gradient_expected":True})
    # Explicitly reject all new oracle supervision keys from unchanged capture.
    rejected=[]
    for key in ("anchor_center_support","full_visible","full_pixel_counts","anchor_full_mask"):
        try: arms["P1"].branch.capture(dict(inputs,**{key:torch.zeros(1,device=device)}),prompts)
        except ValueError: rejected.append(key)
        else: raise AssertionError("Oracle supervision accepted in model boundary")
    atomic_json(output/"boundary_checks.json",{"rejected_oracle_keys":rejected,"MODEL_INPUT_KEYS":sorted(model.MODEL_INPUT_KEYS),
        "supervision_never_in_capture_or_predict":True})

    timing_items=train[:8]; timing_batch=batch(timing_items); timing_prompts=[p["prompt"] for p in timing_items]
    measurements={k:[] for k in ("M0","C1","P1","P2")}
    def call(key):
        with torch.no_grad(),autocast_context(device,config["optimization"]):
            if key=="M0": arms["C1"].branch.capture(timing_batch,timing_prompts)
            else: arms[key].branch(timing_batch,timing_prompts)
    for _ in range(20):
        for key in measurements: call(key)
    torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(100):
        order=list(measurements); offset=iteration%len(order); order=order[offset:]+order[:offset]
        for key in order:
            torch.cuda.synchronize(device); begin=time.perf_counter(); call(key); torch.cuda.synchronize(device)
            measurements[key].append((time.perf_counter()-begin)*1000)
    latency={k:{"median_ms":statistics.median(v),"p95_ms":float(np.percentile(v,95))} for k,v in measurements.items()}
    for key in arms:
        latency[key]["overhead_fraction"]=latency[key]["median_ms"]/latency["M0"]["median_ms"]-1
        latency[key]["gate_pass"]=latency[key]["overhead_fraction"]<=.2
    atomic_json(output/"latency.json",{"batch_size":8,"warmup":20,"runs_per_arm":100,"scope":"parser/capture/cached-feature sidecar only; no loss/backbone/disk",
        "raw_ms":measurements,"measurements":latency,"peak_shared_allocated_bytes":torch.cuda.max_memory_allocated(device)})
    assert state_digest(model.state_dict())==frozen and all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(state_digest(a.branch.residual.state_dict())==initial[k] and all(p.grad is None for p in a.branch.residual.parameters()) for k,a in arms.items())
    assert hook_counts=={k:len(m._forward_hooks) for k,m in model.named_modules()}
    assert all(sha256_file(Path(p))==d for p,d in protected.items())
    assert prediction_hash==sha256_file(output/"inference_predictions.jsonl")
    result={"status":"S1_PEAK_V2_PREFLIGHT_PASS","technical_gates_pass":True,
        "latency_gate_pass":all(latency[k]["gate_pass"] for k in arms),"runtime":runtime_info(),"version":PEAK_VERSION,
        "modes":modes,"synthetic_checks_passed":len(synthetic),"support":support,"gradient_probes_pass":True,
        "trainable_count":count,"initialization_sha256":initial,"frozen_model_state_sha256":frozen,
        "config_sha256":config_hash,"protocol_sha256":design_hash,"source_sha256":source_hashes,
        "all_baseline_weights_stats_unchanged":True,"residual_initializations_unchanged":True,
        "optimizer_created":False,"optimizer_steps":0,"training_executed":False,"binding_improvement_evaluated":False,
        "oracle_keys_rejected":rejected,"data_and_family_plans_match_v1":True,
        "inference_saved_before_oracle_masks":True,"inference_predictions_sha256":prediction_hash,
        "calibration_access":"bundle hash only; no samples/fit","test_accessed":False,"verifier_or_mc":False,
        "latency":latency,"protected_files":len(protected),"epoch_orchestration_implemented":False,
        "limits":["Preflight only; no learned P1/P2 binding/evidence results", "W1 gradient zero is expected at zero W2", "Latency cached sidecar only; auxiliary loss has no inference cost"]}
    atomic_json(output/"summary.json",result)
    atomic_json(output/"provenance.json",{"protected_sha256":protected,"source_sha256":source_hashes,"argv":sys.argv,
        "baseline_state_sha256_before_after":frozen,"initialization_sha256":initial})
    print(json.dumps({"status":result["status"],"latency_gate_pass":result["latency_gate_pass"],"support":support,"latency":latency,
        "optimizer_created":False,"optimizer_steps":0},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
