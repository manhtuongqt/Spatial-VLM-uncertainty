#!/usr/bin/env python3
"""Frozen post-pilot failure audit and conditioning-span interventions, train/dev.

No optimizer, calibration, verifier, label-based model inputs or checkpoint
selection. Alternate spans change only M1 residual pooling on the same capture.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrau.anchor_shadow import AnchorShadow
from pcrau.anchor_shadow_experiment import observable_batch, state_digest
from pcrau.anchor_shadow_pilot import spatial_summary
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchivedPCRAUDataset
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.utils import atomic_json, read_json, seed_everything, sha256_file, workspace_path
from run_anchor_shadow_pilot import FeatureCache, jsonl

MODES = ("M0", "M1_anchor", "M1_target", "M1_whole", "M2_whole")


def logits_stats(logits, mask=None):
    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    sig = 1/(1+np.exp(-np.clip(flat, -700, 700)))
    p = np.exp(flat-flat.max()); p /= p.sum()
    loss = np.logaddexp(0, flat)
    out = {**spatial_summary(np.asarray(logits)), "zero_bce_mean": float(loss.mean()),
        "zero_bce_max": float(loss.max()), "zero_bce_top16_mean": float(np.sort(loss)[-16:].mean()),
        "logit_max": float(flat.max()), "positive_cells": int((flat>0).sum()),
        "sigmoid_ge_0p9_cells": int((sig>=.9).sum()), "sigmoid_ge_0p5_cells": int((sig>=.5).sum()),
        "unweighted_per_sample_mean_zero_bce_peak_gradient": float(sig.max()/len(flat))}
    if mask is not None:
        out["visible_pixels"] = int(mask.sum())
        x,y = out["map_pixel_xy"]; gx,gy = out["map_grid_xy"]
        centers = mask[10::20,10::20]
        small = cv2.resize(mask.astype(np.float32),(32,24),interpolation=cv2.INTER_AREA)
        out.update({"hit": bool(mask[y,x]) if mask.any() else None,
            "cell_overlap": bool(small[gy,gx]>0) if mask.any() else None,
            "grid_centers_inside_mask": int(centers.sum()),
            "fractional_mask_weighted_softmax_mass": float((p.reshape(24,32)*small).sum()) if mask.any() else None})
        if mask.any():
            ys,xs=np.nonzero(mask)
            out.update({"distance_to_mask_px": float(cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[y,x]),
                "mask_centroid_xy": [float(xs.mean()),float(ys.mean())],
                "best_center_mask_logit": float(flat[centers.reshape(-1)].max()) if centers.any() else None})
            best=out["best_center_mask_logit"]
            out["best_center_mask_rank"] = int((flat>best).sum()+1) if best is not None else None
            out["outside_vs_best_mask_logit_gap"] = float(flat.max()-best) if best is not None else None
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="new/outputs/pcrau_s1_anchor_failure_audit_20261005")
    args=parser.parse_args()
    output=workspace_path(args.output)
    assert workspace_path("new/outputs") in output.parents
    output.mkdir(exist_ok=False); (output/"cases").mkdir()
    run=workspace_path("new/outputs/pcrau_s1_anchor_shadow_pilot_20261005")
    protected={}
    def protect(path,expected=None):
        path=Path(path).resolve(); digest=sha256_file(path)
        assert expected is None or digest==expected,(str(path),"hash mismatch")
        protected[str(path)]=digest
        return digest
    # Bind all original run artifacts, including logs and selected checkpoints.
    for path in run.rglob("*"):
        if path.is_file(): protect(path)
    for path in workspace_path("new/src/pcrau").glob("*.py"): protect(path)
    protect(__file__); protect(workspace_path("new/scripts/run_anchor_shadow_pilot.py"))
    design=workspace_path("plan/S1_PHRASE_CONDITIONED_ANCHOR_EXPERIMENT_20261005.md")
    summary=read_json(run/"paired_report_r2/summary.json")
    protect(design,summary["protocol_sha256"])
    active_path=workspace_path("new/outputs/active_experimental_profile.json")
    active=read_json(active_path); protect(active_path)
    for key in ("config","checkpoint","calibrator","profiles"):
        protect(active[key],active[key+"_sha256"])
    config=read_json(Path(active["config"]))
    saved=np.load(run/"selected_spatial_logits.npz")
    saved_order=[str(s) for s in saved["sample_ids"]]; index={s:i for i,s in enumerate(saved_order)}
    predictions=[json.loads(line) for line in (run/"selected_inference_predictions.jsonl").read_text().splitlines()]
    scored=[json.loads(line) for line in (run/"evaluator_rows.jsonl").read_text().splitlines()]
    assert [r["sample_id"] for r in predictions]==saved_order
    # Repeat identical order/batch/AMP, reproducing selected inference before any
    # evaluator mask/registry/semantic-label access in this process.
    seed_everything(24082026); torch.set_num_threads(4)
    device=torch.device("cuda")
    baseline=PCRAUTargetV2(config).to(device).eval()
    load_model_checkpoint(baseline,active["checkpoint"],active["config_sha256"])
    arms={"M1":AnchorShadow(baseline,"phrase",24082026),"M2":AnchorShadow(baseline,"whole_text",24082026)}
    for key,arm in arms.items():
        arm.residual.load_state_dict(load_file(str(run/key/"checkpoints/best/mlp.safetensors")),strict=True)
        arm.eval().requires_grad_(False)
    states={"baseline":state_digest(baseline.state_dict()),**{k:state_digest(a.residual.state_dict()) for k,a in arms.items()}}
    assert states["baseline"]==summary["baseline_state_sha256_before_after"]
    datasets={s:ArchivedPCRAUDataset(config,s,verify_feature_hash=True) for s in ("train","dev")}
    caches={s:FeatureCache(d,protect) for s,d in datasets.items()}
    layout=datasets["train"].layout; protect(layout.manifest); protect(layout.feature_index)
    entries={e["sample_id"]:e for d in datasets.values() for e in d.entries}
    raw={k:[] for k in MODES}; diagnostic=[]
    for start in range(0,len(predictions),8):
        sub=predictions[start:start+8]; split=sub[0]["split"]
        assert all(p["split"]==split for p in sub)
        inputs=observable_batch(caches[split],[entries[p["sample_id"]] for p in sub],[p["prompt"] for p in sub],device)
        with torch.no_grad(),autocast_context(device,config["optimization"]):
            cap=arms["M1"].capture(inputs,[p["prompt"] for p in sub])
            target_masks=torch.tensor([q.token_masks()["target_token_mask"] for q in cap.queries],device=device,dtype=torch.bool)
            target_masks=target_masks[:,None,:].expand_as(cap.phrase_masks)
            whole_masks=cap.token_mask[:,None,:].expand_as(cap.phrase_masks)
            captures={"M1_anchor":cap,"M1_target":replace(cap,phrase_masks=target_masks),
                "M1_whole":replace(cap,phrase_masks=whole_masks)}
            maps={"M0":cap.baseline["anchor_logits"],"M2_whole":arms["M2"].predict(cap),
                **{k:arms["M1"].predict(c) for k,c in captures.items()}}
            deltas={}; pooled={}
            with torch.autocast(device_type=device.type,enabled=False):
                for mode,c in captures.items():
                    weights=c.phrase_masks.float()
                    pooled[mode]=torch.einsum("bsl,bld->bsd",weights,c.text.float())/weights.sum(-1,keepdim=True).clamp_min(1)
                    deltas[mode]=arms["M1"].residual(pooled[mode])
            arrays={k:v.float().cpu().numpy() for k,v in maps.items()}
            target=cap.baseline["target_logits"].float().cpu().numpy()
        for j,pred in enumerate(sub):
            sample=pred["sample_id"]; ix=index[sample]
            for mode,old in (("M0","M0"),("M1_anchor","M1"),("M2_whole","M2")):
                assert np.array_equal(arrays[mode][j],saved[old][ix]),(sample,mode,"replay mismatch")
            assert np.array_equal(target[j],saved["target"][ix])
            inactive=~cap.active_slots[j].cpu().numpy()
            for mode in MODES:
                assert np.array_equal(arrays[mode][j][inactive],saved["M0"][ix][inactive])
                raw[mode].append(arrays[mode][j,0])
            row={"sample_id":sample,"family_id":pred["family_id"],"split":pred["split"],"prompt":pred["prompt"],
                "parsed":pred["parsed"],"feature_key":pred["feature_key"],"scope_status":pred["scope_status"],
                "modes":{k:spatial_summary(arrays[k][j,0]) for k in MODES},"intervention":{}}
            for mode in ("M1_target","M1_whole"):
                a,b=pooled["M1_anchor"][j,0],pooled[mode][j,0]
                row["intervention"][mode]={"pooled_cosine_vs_anchor":float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                    "pooled_l2_vs_anchor":float((a-b).norm()),
                    "residual_l2_vs_anchor":float((deltas[mode][j,0]-deltas["M1_anchor"][j,0]).norm()),
                    "logit_max_abs_vs_anchor":float(np.abs(arrays[mode][j,0]-arrays["M1_anchor"][j,0]).max()),
                    "peak_changed":row["modes"][mode]["map_grid_xy"]!=row["modes"]["M1_anchor"]["map_grid_xy"]}
            diagnostic.append(row)
    raw={k:np.stack(v) for k,v in raw.items()}
    jsonl(output/"conditioning_predictions.jsonl",diagnostic)
    np.savez_compressed(output/"conditioning_logits.npz",sample_ids=np.asarray(saved_order),**raw)
    inference_hash=protect(output/"conditioning_predictions.jsonl")
    # Only here, evaluator-only masks/labels/IDs may be read and joined.
    registry_path=layout.dataset_root/"object_registry.json"; protect(registry_path)
    registry=read_json(registry_path); by_label={int(v["label"]):v for v in registry.values()}
    label_by_id={v["id"]:int(v["label"]) for v in registry.values()}
    diagnostics={r["sample_id"]:r for r in diagnostic}
    evaluated=[]; focus=[]
    misses=[r for r in scored if r["split"]=="dev" and r["anchor_pixels"] and not r["models"]["M1"]["anchor_hit"]]
    empty=[r for r in scored if r["split"]=="train" and not r["anchor_pixels"]]
    gains=[r for r in scored if r["split"]=="dev" and r["anchor_pixels"] and r["models"]["M1"]["anchor_hit"]!=r["models"]["M0"]["anchor_hit"]]
    assert len(misses)==13 and len(empty)==5 and len(gains)==4
    focus_ids={r["sample_id"] for r in misses+empty+gains}
    for old in scored:
        sample=old["sample_id"]; ix=index[sample]; pred=diagnostics[sample]
        e=entries[sample]
        for desc in [e["supervision"]["anchor_masks"][0],{"path":e["supervision"]["target_mask_path"],"sha256":e["supervision"]["target_mask_sha256"]}]:
            protect(layout.dataset_path(desc["path"]),desc["sha256"])
        anchor=cv2.imread(old["anchor_mask_path"],0)>0; target_mask=cv2.imread(old["target_mask_path"],0)>0
        stats={mode:logits_stats(raw[mode][ix],anchor) for mode in MODES}
        for mode in MODES:
            x,y=stats[mode]["map_pixel_xy"]; stats[mode]["inside_valid_target_mask"]=bool(target_mask[y,x])
        assert stats["M1_anchor"]["hit"]==old["models"]["M1"]["anchor_hit"]
        row={**pred,"truth":old["truth"],"variant":old["variant"],"anchor_pixels":old["anchor_pixels"],
            "anchor_ids":old["anchor_ids"],"modes":stats,
            "rgb_path":old["rgb_path"],"anchor_mask_path":old["anchor_mask_path"],"target_mask_path":old["target_mask_path"]}
        if sample in focus_ids:
            record_path=layout.dataset_path(e["record_path"]); protect(record_path,e["record_sha256"])
            ev=read_json(record_path)["evaluator_only"]
            descriptor=ev["semantic_instance_labels"]; sem_path=layout.dataset_path(descriptor["path"]); protect(sem_path,descriptor["sha256"])
            semantic=cv2.imread(str(sem_path),cv2.IMREAD_UNCHANGED)
            assert semantic is not None and semantic.shape[:2]==anchor.shape
            if semantic.ndim==3:
                assert np.array_equal(semantic[:,:,0],semantic[:,:,1]) and np.array_equal(semantic[:,:,0],semantic[:,:,2])
                semantic=semantic[:,:,0]
            wanted=label_by_id[old["anchor_ids"][0]]
            assert np.array_equal(anchor,semantic==wanted),(sample,"anchor/semantic mismatch")
            target_ids=ev["spatial_label"]["candidate_target_ids"]
            target_labels=[label_by_id[v] for v in target_ids]
            requested_target=np.isin(semantic,target_labels)
            row.update({"semantic_path":str(sem_path),"semantic_anchor_label":wanted,"candidate_target_ids":target_ids,
                "requested_active_ids":ev["object_oracle"]["active_scene_ids"],
                "state_submode":ev["uncertainty_label"]["state_submode"],
                "anchor_in_requested_active_ids":old["anchor_ids"][0] in ev["object_oracle"]["active_scene_ids"],
                "observed_registry_objects":[v["id"] for k,v in sorted(by_label.items()) if (semantic==k).any()]})
            for mode in MODES:
                x,y=stats[mode]["map_pixel_xy"]; value=int(semantic[y,x]); identity=by_label.get(value)
                stats[mode].update({"semantic_label_at_peak":value,"object_id_at_peak":identity["id"] if identity else None,
                    "semantic_class_at_peak":identity["semantic_class"] if identity else "unmapped/background",
                    "inside_requested_target_instance":bool(requested_target[y,x])})
            protect(old["rgb_path"],e["feature_input"]["rgb_sha256"])
            # Scientific overlays derived from actual RGB, logits and evaluator
            # masks; no generated/edited confidence or synthetic object imagery.
            rgb=cv2.cvtColor(cv2.imread(old["rgb_path"]),cv2.COLOR_BGR2RGB)
            fig,axes=plt.subplots(1,3,figsize=(13,4.9),layout="constrained")
            axes[0].imshow(rgb)
            if anchor.any(): axes[0].contour(anchor,levels=[.5],colors=["cyan"],linewidths=1)
            if requested_target.any(): axes[0].contour(requested_target,levels=[.5],colors=["lime"],linewidths=1)
            colors={"M0":"red","M1_anchor":"white","M1_target":"magenta","M1_whole":"orange","M2_whole":"yellow"}
            markers={"M0":"x","M1_anchor":"+","M1_target":"o","M1_whole":"s","M2_whole":"v"}
            for mode in MODES:
                x,y=stats[mode]["map_pixel_xy"]
                if mode in {"M1_target","M1_whole"}:
                    axes[0].scatter([x],[y],edgecolors=colors[mode],facecolors="none",marker=markers[mode],s=100,label=mode)
                else:
                    axes[0].scatter([x],[y],c=colors[mode],marker=markers[mode],s=100,label=mode)
            axes[0].legend(fontsize=6,loc="lower left"); axes[0].set_title("RGB: cyan anchor / lime requested target",fontsize=9)
            lo=min(raw[k][ix].min() for k in ("M0","M1_anchor")); hi=max(raw[k][ix].max() for k in ("M0","M1_anchor"))
            for ax,mode in zip(axes[1:],("M0","M1_anchor")):
                im=ax.imshow(raw[mode][ix],cmap="coolwarm",vmin=lo,vmax=hi,extent=(0,640,480,0))
                if anchor.any(): ax.contour(anchor,levels=[.5],colors=["cyan"],linewidths=1)
                x,y=stats[mode]["map_pixel_xy"]; ax.scatter([x],[y],c="white",marker="x",s=70)
                ax.set_title(f"{mode}: raw logits; peak={stats[mode]['sigmoid_max']:.4f}\nzero-BCE mean={stats[mode]['zero_bce_mean']:.6f}",fontsize=9)
            fig.colorbar(im,ax=list(axes[1:]),shrink=.8,label="raw logit (shared scale)")
            for ax in axes: ax.set_xlim(0,640); ax.set_ylim(480,0); ax.tick_params(labelsize=7)
            fig.suptitle(f"{sample}\n{old['parsed']['target']['text']} {old['parsed']['predicate']} {old['parsed']['anchors'][0]['text']} | truth={old['truth']}",fontsize=10)
            path=output/"cases"/(sample+".png"); fig.savefig(path,dpi=140); plt.close(fig)
            row["figure"]=str(path.relative_to(output)); focus.append(row)
        evaluated.append(row)
    jsonl(output/"conditioning_evaluator.jsonl",evaluated)
    jsonl(output/"focus_cases.jsonl",focus)
    jsonl(output/"dev_misses.jsonl",[r for r in focus if r["sample_id"] in {x["sample_id"] for x in misses}])
    jsonl(output/"train_empty.jsonl",[r for r in focus if r["sample_id"] in {x["sample_id"] for x in empty}])
    groups={}
    for split in ("train","dev"):
        rs=[r for r in evaluated if r["split"]==split]; visible=[r for r in rs if r["anchor_pixels"]]
        group={"samples":len(rs),"families":len({r["family_id"] for r in rs}),"visible":len(visible),"empty":len(rs)-len(visible),"modes":{}}
        for mode in MODES:
            group["modes"][mode]={"original_anchor_hits":sum(r["modes"][mode]["hit"] for r in visible),
                "peaks_changed_vs_M1_anchor":sum(r["modes"][mode]["map_grid_xy"]!=r["modes"]["M1_anchor"]["map_grid_xy"] for r in rs),
                "fixed_vs_M1_anchor":sum(r["modes"][mode]["hit"] and not r["modes"]["M1_anchor"]["hit"] for r in visible),
                "broken_vs_M1_anchor":sum(not r["modes"][mode]["hit"] and r["modes"]["M1_anchor"]["hit"] for r in visible)}
        groups[split]=group
    history=read_json(run/"epoch_history.json")["records"]
    loss_history=[{"arm":r["arm"],"epoch":r["epoch"],"mean_train_losses":r["mean_train_losses"],
        "batches_with_empty":sum(b["empty_count"]>0 for b in r["batch_losses"]),
        "batches_total":len(r["batch_losses"]),"empty_presentations":sum(b["empty_count"] for b in r["batch_losses"])} for r in history]
    jsonl(output/"loss_history_audit.jsonl",loss_history)
    assert states=={"baseline":state_digest(baseline.state_dict()),**{k:state_digest(a.residual.state_dict()) for k,a in arms.items()}}
    assert all(p.grad is None and not p.requires_grad for a in arms.values() for p in a.parameters())
    assert all(sha256_file(Path(p))==d for p,d in protected.items())
    assert sha256_file(output/"conditioning_predictions.jsonl")==inference_hash
    final={"status":"FROZEN_FAILURE_AUDIT_COMPLETE","groups":groups,"focus_counts":{"dev_misses":13,"train_empty":5,"dev_gains":4},
        "replay_selected_logits_exact":True,"all_656_replay_target_and_anchor_exact":True,
        "diagnostic_saved_before_evaluator_read":True,"weights_and_baseline_unchanged":True,
        "optimizer_or_backward_used":False,"new_checkpoint_selected":False,"v1_gate_unchanged":True,
        "verifier_calibration_iid_ood_accessed":False,"state_sha256":states,
        "interpretation":"M1 target/whole are residual-pooling interventions on unchanged prompts/capture, not new annotated queries or selected models"}
    atomic_json(output/"summary.json",final)
    atomic_json(output/"provenance.json",{"protected_sha256":protected,"summary":final,"source_sha256":sha256_file(Path(__file__))})
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
