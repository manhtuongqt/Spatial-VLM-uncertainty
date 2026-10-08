#!/usr/bin/env python3
"""S0 audit: frozen train/dev anchor inference, then separate oracle evaluation.

No training, calibration, test-IID, metric-depth access or prediction refinement.
Outputs never overwrite an existing audit directory.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrau.checkpoint import load_model_checkpoint
from pcrau.dataset import ArchiveLayout, ArchivedPCRAUDataset
from pcrau.engine import autocast_context
from pcrau.model import PCRAUTargetV2
from pcrau.text import prompt_anchor_mask, relation_ids, tokenize
from pcrau.utils import atomic_json, read_json, runtime_info, seed_everything, sha256_file, workspace_path


HORIZONTAL = {"left_of", "right_of"}
FRUIT = {"apple", "orange", "lemon", "mango"}


def parse_query(prompt: str) -> dict:
    """Prompt-only narrow audit parser; intentionally not a production parser."""
    cores = re.findall(r"\blocate the ([^.]+)\.", prompt.lower())
    if len(cores) != 1:
        return {"status": "unsupported", "reason": "missing_or_multiple_locate_clause"}
    core = cores[0].strip()
    match = re.fullmatch(r"(.+?) that is (left|right) of the (.+)", core)
    if match:
        target, direction, anchor = match.groups()
        if any(re.search(r"\b(and|or|that|between)\b", v) for v in (target, anchor)):
            return {"status": "unsupported", "reason": "unsupported_scope"}
        return {"status": "supported", "target_phrase": target,
                "anchor_phrases": [anchor], "predicate": direction + "_of",
                "reference_frame": "image", "frame_source": "fixed_horizontal_scope"}
    if re.fullmatch(r"[a-z ]+", core) and not re.search(
            r"\b(that|between|nearer|farther|closer|left|right|behind|front)\b", core):
        return {"status": "supported", "target_phrase": core, "anchor_phrases": [],
                "predicate": "direct", "reference_frame": None}
    return {"status": "unsupported", "reason": "outside_direct_horizontal_grammar"}


def self_check() -> None:
    a = parse_query("Please locate the apple that is right of the purple cube. Output (x,y).")
    assert (a["target_phrase"], a["anchor_phrases"], a["predicate"]) == (
        "apple", ["purple cube"], "right_of")
    b = parse_query("locate the purple cube that is left of the apple.")
    assert b["target_phrase"] == "purple cube" and b["anchor_phrases"] == ["apple"]
    assert parse_query("locate the mango that is behind the apple.")["status"] == "unsupported"
    assert parse_query("locate the apple that is right of the cube and left of the lemon.")["status"] == "unsupported"
    assert parse_query("locate the fruit.")["predicate"] == "direct"
    assert parse_query("locate the apple. locate the cube.")["status"] == "unsupported"


def dump_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def map_summary(logits: np.ndarray) -> dict:
    flat = logits.reshape(-1).astype(np.float64)
    probability = np.exp(flat - flat.max())
    probability /= probability.sum()
    h, w = logits.shape
    y, x = divmod(int(probability.argmax()), w)
    ys, xs = np.indices((h, w))
    return {"map_grid_xy": [x, y], "map_pixel_xy": [int((x + .5) * 640 / w), int((y + .5) * 480 / h)],
            "mean_pixel_xy": [float((probability.reshape(h, w) * (xs + .5)).sum() * 640 / w),
                              float((probability.reshape(h, w) * (ys + .5)).sum() * 480 / h)],
            "entropy_normalized": float(-(probability * np.log(probability.clip(1e-300))).sum() / math.log(h*w)),
            "softmax_peak": float(probability.max()), "sigmoid_max": float(1 / (1 + np.exp(-flat.max())))}


def phrase_matches(phrase: str, ids: list[str], registry: dict) -> bool:
    if not ids or any(v not in registry for v in ids):
        return False
    names = {registry[v]["semantic_class"] for v in ids}
    return names.issubset(FRUIT) if phrase == "fruit" else (bool(names) if phrase == "object" else names == {phrase})


def evaluate_mask(mask: np.ndarray, logits: np.ndarray, summary: dict) -> dict:
    assert mask.shape == (480, 640)
    h, w = logits.shape
    small = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    x, y = summary["map_pixel_xy"]
    gx, gy = summary["map_grid_xy"]
    pixels = int(mask.sum())
    flat = logits.reshape(-1).astype(np.float64)
    probability = np.exp(flat - flat.max()); probability /= probability.sum()
    result = {"visible_pixels": pixels, "visible": pixels > 0,
              "peak_hit_full": bool(mask[y, x]) if pixels else None,
              "peak_cell_overlaps": bool(small[gy, gx] > 0) if pixels else None,
              "fractional_mask_weighted_mass": float((probability.reshape(h, w) * small).sum()) if pixels else None}
    if pixels:
        my, mx = np.nonzero(mask)
        result["centroid_pixel_xy"] = [float(mx.mean()), float(my.mean())]
        result["distance_peak_to_mask_px"] = float(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[y, x])
    else:
        result["centroid_pixel_xy"] = None
        result["distance_peak_to_mask_px"] = None
    return result


def aggregate(rows: list[dict]) -> dict:
    active = [r for r in rows if r["parsed"]["predicate"] in HORIZONTAL]
    visible = [r for r in active if r["anchor_eval"]["visible"]]
    empty = [r for r in active if not r["anchor_eval"]["visible"]]
    hits = sum(r["anchor_eval"]["peak_hit_full"] for r in visible)
    cells = sum(r["anchor_eval"]["peak_cell_overlaps"] for r in visible)
    found = [r for r in active if r["truth_answerability"] == "FOUND"]
    both = [r for r in found if r["anchor_eval"]["peak_hit_full"] and r["target_eval"]["peak_hit_full"]]
    def values(rs, key):
        v = [r["anchor_pred"][key] for r in rs]
        return {"n": len(v), "min": min(v) if v else None, "median": float(np.median(v)) if v else None,
                "max": max(v) if v else None}
    return {"samples": len(rows), "families": len({r["family_id"] for r in rows}),
            "active_horizontal": len(active), "visible_anchor": len(visible), "empty_anchor": len(empty),
            "anchor_peak_hits": hits, "anchor_peak_hit_rate": hits/len(visible) if visible else None,
            "anchor_peak_cell_overlaps": cells, "anchor_cell_overlap_rate": cells/len(visible) if visible else None,
            "found": len(found), "found_target_hits": sum(bool(r["target_eval"]["peak_hit_full"]) for r in found),
            "found_both_target_anchor_hits": len(both),
            "annotation_matches": sum(r["annotation_match"] for r in rows),
            "states": dict(Counter(r["truth_answerability"] for r in active)),
            "visible_sigmoid_max": values(visible,"sigmoid_max"), "empty_sigmoid_max": values(empty,"sigmoid_max"),
            "visible_entropy": values(visible,"entropy_normalized"), "empty_entropy": values(empty,"entropy_normalized"),
            "predicate_peak_pass": sum(bool(r.get("predicted_peak_relation_pass")) for r in active),
            "observed_centroid_relation_evaluable": sum(r.get("observed_centroid_relation_pass") is not None for r in active)}


def bootstrap_anchor(rows: list[dict], seed: int = 24082026) -> dict:
    families = defaultdict(list)
    for r in rows:
        if r["parsed"]["predicate"] in HORIZONTAL and r["anchor_eval"]["visible"]:
            families[r["family_id"]].append(r)
    counts = np.array([[sum(r["anchor_eval"]["peak_hit_full"] for r in rs),len(rs)] for _,rs in sorted(families.items())])
    if not len(counts): return {"families": 0}
    rng = np.random.default_rng(seed)
    sampled = counts[rng.integers(len(counts),size=(5000,len(counts)))].sum(1)
    ci = np.quantile(sampled[:,0]/sampled[:,1],[.025,.975])
    return {"families":len(counts),"resamples":5000,"seed":seed,"percentile_ci95":ci.tolist()}


def render_cases(root: Path, rows: list[dict], outputs: dict, layout: ArchiveLayout, protect) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    categories = [
        ("found_binding_hit", lambda r: r["truth_answerability"]=="FOUND" and r["anchor_eval"]["peak_hit_full"] is True),
        ("found_binding_miss", lambda r: r["truth_answerability"]=="FOUND" and r["anchor_eval"]["peak_hit_full"] is False),
        ("missing_anchor_peak", lambda r: not r["anchor_eval"]["visible"]),
        ("insufficient_anchor_miss", lambda r: r["truth_answerability"]=="INSUFFICIENT_EVIDENCE" and r["anchor_eval"]["peak_hit_full"] is False),
    ]
    chosen=[]
    for name, pred in categories:
        candidates=[r for r in rows if r["parsed"]["predicate"] in HORIZONTAL and pred(r)]
        candidates.sort(key=lambda r: (r["split"]!="dev",r["sample_id"]))
        if candidates: chosen.append((name,candidates[0]))
    cases=[]; (root/"cases").mkdir()
    for category,r in chosen:
        rgb_path=layout.dataset_path(r["rgb_path"]); protect(rgb_path)
        rgb=cv2.cvtColor(cv2.imread(str(rgb_path)),cv2.COLOR_BGR2RGB)
        anchor=cv2.imread(str(layout.dataset_path(r["anchor_mask_path"])),0)>0
        target=cv2.imread(str(layout.dataset_path(r["target_mask_path"])),0)>0
        logits=outputs[r["sample_id"]]["anchor_logits"][0]
        p=np.exp(logits.astype(np.float64)-logits.max());p/=p.sum()
        fig,axes=plt.subplots(1,3,figsize=(15,4.4))
        axes[0].imshow(rgb)
        overlay=np.zeros((480,640,4));overlay[anchor]=[0,1,1,.5];overlay[target]=[0,1,0,.4]
        axes[0].imshow(overlay)
        axes[0].scatter(*r["anchor_pred"]["map_pixel_xy"],color="red",marker="x",s=80,label="Pred anchor MAP")
        axes[0].scatter(*r["target_pred"]["map_pixel_xy"],color="yellow",marker="+",s=80,label="Pred target MAP")
        axes[0].legend(fontsize=7);axes[0].set_title("RGB + evaluator masks (cyan anchor / green target)",fontsize=9)
        im=axes[1].imshow(p,origin="upper",extent=(0,640,480,0),interpolation="nearest",vmin=0,vmax=p.max())
        axes[1].set_title(f"Actual anchor softmax | peak={p.max():.4f}",fontsize=9);fig.colorbar(im,ax=axes[1],fraction=.046)
        axes[2].axis("off")
        lines=[category,r["sample_id"],f"split: {r['split']} | truth: {r['truth_answerability']}",
               f"target: {r['parsed']['target_phrase']}",f"anchor: {r['parsed']['anchor_phrases'][0]}",
               f"predicate: {r['parsed']['predicate']} (image)",f"oracle anchor pixels: {r['anchor_eval']['visible_pixels']}",
               f"full-pixel hit: {r['anchor_eval']['peak_hit_full']}",f"cell overlap: {r['anchor_eval']['peak_cell_overlaps']}",
               f"sigmoid max: {r['anchor_pred']['sigmoid_max']:.4f}",f"normalized entropy: {r['anchor_pred']['entropy_normalized']:.4f}",
               "Masks used only after prediction; no verifier applied."]
        axes[2].text(0,1,"\n".join(lines),va="top",fontsize=8)
        for ax in axes[:2]:ax.set_xlim(0,640);ax.set_ylim(480,0)
        fig.tight_layout();path=root/"cases"/(category+".png");fig.savefig(path,dpi=130);plt.close(fig)
        cases.append({"category":category,"sample_id":r["sample_id"],"split":r["split"],"path":str(path.relative_to(root)),
                      "selection":"first lexicographic sample satisfying diagnostic category, dev preferred; illustrative, not representative"})
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="new/outputs/pcrau_s0_anchor_audit_20261005")
    parser.add_argument("--self-check-only",action="store_true")
    args=parser.parse_args();self_check()
    if args.self_check_only:
        print("Prompt-only parser self checks: OK");return
    active_path=workspace_path("new/outputs/active_experimental_profile.json")
    active=read_json(active_path);config_path=Path(active["config"]);config=read_json(config_path)
    output_root=workspace_path(args.output)
    if workspace_path("new/outputs") not in output_root.parents:raise ValueError("Output must be under new/outputs")
    output_root.mkdir(exist_ok=False)
    protected={}
    def protect(path):
        path=Path(path).resolve()
        if str(path) not in protected:protected[str(path)]=sha256_file(path)
        return protected[str(path)]
    for key in ("config","checkpoint","calibrator","profiles"):
        assert protect(active[key])==active[key+"_sha256"]
    protect(active_path);protect(__file__)
    layout=ArchiveLayout(config,"development")
    protect(layout.manifest);protect(layout.feature_index)
    manifest=read_json(layout.manifest)
    entries=manifest["entries"]
    assert len(entries)==2000 and Counter(e["split"] for e in entries)=={"train":1600,"dev":400}
    assert {e['family_id'] for e in entries if e['split']=='train'}.isdisjoint(e['family_id'] for e in entries if e['split']=='dev')
    inventory=Counter((e["split"],e["audit_only"]["relation"],parse_query(e["feature_input"]["prompt"])["status"]) for e in entries)
    family_ids={e["family_id"] for e in entries if parse_query(e["feature_input"]["prompt"]).get("predicate") in HORIZONTAL}
    selected=[e for e in entries if e["family_id"] in family_ids]
    selected.sort(key=lambda e:(e["split"],e["family_id"],e["variant"]))
    assert len(selected)==400 and len(family_ids)==80
    for e in selected:assert parse_query(e["feature_input"]["prompt"])["status"]=="supported"
    datasets={split:ArchivedPCRAUDataset(config,split) for split in ("train","dev")}
    seed_everything(24082026);torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=PCRAUTargetV2(config).to(device).eval().requires_grad_(False)
    metadata=load_model_checkpoint(model,active["checkpoint"],sha256_file(config_path))
    original_state=load_file(active["checkpoint"],device="cpu")
    outputs={};predictions=[]
    for start in range(0,len(selected),8):
        subset=selected[start:start+8];model_samples=[]
        for e in subset:
            data=datasets[e["split"]];key=data.sample_to_feature[e["sample_id"]]
            desc=data.features[key];feature_path=data.layout.feature_root/desc["path"]
            assert protect(feature_path)==desc["sha256"]
            assert desc["rgb_sha256"]==e["feature_input"]["rgb_sha256"] and desc["depth_sha256"]==e["feature_input"]["depth_sha256"]
            tensors=data._feature(e);prompt=e["feature_input"]["prompt"];cfg=config["model"]
            tid,tm=tokenize(prompt,cfg["max_tokens"],cfg["vocab_size"])
            rid,rm=relation_ids(prompt,cfg["max_relations"])
            model_samples.append({"r0":tensors["R0_GRID"].float(),"d0":tensors["D0_GRID"].float(),
                "r_thumb":tensors["R0_THUMB"].float(),"d_thumb":tensors["D0_THUMB"].float(),
                "token_ids":torch.tensor(tid),"token_mask":torch.tensor(tm),"relation_ids":torch.tensor(rid),
                "relation_mask":torch.tensor(rm),"anchor_mask":torch.tensor(prompt_anchor_mask(prompt,cfg["max_anchors"]))})
        batch={key:torch.stack([s[key] for s in model_samples]).to(device) for key in model_samples[0]}
        assert set(batch)==model.MODEL_INPUT_KEYS and not model.training
        with torch.inference_mode(),autocast_context(device,config["optimization"]):out=model(batch)
        for i,e in enumerate(subset):
            o={key:out[key][i].float().cpu().numpy() for key in ("anchor_logits","target_logits")}
            outputs[e["sample_id"]]=o
            predictions.append({"sample_id":e["sample_id"],"prompt":e["feature_input"]["prompt"],
                "parsed":parse_query(e["feature_input"]["prompt"]),"anchor_pred":map_summary(o["anchor_logits"][0]),
                "target_pred":map_summary(o["target_logits"]),
                "predicted_answerability":config["model"]["answerability_classes"][int(out["answerability_logits"][i].argmax())]})
        if start%80==0 or start+8>=len(selected):print(f"frozen train/dev inference {min(start+8,len(selected))}/{len(selected)}",flush=True)
    assert all(torch.equal(original_state[k],v.detach().cpu()) for k,v in model.state_dict().items())
    dump_rows(output_root/"inference_predictions.jsonl",predictions)
    np.savez_compressed(output_root/"spatial_logits.npz",sample_ids=np.array([e["sample_id"] for e in selected]),
                        anchor_logits=np.stack([outputs[e["sample_id"]]["anchor_logits"] for e in selected]),
                        target_logits=np.stack([outputs[e["sample_id"]]["target_logits"] for e in selected]))
    inference_hash=protect(output_root/"inference_predictions.jsonl")
    # Boundary: all model predictions are complete and saved BEFORE reading masks/graphs/registry.
    registry_path=layout.dataset_root/"object_registry.json";protect(registry_path)
    registry={v["id"]:v for v in read_json(registry_path).values()}
    rows=[];by_pred={r["sample_id"]:r for r in predictions}
    for e in selected:
        pred=by_pred[e["sample_id"]];o=outputs[e["sample_id"]]
        record_path=layout.dataset_path(e["record_path"]);assert protect(record_path)==e["record_sha256"]
        record=read_json(record_path);ev=record["evaluator_only"];sp=ev["spatial_label"];parsed=pred["parsed"]
        target_path=layout.dataset_path(e["supervision"]["target_mask_path"])
        assert protect(target_path)==e["supervision"]["target_mask_sha256"]
        target=cv2.imread(str(target_path),0)>0
        target_eval=evaluate_mask(target,o["target_logits"],pred["target_pred"])
        horizontal=parsed["predicate"] in HORIZONTAL
        annotation_match=phrase_matches(parsed["target_phrase"],sp["candidate_target_ids"],registry)
        annotation_match &= sp["relations"]==[parsed["predicate"]]
        anchor_eval=None;anchor_path=None
        if horizontal:
            assert len(sp["anchor_ids"])==len(ev["masks"]["anchor"])==len(e["supervision"]["anchor_masks"])==1
            d=ev["masks"]["anchor"][0];assert d["object_id"]==sp["anchor_ids"][0]
            anchor_path=layout.dataset_root/d["path"]
            assert protect(anchor_path)==d["sha256"]==e["supervision"]["anchor_masks"][0]["sha256"]
            assert anchor_path.resolve()==layout.dataset_path(e["supervision"]["anchor_masks"][0]["path"])
            anchor=cv2.imread(str(anchor_path),0)>0
            anchor_eval=evaluate_mask(anchor,o["anchor_logits"][0],pred["anchor_pred"])
            annotation_match &= phrase_matches(parsed["anchor_phrases"][0],sp["anchor_ids"],registry)
            graph=sp["relation_graph"]
            # Graph encodes resolved valid targets; ABSENT has source_ids=[],
            # while candidate_target_ids still describes the requested noun.
            annotation_match &= len(graph)==1 and graph[0]["target_ids"]==sp["anchor_ids"] and graph[0]["source_ids"]==sp["valid_target_ids"] and graph[0]["predicate"]==parsed["predicate"] and sp["reference_frame"]==graph[0]["reference_frame"]=="image"
        else:annotation_match &= sp["anchor_ids"]==[] and e["supervision"]["anchor_masks"]==[]
        row={**pred,"split":e["split"],"family_id":e["family_id"],"variant":e["variant"],
             "truth_answerability":e["supervision"]["answerability_state"],"truth_submode":ev["uncertainty_label"]["state_submode"],
             "annotation_match":bool(annotation_match),"oracle_anchor_ids":sp["anchor_ids"],
             "oracle_candidate_target_ids":sp["candidate_target_ids"],"anchor_eval":anchor_eval,"target_eval":target_eval,
             "record_path":str(record_path.relative_to(workspace_path('.'))),"record_sha256":e["record_sha256"],
             "rgb_path":e["feature_input"]["rgb_path"],"anchor_mask_path":str(anchor_path.relative_to(layout.dataset_root)) if anchor_path else None,
             "target_mask_path":str(target_path.relative_to(layout.dataset_root))}
        if horizontal:
            sign=1 if parsed["predicate"]=="right_of" else -1
            delta=sign*(pred["target_pred"]["map_pixel_xy"][0]-pred["anchor_pred"]["map_pixel_xy"][0])-12
            row["predicted_peak_signed_margin_px"]=delta;row["predicted_peak_relation_pass"]=delta>0
            tc=target_eval["centroid_pixel_xy"];ac=anchor_eval["centroid_pixel_xy"]
            row["observed_centroid_relation_pass"]=sign*(tc[0]-ac[0])-12>0 if tc and ac else None
            row["predicted_anchor_peak_inside_target"]=bool(target[pred["anchor_pred"]["map_pixel_xy"][1],pred["anchor_pred"]["map_pixel_xy"][0]])
        rows.append(row)
    dump_rows(output_root/"evaluator_rows.jsonl",rows)
    fields=["sample_id","split","family_id","variant","truth","target_phrase","anchor_phrase","predicate","annotation_match","visible_pixels","anchor_map_xy","anchor_hit","cell_overlap","distance_px","sigmoid_max","entropy","anchor_peak_in_target","predicate_peak_pass"]
    with (output_root/"anchor_rows.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            ae=r["anchor_eval"] or {}
            writer.writerow(dict(zip(fields,[r['sample_id'],r['split'],r['family_id'],r['variant'],r['truth_answerability'],r['parsed']['target_phrase']," | ".join(r['parsed']['anchor_phrases']),r['parsed']['predicate'],r['annotation_match'],ae.get('visible_pixels'),r['anchor_pred']['map_pixel_xy'],ae.get('peak_hit_full'),ae.get('peak_cell_overlaps'),ae.get('distance_peak_to_mask_px'),r['anchor_pred']['sigmoid_max'],r['anchor_pred']['entropy_normalized'],r.get('predicted_anchor_peak_inside_target'),r.get('predicted_peak_relation_pass')])))
    all_groups={"all":aggregate(rows)}
    for split in ('train','dev'):all_groups[split]=aggregate([r for r in rows if r['split']==split])
    for key in ('variant','truth_answerability'):
        for name in sorted({r[key] for r in rows}):
            for split in ('train','dev'):all_groups[f'{split}/{key}/{name}']=aggregate([r for r in rows if r[key]==name and r['split']==split])
    horizontal_rows=[r for r in rows if r['anchor_eval']]
    by_id={r['sample_id']:r for r in rows};pairs=[]
    for fid in sorted(family_ids):
        a,b=by_id[fid+'__clean'],by_id[fid+'__relation_counterfactual']
        pairs.append({'family_id':fid,'split':a['split'],'both_visible':a['anchor_eval']['visible'] and b['anchor_eval']['visible'],
                      'both_anchor_hits':a['anchor_eval']['peak_hit_full'] is True and b['anchor_eval']['peak_hit_full'] is True,
                      'same_predicted_anchor_map':a['anchor_pred']['map_pixel_xy']==b['anchor_pred']['map_pixel_xy'],
                      'anchor_phrase_changed':a['parsed']['anchor_phrases']!=b['parsed']['anchor_phrases'],
                      'same_cached_features':datasets[a['split']].sample_to_feature[a['sample_id']]==datasets[b['split']].sample_to_feature[b['sample_id']]})
    dump_rows(output_root/'swap_pairs.jsonl',pairs)
    cases=render_cases(output_root,rows,outputs,layout,protect)
    bootstrap={split:bootstrap_anchor([r for r in rows if r['split']==split]) for split in ('train','dev')}
    state_hashes_unchanged=all(sha256_file(Path(p))==v for p,v in protected.items())
    assert state_hashes_unchanged
    summary={'status':'S0_TRAIN_DEV_FROZEN_AUDIT_COMPLETE','active_selection':active['selection'],
        'checkpoint_sha256':active['checkpoint_sha256'],'config_sha256':active['config_sha256'],
        'runtime':runtime_info(),'model_input_keys':sorted(model.MODEL_INPUT_KEYS),
        'inference_before_oracle_evaluation':True,'inference_predictions_sha256':inference_hash,
        'model_state_unchanged':True,'protected_inputs_unchanged':state_hashes_unchanged,
        'calibration_access':'hash only; no samples/fit','test_accessed':False,'metric_depth_accessed':False,
        'training':False,'mc_dropout':False,'verifier_applied':False,'metadata_epoch':metadata['epoch'],
        'inventory':[{'split':a,'annotation_relation':b,'parse_status':c,'count':n} for (a,b,c),n in sorted(inventory.items())],
        'groups':all_groups,'anchor_hit_family_bootstrap':bootstrap,'cases':cases,
        'swap_pairs':{split:{'pairs':len(ps),'both_visible':sum(p['both_visible'] for p in ps),
                            'both_hits':sum(p['both_anchor_hits'] for p in ps if p['both_visible']),
                            'same_peak':sum(p['same_predicted_anchor_map'] for p in ps),
                            'same_cached_features':sum(p['same_cached_features'] for p in ps)}
                      for split in ('train','dev') for ps in [[p for p in pairs if p['split']==split]]}}
    atomic_json(output_root/'summary.json',summary);atomic_json(output_root/'provenance.json',{'protected_sha256_before_after':protected,'unchanged':True})
    print(json.dumps({'groups':{k:all_groups[k] for k in ('train','dev')},'swap_pairs':summary['swap_pairs'],'bootstrap':bootstrap,'cases':cases},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
