#!/usr/bin/env python3
"""Audit existing captures; add reference-completion supervision, never observations."""
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcrau.utils import sha256_file,atomic_json

OUT=Path('new/outputs/pcrau_task_uncertainty_20261007')
ARCH=Path('old/roborefer_dataset_v2_1_1_development_400_20260824')

def prepare():
    manifest=Path('old/protocol/pcra_u_development_train_manifest.json')
    catalog=Path('old/protocol/dataset_v2_1_shutdown_gate_capture_plan.json')
    registry=json.loads(catalog.read_text())['object_registry']
    labels={v['id']:int(v['label']) for v in registry.values()}
    classes=['background']+sorted({v['semantic_class'] for v in registry.values()})
    label_class={str(v['label']):classes.index(v['semantic_class']) for v in registry.values()}
    records={}; protected={str(manifest.resolve()):sha256_file(manifest),str(catalog.resolve()):sha256_file(catalog)}
    for e in json.loads(manifest.read_text())['entries']:
        parts=Path(e['record_path']).parts;i=parts.index(ARCH.name);p=ARCH.joinpath(*parts[i+1:])
        records[e['family_id'],e['variant']]=(e,json.loads(p.read_text()))
        protected[str(p.resolve())]=sha256_file(p)
    def descriptor(d):
        p=ARCH/d['path']; h=sha256_file(p)
        assert h==d['sha256'],p
        protected[str(p.resolve())]=h
        return {'path':str(p.resolve()),'sha256':h}
    observations=[]; pairs=[]; counts=Counter()
    for (fam,var),(e,r) in records.items():
        sem=descriptor(r['evaluator_only']['semantic_instance_labels'])
        depth=descriptor(r['sensor_evidence']['depth_metric'])
        rgb=descriptor(r['inference_payload']['rgb_model_input'])
        observations.append({'sample_id':e['sample_id'],'family_id':fam,'split':e['split'],
            'instance_labels_supervision_only':sem,'reference_metric_depth_supervision_only':depth,
            'rgb_model_input':rgb,'source_variant_evaluator_only':var})
        if var!='occlusion_view_counterfactual':continue
        ec,c=records[fam,'clean']; ids=c['evaluator_only']['spatial_label']['valid_target_ids']
        occluder=r['provenance']['perturbation'].get('occluder_id')
        okeys=[k for k,v in registry.items() if v['id']==occluder]
        if c['evaluator_only']['uncertainty_label']['state']!='FOUND' or len(ids)!=1 or ids[0]==occluder:
            counts['excluded_nonunique_reference']+=1;continue
        def load(d):return json.loads((ARCH/d['path']).read_text())
        if any(load(c['sensor_evidence'][k])!=load(r['sensor_evidence'][k]) for k in ('camera_info','tf_snapshot')):
            counts['excluded_camera_or_tf']+=1;continue
        x=c['evaluator_only']['object_oracle']['requested_layout_base_link']
        y=r['evaluator_only']['object_oracle']['requested_layout_base_link']
        if any(k not in okeys for k in x.keys()|y.keys() if x.get(k)!=y.get(k)):
            counts['excluded_other_layout_change']+=1;continue
        A=cv2.imread(str(ARCH/c['evaluator_only']['semantic_instance_labels']['path']),cv2.IMREAD_UNCHANGED)==labels[ids[0]]
        B=cv2.imread(sem['path'],cv2.IMREAD_UNCHANGED)==labels[ids[0]]
        dil=cv2.dilate(A.astype('uint8'),np.ones((5,5),'uint8'))>0
        core=cv2.erode(A.astype('uint8'),np.ones((5,5),'uint8'))>0
        if not A.any() or (B&~dil).any():
            counts['excluded_non_subset_or_empty_reference']+=1;continue
        hidden=core&~B
        pair={'sample_id':e['sample_id'],'family_id':fam,'split':e['split'],
              'clean_sample_id':ec['sample_id'],'reference_label_id_supervision_only':labels[ids[0]],
              'reference_semantic_labels_supervision_only':descriptor(c['evaluator_only']['semantic_instance_labels']),
              'occluded_semantic_labels_supervision_only':sem,
              'reference_rgb':descriptor(c['inference_payload']['rgb_model_input']), 'occluded_rgb':rgb,
              'reference_pixels':int(A.sum()),'observed_pixels':int(B.sum()),'hidden_core_pixels':int(hidden.sum()),
              'hidden_core_fraction':float(hidden.sum()/A.sum()),
              'completion_target':'reference_visible_support_not_full_amodal_object',
              'camera_info_and_tf_equal':True,'only_occluder_requested_pose_changed':True,
              'registration_boundary_tolerance_px':2,'observed_subset_of_reference_dilation':True}
        pairs.append(pair);counts['completion_pairs_'+e['split']]+=1
        if pair['hidden_core_fraction']>=.02:counts['meaningful_hidden_pairs_'+e['split']]+=1
    OUT.joinpath('data').mkdir(exist_ok=False)
    atomic_json(OUT/'data/supplement.json',{'classes':classes,'label_class':label_class,
        'observations':observations,'completion_pairs':pairs,'counts':dict(counts),
        'new_photos_or_family_split_changes':False,'annotation_use':'train_loss_or_evaluator_only',
        'no_physical_causal_decomposition_claim':True,'archive_hashes':protected})
    chosen=[]
    for split in ('train','dev'):
        chosen.extend(sorted([p for p in pairs if p['split']==split and p['hidden_core_fraction']>=.02],
                              key=lambda p:-p['hidden_core_fraction'])[:3])
    fig,axes=plt.subplots(len(chosen),3,figsize=(14,3.8*len(chosen)),squeeze=False)
    for row,p in enumerate(chosen):
        ref=cv2.cvtColor(cv2.imread(p['reference_rgb']['path']),cv2.COLOR_BGR2RGB)
        occ=cv2.cvtColor(cv2.imread(p['occluded_rgb']['path']),cv2.COLOR_BGR2RGB)
        A=cv2.imread(p['reference_semantic_labels_supervision_only']['path'],cv2.IMREAD_UNCHANGED)==p['reference_label_id_supervision_only']
        B=cv2.imread(p['occluded_semantic_labels_supervision_only']['path'],cv2.IMREAD_UNCHANGED)==p['reference_label_id_supervision_only']
        core=cv2.erode(A.astype('uint8'),np.ones((5,5),'uint8'))>0
        axes[row,0].imshow(ref);axes[row,0].contour(A,levels=[.5],colors=['lime'],linewidths=1)
        axes[row,1].imshow(occ)
        axes[row,2].imshow(occ)
        overlay=np.zeros((480,640,4));overlay[B]=[0,1,0,.45];overlay[core&~B]=[1,0,1,.7]
        axes[row,2].imshow(overlay)
        for ax in axes[row]:ax.axis('off')
        axes[row,0].set_title(p['family_id']+' / '+p['split']+' / reference')
        axes[row,1].set_title('Existing physical occlusion capture')
        axes[row,2].set_title('Green: observed; magenta: hidden reference core\n'+f"{p['hidden_core_pixels']} hidden pixels / {p['reference_pixels']} reference pixels")
    fig.suptitle('Supplementary labels from existing photos — reference completion, not full amodal truth',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(OUT/'data/occlusion_label_preview.png',dpi=130);plt.close(fig)
    atomic_json(OUT/'data/preview_cases.json',chosen)
    print(json.dumps({'counts':dict(counts),'classes':classes,'preview':str(OUT/'data/occlusion_label_preview.png')}))

if __name__=='__main__':prepare()
