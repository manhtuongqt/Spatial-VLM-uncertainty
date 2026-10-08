from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pcrau.selective_experiment import (
    apply_risk, choose_policy_threshold, experimental_action, fit_risk, mine_hard_examples, policy_result, weighted_answer_loss,
)


def row(index, truth="FOUND", predicted="FOUND", inside=True):
    names=["FOUND","AMBIGUOUS","ABSENT","INSUFFICIENT_EVIDENCE"]
    probabilities={name:0.01 for name in names};probabilities[predicted]=0.97
    return {"sample_id":f"s{index}","family_id":f"f{index//5}",
            "answerability_probabilities":probabilities,
            "source_probabilities":{name:0.1 for name in ("semantic","relation","spatial","depth","occlusion")},
            "spatial":{"entropy_normalized":0.1,"peak_margin":0.5,"mode_count":1},
            "relation_consistency":0.5,"fusion_gate_mean":0.5,"rgb_depth_cosine":0.8,"rgb_depth_mae":0.2,
            "evaluation":{"answerability_state":truth,"map_inside_target":inside,
                          "error_event":truth!="FOUND" or not inside}}


def test_mining_uses_success_event_and_keeps_localization_failures_found():
    rows=[row(0,predicted="ABSENT"),row(1),row(2,truth="AMBIGUOUS",inside=True),
          row(3,inside=False),row(4,truth="ABSENT",predicted="ABSENT")]
    positive,negative,location=mine_hard_examples(rows,np.array([0.1,0.3,0.2,0.1,0.01]),0.2)
    assert positive.tolist()==[True,True,False,False,False]
    assert negative.tolist()==[False,False,True,False,True]
    assert location.tolist()==[False,False,False,True,False]
    assert rows[3]["evaluation"]["answerability_state"]=="FOUND"


def test_unit_example_weights_match_original_weighted_cross_entropy():
    logits=torch.tensor([[2.,0.,1.,0.],[0.,1.,2.,0.]],requires_grad=True)
    labels=torch.tensor([0,2]);weights=torch.tensor([0.5,2.,1.5,1.])
    loss=weighted_answer_loss(logits,labels,weights,torch.ones(2))
    torch.testing.assert_close(loss,F.cross_entropy(logits,labels,weight=weights))
    loss.backward();assert torch.isfinite(logits.grad).all()


def test_threshold_accounts_for_answer_gate_and_none_rejects_all():
    risk=np.array([0.01]*80+[0.02]*20)
    error=np.array([False]*80+[True]*20)
    families=[f"f{i//5}" for i in range(100)]
    gate=np.array([True]*80+[False]*20)
    selected=choose_policy_threshold(risk,error,families,gate,0.05,10)
    assert selected["accepted_samples"]==80 and selected["errors"]==0 and selected["coverage"]==0.8
    failed=choose_policy_threshold(risk,error,families,np.zeros(100,dtype=bool),0.05,10)
    assert failed["threshold"] is None
    rows=[row(0,predicted="ABSENT"),row(1,truth="ABSENT")]
    metrics,records=policy_result(rows,np.array([0.01,0.8]),"risk_only",0.05)
    assert metrics["accepted"]==1 and metrics["bypass_correct"]==1 and metrics["bypass_errors"]==0
    assert records[0]["action"]=="EXECUTE"
    metrics,_=policy_result(rows,np.array([0.01,0.01]),"risk_only",None)
    assert metrics["accepted"]==0


def test_crossfit_keeps_families_whole_and_apply_ignores_evaluation_labels():
    rows=[row(i,truth="ABSENT" if i%3==0 else "FOUND",predicted="ABSENT" if i%3==0 else "FOUND") for i in range(50)]
    fitted,crossfit=fit_risk(rows)
    assert len(fitted["family_folds"])==10 and np.isfinite(crossfit).all()
    risk=apply_risk(rows,fitted)
    without_labels=[{key:value for key,value in item.items() if key!="evaluation"} for item in rows]
    np.testing.assert_array_equal(risk,apply_risk(without_labels,fitted))


def test_runtime_policy_does_not_need_annotation_or_metadata():
    item=row(0,predicted="ABSENT")
    deployable={key:value for key,value in item.items() if key not in {"evaluation","sample_id","family_id"}}
    assert experimental_action(deployable,0.01,"risk_only",0.05)=="EXECUTE"
    assert experimental_action(deployable,0.01,"hard_found",0.05)=="ABSTAIN"
    assert experimental_action(deployable,0.01,"risk_only",None)=="ABSTAIN"


def test_optional_ranker_feature_is_prediction_only_and_backwards_compatible():
    from pcrau.selective_experiment import vector
    item = row(0)
    baseline = vector(item)
    item['ranker_error_logit'] = -1.5
    expanded = vector(item, ranker_feature=True)
    np.testing.assert_array_equal(expanded[:-1], baseline)
    assert expanded[-1] == -1.5
    without_labels = {k:v for k,v in item.items() if k != 'evaluation'}
    np.testing.assert_array_equal(expanded, vector(without_labels, ranker_feature=True))
