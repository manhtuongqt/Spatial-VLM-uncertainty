"""V2 dispatch, optimizer lock, selection and paired-gate regression checks."""
from copy import deepcopy
from pathlib import Path
import runpy
import tempfile
import json

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from pcrau.anchor_peak_experiment import locked_peak_config, prepare_peak_arms, supervision_from_full_masks
from pcrau.anchor_peak_runner import PeakPilotRunner
from pcrau.anchor_peak_pilot import aggregate, family_bootstrap, locked_gates, pair_rows
from pcrau.anchor_shadow_experiment import anchor_shadow_loss, state_digest


def setup():
    fixture = runpy.run_path(str(Path(__file__).with_name('test_anchor_shadow.py')))
    model,batch,prompts = fixture['setup']()
    arms=prepare_peak_arms(model)
    receipt={'status':'S1_PEAK_V2_PREFLIGHT_PASS','technical_gates_pass':True,
        'config_sha256':'config','protocol_sha256':'protocol',
        'initialization_sha256':{k:state_digest(a.branch.residual.state_dict()) for k,a in arms.items()},
        'frozen_model_state_sha256':state_digest(model.state_dict())}
    execution={'pre_optimizer_checks_pass':True,'training_authorized':True,
        'config':json.loads(json.dumps(locked_peak_config())),'config_sha256':'config','protocol_sha256':'protocol'}
    return model,batch,prompts,arms,receipt,execution


def test_v2_actual_epoch_dispatch_freeze_and_checkpoint_reload():
    model,batch,prompts,arms,receipt,execution=setup()
    cap=arms['C1'].branch.capture(batch,prompts)
    masks=np.zeros((3,3,2,3),dtype=bool);masks[0,0,0,0]=True
    sup=supervision_from_full_masks(masks,(2,3))
    presentations=[{'entry':{'split':'train'},'family_id':'toy','sample_id':'toy'}]
    for name,arm in arms.items():
        runner=PeakPilotRunner(arm);runner.start_optimizer(receipt,execution)
        ids={id(p) for g in runner.optimizer.param_groups for p in g['params']}
        assert ids=={id(p) for p in arm.branch.residual.parameters()}
        if name=='C1':
            z=arm.branch.predict(cap)
            assert torch.equal(arm.loss(z,sup,cap.active_slots)['total'],anchor_shadow_loss(z,sup.area_masks,cap.active_slots)['total'])
        history=runner.run_epoch(0,presentations,lambda _: (cap,sup))
        assert ('peak' in history[0])==(name!='C1') and runner.optimizer_steps==1
        assert state_digest(arm.branch.residual.state_dict())!=receipt['initialization_sha256'][name]
        assert state_digest(model.state_dict())==receipt['frozen_model_state_sha256']
        assert all(p.grad is None for p in model.parameters())
        runner.record_dev_selection(0,49,8,.5)
        with tempfile.TemporaryDirectory(prefix='pcrau_peak_runner_test_') as directory:
            path=Path(directory)/'mlp.safetensors'
            save_file(arm.branch.residual.state_dict(),str(path))
            expected=arm.branch.predict(cap).detach()
            with torch.no_grad():arm.branch.residual[2].bias.add_(1)
            arm.branch.residual.load_state_dict(load_file(str(path)),strict=True)
            torch.testing.assert_close(expected,arm.branch.predict(cap),rtol=0,atol=0)


def test_optimizer_rejects_missing_auth_drifted_hash_config_and_init():
    model,batch,prompts,arms,receipt,execution=setup()
    cases=[]
    for field,value in [('training_authorized',False),('pre_optimizer_checks_pass',False),('config_sha256','drift')]:
        e=deepcopy(execution);e[field]=value;cases.append((receipt,e))
    e=deepcopy(execution);e['config']['objective']['margin']=2;cases.append((receipt,e))
    r=deepcopy(receipt);r['initialization_sha256']['P1']='wrong';cases.append((r,execution))
    r=deepcopy(receipt);r['frozen_model_state_sha256']='wrong';cases.append((r,execution))
    for r,e in cases:
        runner=PeakPilotRunner(arms['P1'])
        try:runner.start_optimizer(r,e)
        except ValueError:pass
        else:raise AssertionError('Invalid optimizer receipt accepted')
        assert runner.optimizer is None


def test_lexicographic_selection_ties_and_patience():
    _,_,_,arms,_,_=setup();runner=PeakPilotRunner(arms['P1'])
    for epoch in range(6):
        runner.next_epoch=epoch+1;runner.pending_selection_epoch=epoch
        improved=runner.record_dev_selection(epoch,49,8,.5)
        assert improved==(epoch==0)
    assert runner.best_epoch==0 and runner.stale_epochs==5
    try:runner.run_epoch(6,[],None)
    except RuntimeError:pass
    else:raise AssertionError('Early-stop bypassed')


def rows_fixture():
    old=runpy.run_path(str(Path(__file__).with_name('test_anchor_shadow_pilot.py')))['rows_fixture']()
    for r in old:
        m=r['models'];r['models']={'M0':m['M0'],'C1':deepcopy(m['M0']),'P1':m['M1'],'P2':m['M2']}
    return old


def test_peak_gates_require_objective_control_and_each_empty():
    rows=rows_fixture();assert locked_gates(rows,True,True)['pass']
    for r in rows:r['models']['C1']=deepcopy(r['models']['P1'])
    assert not locked_gates(rows,True,True)['gates']['peak_objective_vs_C1']
    rows=rows_fixture();empty=[r for r in rows if not r['anchor_pixels']]
    empty[0]['models']['P1']['sigmoid_max']=.80001
    for r in empty[1:]:r['models']['P1']['sigmoid_max']=.3
    assert not locked_gates(rows,True,True)['gates']['empty_each_no_increase_1e_6']


def test_four_arm_pairing_and_family_bootstrap_controls():
    rows=rows_fixture();rows[1]['parsed']['anchors'][0]['text']='cube'
    assert sum(p['eligible'] for p in pair_rows(rows))==15
    m=aggregate(rows);assert m['models']['P1']['anchor_hits']==49
    for comparison in [('P1','C1'),('P1','P2')]:
        a=family_bootstrap(rows,*comparison);b=family_bootstrap(rows,*comparison)
        assert a==b and a['families']==16 and a['resamples']==5000
