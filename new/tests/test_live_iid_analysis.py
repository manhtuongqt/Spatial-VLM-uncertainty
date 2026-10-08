"""Meaningful paired accounting/bootstrap invariants without training."""
import copy
from pcrau.live_iid_analysis import paired_changes, paired_family_bootstrap


def fixture():
    primary=[];reference=[]
    for i,(aa,ba,error) in enumerate([(1,0,0),(0,1,0),(1,0,1),(0,1,1),(1,1,0)]):
        row={'sample_id':str(i),'family_id':'f'+str(i//2),'variant':'clean',
             'evaluation':{'error_event':bool(error),'answerability_state':'FOUND' if not error else 'ABSENT'},
             'verifiers':{'P1':{'scope_status':'SUPPORTED_DIAGNOSTIC'}}}
        primary.append({**row,'decision':{'action':'EXECUTE' if aa else 'REOBSERVE','risk':.1}})
        reference.append({**row,'decision':{'action':'EXECUTE' if ba else 'REOBSERVE','risk':.2}})
    return primary,reference


def test_paired_changes_reports_four_directions_and_no_mutation():
    a,b=fixture();before=copy.deepcopy((a,b))
    result,rows=paired_changes(a,b)
    assert result['counts']=={'new_correct_accept':1,'removed_correct_accept':1,
                             'new_error_accept':1,'removed_error_accept':1}
    assert result['changed_acceptance']==4 and len(rows)==4
    assert (a,b)==before


def test_family_bootstrap_determinism_and_identical_arm_zero():
    a,b=fixture()
    x=paired_family_bootstrap(a,b,resamples=200)
    assert x==paired_family_bootstrap(a,b,resamples=200)
    same=paired_family_bootstrap(a,a,resamples=200)
    for name, m in same['metrics'].items():
        assert m['point']==0.
        if m['valid_resamples']: assert m['ci95']==[0.,0.]
    reverse=paired_family_bootstrap(b,a,resamples=200)
    for name,m in x['metrics'].items():
        assert abs(m['point']+reverse['metrics'][name]['point'])<1e-12
        if m['valid_resamples']:
            assert abs(m['ci95'][0]+reverse['metrics'][name]['ci95'][1])<1e-12


def test_mismatched_pair_rejected():
    a,b=fixture();b[0]['sample_id']='different'
    for fn in (paired_changes,paired_family_bootstrap):
        try:fn(a,b)
        except ValueError:continue
        raise AssertionError('Wrong paired join accepted')
