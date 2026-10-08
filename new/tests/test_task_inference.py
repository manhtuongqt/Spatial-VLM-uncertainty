import numpy as np
from scipy.special import ndtr
from pcrau.task_inference import mixture_interval,summarize_samples
from pcrau.query_parser import parse_query

def test_relation_float32_endpoint_and_scope():
    classes=['background','apple','purple cube']+[f'class{i}' for i in range(19)]
    t=np.zeros((20,1,768),dtype=np.float32);t[:,:,15]=.6;t[:,:,16]=.40000006
    a=np.zeros_like(t);a[:,:,0]=1
    raw={'target':t,'anchor':a,'semantic_target':np.full((20,1,22),1/22,dtype=np.float32),
         'semantic_anchor':np.full((20,1,22),1/22,dtype=np.float32),
         'depth_mean':np.full((20,1),.6),'depth_variance':np.full((20,1),.01),
         'completion':np.full((20,1,768),1/768,dtype=np.float32)}
    base=[{'spatial':{'map_pixel_xy':[330,10]}}]
    q=parse_query('Locate the apple that is right of the purple cube.')
    rows,out=summarize_samples(raw,base,[q],classes,np.zeros((1,768)))
    assert np.allclose(out['relation'],1) and rows[0]['task_uncertainty']['relation']['supported']
    q=parse_query('Locate the apple behind the purple cube.')
    rows,_=summarize_samples(raw,base,[q],classes,np.zeros((1,768)))
    assert not rows[0]['task_uncertainty']['relation']['supported']
    assert rows[0]['task_uncertainty']['relation']['MI'] is None

def test_exact_mixture_interval_quantiles():
    m=np.array([[.5,.7],[.8,.9]]);v=np.array([[.01,.02],[.03,.04]])
    lo,hi=mixture_interval(m,v)
    assert np.allclose(ndtr((lo[None]-m)/np.sqrt(v)).mean(0),.025,atol=1e-12)
    assert np.allclose(ndtr((hi[None]-m)/np.sqrt(v)).mean(0),.975,atol=1e-12)
