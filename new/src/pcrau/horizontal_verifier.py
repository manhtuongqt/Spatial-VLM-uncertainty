"""Observable image-frame diagnostic; no identity/presence certification."""
from __future__ import annotations

import math
import numpy as np

from .query_parser import parse_query

VERSION = 'pcrau_horizontal_verifier_evidence_v1'
ANCHOR_FEATURES = ['horizontal_supported','direct_bypass','parse_unsupported',
    'anchor_max_logit','anchor_sigmoid_max','anchor_entropy','anchor_spatial_peak','anchor_shift']
GEOMETRY_FEATURES = ['signed_peak_margin_normalized','peak_compatible','pair_compatibility']
EXTRA_FEATURES = ANCHOR_FEATURES + GEOMETRY_FEATURES


def _array(value):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=(24,32) or not np.isfinite(a).all():
        raise ValueError('Expected finite observable24x32map')
    return a


def _point(a):
    y,x=divmod(int(a.argmax()),32)
    return [20*x+10,20*y+10]


def verify(prompt, target_probability_grid, anchor_logits=None, old_anchor_logits=None):
    """No kwargs/annotation channel; arrays are read-only, MAP is never changed."""
    target=_array(target_probability_grid)
    if (target<0).any() or not np.isclose(target.sum(),1,atol=1e-5,rtol=0):
        raise ValueError('Target grid must be a normalized model distribution')
    query=parse_query(prompt)
    features={k:0. for k in EXTRA_FEATURES}
    trace={'version':VERSION,'query':query.to_dict(),'binding_status':'UNVERIFIED',
        'presence_status':'UNVERIFIED','geometry_representation':'predicted_grid_peaks_and_distribution_pairs',
        'margin_px':12,'target_map_changed':False,'target_peak_xy':_point(target),
        'anchor_peak_xy':None,'signed_margin_px':None,'pair_compatibility':None,
        'failure_flags':['binding_unverified','presence_unverified']}
    if not query.supported:
        features['parse_unsupported']=1.
        trace.update(scope_status='PARSE_UNSUPPORTED');return {'features':features,**trace}
    if query.predicate=='direct':
        features['direct_bypass']=1.
        trace.update(scope_status='DIRECT_BYPASS');return {'features':features,**trace}
    if query.predicate not in {'left_of','right_of'} or query.reference_frame!='image' or len(query.anchors)!=1:
        raise ValueError('Parser/verifier scope contract mismatch')
    anchor,old=_array(anchor_logits),_array(old_anchor_logits)
    features['horizontal_supported']=1.
    prob=np.exp(anchor-anchor.max());prob/=prob.sum()
    tx,ty=trace['target_peak_xy'];ax,ay=_point(anchor);ox,oy=_point(old)
    sign=1 if query.predicate=='right_of' else -1
    signed=sign*(tx-ax)-12
    xs=np.arange(32)*20+10
    kernel=sign*(xs[:,None]-xs[None,:])-12>0
    mass=float(target.sum(0)@kernel@prob.sum(0))
    maximum=float(anchor.max())
    sigmoid=float(np.exp(-np.logaddexp(0,-maximum)))
    features.update(anchor_max_logit=maximum,anchor_sigmoid_max=sigmoid,
        anchor_entropy=float(-(prob*np.log(np.maximum(prob,1e-300))).sum()/math.log(768)),
        anchor_spatial_peak=float(prob.max()),anchor_shift=math.hypot(ax-ox,ay-oy)/800,
        signed_peak_margin_normalized=signed/640,peak_compatible=float(signed>0),pair_compatibility=mass)
    if maximum<=0:trace['failure_flags'].append('nonpositive_anchor_peak_logit')
    trace.update(scope_status='SUPPORTED_DIAGNOSTIC',anchor_peak_xy=[ax,ay],
        signed_margin_px=signed,pair_compatibility=mass,
        anchor_evidence={k:features[k] for k in ANCHOR_FEATURES[3:]})
    return {'features':features,**trace}
