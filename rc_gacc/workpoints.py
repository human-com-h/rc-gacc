import math
import re
from collections import Counter

import numpy as np

from .stage3_decision import decide, render

def parent_out(row,d,s,c,budget):
    x=render(d,s,[],budget)
    assert x['within_budget']
    return dict(x,selected=s,solver=c,record_id=row['record_id'],group_id=row['group_id'],split=row['split'])

def binding_after(row,d,result,edges,u,cov,cal):
    flat=render(d,result['selected'],[],result['budget'])
    b=decide(row,d,dict(result,tokens=flat['tokens']),u,cov,edges,dict(cal,tau_select=float('inf')))
    for k in ('reader_text','tokens','bindings','labels'):result[k]=b[k]
    return result

def bm25(units,goal):
    from rc_gacc.anchor_satellite import terms
    docs=[re.findall(r'\w+',u['text'].lower()) for u in units]
    q=sorted(terms(goal));n=len(docs);avg=np.mean([len(t) for t in docs]) or 1
    tf=[Counter(t) for t in docs];df={t:sum(t in v for v in tf) for t in q}
    return np.asarray([sum(math.log(1+(n-df[t]+.5)/(df[t]+.5))*v[t]*2.2/
                (v[t]+1.2*(.25+.75*len(docs[i])/avg)) for t in q if v[t]) for i,v in enumerate(tf)])
